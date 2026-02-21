import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def ray_partition(x, line_size):
    """Reshape [N_ray * N_samples, c] -> [N_ray * N_samples // line_size, line_size, c]."""
    n, c = x.shape
    return x.view(n // line_size, line_size, c)


def ray_merge(x):
    """Reshape [N_ray * N_samples // line_size, line_size, c] -> [N_ray * N_samples, c]."""
    line_batch_num, line_size, c = x.shape
    return x.view(line_batch_num * line_size, c)


class LineAttention(nn.Module):
    """
    Line-based self-attention module. Processes features along X-ray projection lines.
    """
    def __init__(self, dim, line_size=24, dim_head=64, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.line_size = line_size

        seq_l = line_size
        self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
        trunc_normal_(self.pos_emb)

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        n, c = x.shape
        l_size = self.line_size

        x_inp = ray_partition(x, line_size=l_size)

        q = self.to_q(x_inp)
        k, v = self.to_kv(x_inp).chunk(2, dim=-1)

        q, k, v = map(
            lambda t: t.contiguous().view(
                t.shape[0], t.shape[1], self.heads, t.shape[2] // self.heads
            ).permute(0, 2, 1, 3),
            (q, k, v)
        )

        q *= self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(out.shape[0], out.shape[2], -1)
        out = self.to_out(out)
        out = ray_merge(out)

        return out


class FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult, bias=False),
            GELU(),
            nn.Linear(dim * mult, dim * mult, bias=False),
            GELU(),
            nn.Linear(dim * mult, dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class LineAttentionBlock(nn.Module):
    def __init__(self, dim, line_size=24, dim_head=32, heads=8, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, LineAttention(dim=dim, line_size=line_size, dim_head=dim_head, heads=heads)),
                PreNorm(dim, FFN(dim=dim))
            ]))

    def forward(self, x):
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Lineformer(nn.Module):
    """
    Structure-aware Lineformer network for sparse-view X-ray CT reconstruction.
    Uses line-based attention to capture structural patterns along X-ray projection paths.

    Based on SAX-NeRF (CVPR 2024): Structure-Aware Sparse-View X-ray 3D Reconstruction.
    """
    def __init__(
        self, encoder, bound=0.2, num_layers=8, hidden_dim=256,
        skips=None, out_dim=1, last_activation="sigmoid",
        line_size=16, dim_head=32, heads=8, num_blocks=1
    ):
        super().__init__()
        if skips is None:
            skips = [4]
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.bound = bound
        self.encoder = encoder
        self.in_dim = encoder.output_dim

        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden_dim)] + [
                LineAttentionBlock(
                    dim=hidden_dim, line_size=line_size,
                    dim_head=dim_head, heads=heads, num_blocks=num_blocks
                )
                if i not in skips
                else nn.Linear(hidden_dim + self.in_dim, hidden_dim)
                for i in range(1, num_layers - 1, 1)
            ]
        )
        self.layers.append(nn.Linear(hidden_dim, out_dim))

        self.activations = nn.ModuleList(
            [nn.LeakyReLU() for _ in range(num_layers - 1)]
        )
        if last_activation == "sigmoid":
            self.activations.append(nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(nn.LeakyReLU())
        else:
            raise NotImplementedError(f"Unknown last activation: {last_activation}")

    def forward(self, x):
        x = self.encoder(x, self.bound)
        input_pts = x[..., :self.in_dim]

        for i in range(len(self.layers)):
            layer = self.layers[i]
            activation = self.activations[i]
            if i in self.skips:
                x = torch.cat([input_pts, x], -1)
            x = layer(x)
            x = activation(x)

        return x
