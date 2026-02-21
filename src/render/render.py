import torch


def render(rays, net, net_fine, n_samples, n_fine, perturb, netchunk, raw_noise_std):
    """
    Render X-ray projections by integrating density along rays.

    Args:
        rays: [N_rays, 8] tensor (origin, direction, near, far)
        net: coarse network
        net_fine: fine network (optional)
        n_samples: number of coarse samples per ray
        n_fine: number of fine samples per ray
        perturb: whether to apply stratified sampling perturbation
        netchunk: batch size for network forward passes
        raw_noise_std: std of noise added to raw density predictions

    Returns:
        dict with 'acc' (rendered projections), 'pts', 'raw', and coarse outputs
    """
    n_rays = rays.shape[0]

    rays_o, rays_d = rays[..., :3], rays[..., 3:6]
    near, far = rays[..., 6:7], rays[..., 7:]

    t_vals = torch.linspace(0., 1., steps=n_samples, device=near.device)
    z_vals = near * (1. - t_vals) + far * t_vals
    z_vals = z_vals.expand([n_rays, n_samples])

    if perturb:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=lower.device)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    bound = net.bound - 1e-6
    pts = pts.clamp(-bound, bound)

    raw = run_network(pts, net, netchunk)
    acc, weights = raw2outputs(raw, z_vals, rays_d, raw_noise_std)

    if net_fine is not None and n_fine > 0:
        acc_0 = acc
        weights_0 = weights
        pts_0 = pts

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_fine, det=(perturb == 0.))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts = pts.clamp(-bound, bound)
        raw = run_network(pts, net_fine, netchunk)
        acc, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std)

    ret = {"acc": acc, "pts": pts, "raw": raw}

    if net_fine is not None and n_fine > 0:
        ret["acc0"] = acc_0
        ret["weights0"] = weights_0
        ret["pts0"] = pts_0

    for k in ret:
        if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def run_network(inputs, fn, netchunk):
    """
    Evaluate the density network on a batch of 3D points.

    Args:
        inputs: [N_rays, N_samples, 3] or [N1, N2, N3, 3] for voxel evaluation
        fn: the density network
        netchunk: max points per forward pass

    Returns:
        outputs with same spatial dims as inputs, last dim = network output dim
    """
    uvt_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    out_flat = torch.cat(
        [fn(uvt_flat[i:i + netchunk]) for i in range(0, uvt_flat.shape[0], netchunk)],
        0
    )
    return out_flat.reshape(list(inputs.shape[:-1]) + [out_flat.shape[-1]])


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.):
    """
    Convert raw network output (density) to rendered projection values.

    Args:
        raw: [N_rays, N_samples, 1] density predictions
        z_vals: [N_rays, N_samples] sample depths
        rays_d: [N_rays, 3] ray directions
        raw_noise_std: noise standard deviation

    Returns:
        acc: [N_rays] integrated X-ray projections (Beer-Lambert law integrals)
        weights: [N_rays, N_samples] importance weights for fine sampling
    """
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists, torch.Tensor([1e-10]).expand(dists[..., :1].shape).to(dists.device)], -1
    )
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 0].shape) * raw_noise_std
        noise = noise.to(raw.device)

    acc = torch.sum((raw[..., 0] + noise) * dists, dim=-1)

    if raw.shape[-1] == 1:
        eps = torch.ones_like(raw[:, :1, -1]) * 1e-10
        weights = torch.cat([eps, torch.abs(raw[:, 1:, -1] - raw[:, :-1, -1])], dim=-1)
        weights = weights / torch.max(weights)
    elif raw.shape[-1] == 2:
        weights = raw[..., 1] / torch.max(raw[..., 1])
    else:
        raise NotImplementedError("Wrong raw shape")

    return acc, weights


def sample_pdf(bins, weights, N_samples, det=False):
    """
    Hierarchical sampling using PDF from coarse model weights.

    Args:
        bins: [N_rays, N_coarse-1] bin edges
        weights: [N_rays, N_coarse-2] importance weights
        N_samples: number of fine samples
        det: deterministic sampling flag

    Returns:
        samples: [N_rays, N_samples] new sample depths
    """
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    u = u.contiguous().to(cdf.device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
