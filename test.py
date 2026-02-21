"""
Evaluation script for MV-SAX and SAX-NeRF models.

Loads a trained checkpoint and evaluates both 2D projection quality and
3D CT reconstruction quality on the validation set.

Usage:
    # Evaluate MV-SAX
    python test.py --config config/MV_SAX/chest_50.yaml \\
                   --weights logs/MV_SAX/chest_50/.../ckpt_best.tar \\
                   --output_path output/chest_mvsax

    # Evaluate baseline SAX-NeRF
    python test.py --config config/Lineformer/chest_50.yaml \\
                   --weights logs/Lineformer/chest_50/.../ckpt_best.tar \\
                   --output_path output/chest_baseline

Authors: Rashed Jafry, Marwan Khayat, Mohammed Ali (ICS483 Course Project, 2025)
"""

import os
import os.path as osp
import argparse
import torch
import numpy as np
import imageio.v2 as iio
from tqdm import tqdm


def config_parser():
    parser = argparse.ArgumentParser(description="Evaluate MV-SAX or SAX-NeRF models")
    parser.add_argument(
        "--config",
        default="./config/MV_SAX/chest_50.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to trained model checkpoint (.tar file)"
    )
    parser.add_argument(
        "--output_path",
        default="./output",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--gpu_id",
        default="0",
        help="GPU device ID"
    )
    return parser


parser = config_parser()
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

from src.config.configloading import load_config
from src.dataset import TIGREDataset
from src.render import render, run_network
from src.network import get_network
from src.encoder import get_encoder
from src.utils import get_psnr, get_psnr_3d, get_ssim, get_ssim_3d, cast_to_image


def main():
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print(f"Loading dataset from {cfg['exp']['datadir']}...")
    dset = TIGREDataset(cfg["exp"]["datadir"], n_rays=1024, type="val", device=device)

    # Build model
    network_cls = get_network(cfg["network"]["net_type"])
    cfg_net = {k: v for k, v in cfg["network"].items() if k != "net_type"}
    encoder = get_encoder(**cfg["encoder"])
    net = network_cls(encoder, **cfg_net).to(device)

    # Load weights
    print(f"Loading weights from {args.weights}...")
    ckpt = torch.load(args.weights, map_location=device)
    net.load_state_dict(ckpt["network"])
    net.eval()

    # Setup output directory
    os.makedirs(args.output_path, exist_ok=True)

    print("Evaluating projections...")
    projs_gt = dset.projs        # [N_val, H, W]
    rays = dset.rays.reshape(-1, 8)
    N, H, W = projs_gt.shape
    n_rays = cfg["train"]["n_rays"]

    projs_pred = []
    with torch.no_grad():
        for i in tqdm(range(0, rays.shape[0], n_rays)):
            ret = render(
                rays[i:i + n_rays], net, None,
                **cfg["render"]
            )
            projs_pred.append(ret["acc"])
    projs_pred = torch.cat(projs_pred, 0).reshape(N, H, W)

    print("Evaluating 3D CT volume...")
    image_gt = dset.image
    with torch.no_grad():
        image_pred = run_network(
            dset.voxels, net, cfg["render"]["netchunk"]
        )
    image_pred = image_pred.squeeze()

    # Compute metrics
    metrics = {
        "proj_psnr": get_psnr(projs_pred, projs_gt).item(),
        "proj_ssim": get_ssim(projs_pred, projs_gt),
        "psnr_3d": get_psnr_3d(image_pred, image_gt),
        "ssim_3d": get_ssim_3d(image_pred, image_gt),
    }

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print("=" * 50)

    # Save metrics
    stats_path = osp.join(args.output_path, "stats.txt")
    with open(stats_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"Metrics saved to {stats_path}")

    # Save projected images
    proj_pred_dir = osp.join(args.output_path, "proj_pred")
    proj_gt_dir = osp.join(args.output_path, "proj_gt")
    os.makedirs(proj_pred_dir, exist_ok=True)
    os.makedirs(proj_gt_dir, exist_ok=True)

    print("Saving projection images...")
    for i in tqdm(range(N)):
        iio.imwrite(
            osp.join(proj_pred_dir, f"proj_pred_{i:03d}.png"),
            ((1 - cast_to_image(projs_pred[i])) * 255).astype(np.uint8)
        )
        iio.imwrite(
            osp.join(proj_gt_dir, f"proj_gt_{i:03d}.png"),
            ((1 - cast_to_image(1 - projs_gt[i])) * 255).astype(np.uint8)
        )

    # Save 3D reconstruction
    vol_path = osp.join(args.output_path, "volume_pred.npy")
    np.save(vol_path, image_pred.cpu().detach().numpy())
    print(f"3D volume saved to {vol_path}")

    # Save comparison slices
    show_slice = 5
    show_step = image_gt.shape[-1] // show_slice
    show_image = image_gt[..., ::show_step]
    show_image_pred = image_pred[..., ::show_step]
    show = [
        torch.concat([show_image[..., i], show_image_pred[..., i]], dim=0)
        for i in range(show_slice)
    ]
    show_density = torch.concat(show, dim=1)
    slice_path = osp.join(args.output_path, "slices_gt_vs_pred.png")
    iio.imwrite(
        slice_path,
        (cast_to_image(show_density) * 255).astype(np.uint8)
    )
    print(f"Slice comparison saved to {slice_path}")


if __name__ == "__main__":
    main()
