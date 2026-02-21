"""
Baseline SAX-NeRF (Lineformer) Training Script.

Train the baseline SAX-NeRF model without Multi-View Consistency Regularization.
Use this for comparison against MV-SAX results.

Usage:
    python train_sax_nerf.py --config config/Lineformer/chest_50.yaml --gpu_id 0

Reference: Cai et al., "Structure-Aware Sparse-View X-ray 3D Reconstruction," CVPR 2024.
"""

import os
import argparse


def config_parser():
    parser = argparse.ArgumentParser(
        description="Train baseline SAX-NeRF (Lineformer)"
    )
    parser.add_argument(
        "--config",
        default="./config/Lineformer/chest_50.yaml",
        help="Path to YAML config file"
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

import torch
import imageio.v2 as iio
import numpy as np
import os.path as osp
from tqdm import tqdm

from src.config.configloading import load_config
from src.render import render, run_network
from src.trainer_mlg import Trainer
from src.loss import calc_mse_loss
from src.utils import get_psnr, get_mse, get_psnr_3d, get_ssim_3d, cast_to_image, get_ssim


cfg = load_config(args.config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaselineTrainer(Trainer):
    def __init__(self):
        """SAX-NeRF baseline trainer (Lineformer, no MVCR)."""
        super().__init__(cfg, device)
        print(f"[Start] exp: {cfg['exp']['expname']}, method: SAX-NeRF baseline")

    def compute_loss(self, data, global_step, idx_epoch):
        rays = data["rays"].reshape(-1, 8)
        projs = data["projs"].reshape(-1)
        ret = render(rays, self.net, self.net_fine, **self.conf["render"])
        projs_pred = ret["acc"]

        loss = {"loss": torch.tensor(0.0, device=rays.device)}
        calc_mse_loss(loss, projs, projs_pred)

        for ls in loss.keys():
            self.writer.add_scalar(f"train/{ls}", loss[ls].item(), global_step)

        return loss["loss"]

    def eval_step(self, global_step, idx_epoch):
        projs = self.eval_dset.projs
        rays = self.eval_dset.rays.reshape(-1, 8)
        N, H, W = projs.shape
        projs_pred = []
        for i in tqdm(range(0, rays.shape[0], self.n_rays)):
            projs_pred.append(
                render(rays[i:i + self.n_rays], self.net, self.net_fine, **self.conf["render"])["acc"]
            )
        projs_pred = torch.cat(projs_pred, 0).reshape(N, H, W)

        image = self.eval_dset.image
        image_pred = run_network(
            self.eval_dset.voxels,
            self.net_fine if self.net_fine is not None else self.net,
            self.netchunk
        )
        image_pred = image_pred.squeeze()

        loss = {
            "proj_psnr": get_psnr(projs_pred, projs),
            "proj_ssim": get_ssim(projs_pred, projs),
            "psnr_3d": get_psnr_3d(image_pred, image),
            "ssim_3d": get_ssim_3d(image_pred, image),
        }

        if loss["psnr_3d"] > self.best_psnr_3d:
            torch.save(
                {
                    "epoch": idx_epoch,
                    "network": self.net.state_dict(),
                    "network_fine": (
                        self.net_fine.state_dict() if self.n_fine > 0 else None
                    ),
                    "optimizer": self.optimizer.state_dict(),
                },
                self.ckpt_best_dir,
            )
            self.best_psnr_3d = loss["psnr_3d"]
            self.logger.info(
                f"best model update, epoch:{idx_epoch}, "
                f"best 3d psnr:{self.best_psnr_3d:.4g}"
            )

        show_slice = 5
        show_step = image.shape[-1] // show_slice
        show_image = image[..., ::show_step]
        show_image_pred = image_pred[..., ::show_step]
        show = [
            torch.concat([show_image[..., i], show_image_pred[..., i]], dim=0)
            for i in range(show_slice)
        ]
        show_density = torch.concat(show, dim=1)
        self.writer.add_image(
            "eval/density (row1: gt, row2: pred)",
            cast_to_image(show_density),
            global_step,
            dataformats="HWC"
        )

        proj_pred_dir = osp.join(self.expdir, "proj_pred")
        proj_gt_dir = osp.join(self.expdir, "proj_gt")
        os.makedirs(proj_pred_dir, exist_ok=True)
        os.makedirs(proj_gt_dir, exist_ok=True)
        for i in tqdm(range(N)):
            iio.imwrite(
                osp.join(proj_pred_dir, f"proj_pred_{i}.png"),
                ((1 - cast_to_image(projs_pred[i])) * 255).astype(np.uint8)
            )
            iio.imwrite(
                osp.join(proj_gt_dir, f"proj_gt_{i}.png"),
                ((1 - cast_to_image(1 - projs[i])) * 255).astype(np.uint8)
            )

        for ls in loss.keys():
            self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)

        eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")
        os.makedirs(eval_save_dir, exist_ok=True)
        np.save(osp.join(eval_save_dir, "image_pred.npy"), image_pred.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "image_gt.npy"), image.cpu().detach().numpy())
        iio.imwrite(
            osp.join(eval_save_dir, "slice_show_row1_gt_row2_pred.png"),
            (cast_to_image(show_density) * 255).astype(np.uint8)
        )
        with open(osp.join(eval_save_dir, "stats.txt"), "w") as f:
            for key, value in loss.items():
                if torch.is_tensor(value):
                    f.write(f"{key}: {value.item():.6f}\n")
                else:
                    f.write(f"{key}: {value:.6f}\n")

        return loss


trainer = BaselineTrainer()
trainer.start()
