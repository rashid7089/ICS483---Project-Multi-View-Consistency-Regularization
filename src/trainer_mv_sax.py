"""
MV-SAX Trainer: Multi-View Consistency Regularization for Sparse-View CT Reconstruction.

This trainer extends the base SAX-NeRF (Lineformer) training pipeline by incorporating
Multi-View Consistency Regularization (MVCR). The MVCR loss enforces that the model's
3D density representation produces geometrically consistent projections at unseen
intermediate viewing angles, improving generalization from sparse input views.

Key additions over baseline SAX-NeRF:
  - Multi-View Consistency Regularization loss (src/loss/mv_consistency.py)
  - Curriculum scheduling of the MVCR weight (warm-up then plateau)
  - Logging of MVCR loss component for analysis

Authors: Rashed Jafry, Marwan Khayat, Mohammed Ali (ICS483 Course Project, 2025)
"""

import os
import os.path as osp
import pickle
import numpy as np
import torch
import imageio.v2 as iio
from tqdm import tqdm

from .trainer_mlg import Trainer
from .render import render, run_network
from .loss import calc_mse_loss
from .loss.mv_consistency import calc_mv_consistency_loss
from .utils import (
    get_psnr, get_mse, get_psnr_3d, get_ssim_3d, cast_to_image, get_ssim
)


class MV_SAX_Trainer(Trainer):
    """
    MV-SAX trainer that adds Multi-View Consistency Regularization to SAX-NeRF.

    Configuration parameters (in addition to base SAX-NeRF config):
      mvcr:
        lambda_mvcr: float     - Weight for the consistency loss (default: 0.1)
        n_virtual_views: int   - Virtual views per training step (default: 2)
        n_virtual_rays: int    - Sampled rays per virtual view (default: 512)
        warmup_epochs: int     - Epochs before full MVCR weight kicks in (default: 100)
        use_model_renders: bool - Use model renders as pseudo GT (default: False)
    """

    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device)
        self.device = device

        # Load MVCR-specific config with defaults
        mvcr_cfg = cfg.get("mvcr", {})
        self.lambda_mvcr = mvcr_cfg.get("lambda_mvcr", 0.1)
        self.n_virtual_views = mvcr_cfg.get("n_virtual_views", 2)
        self.n_virtual_rays = mvcr_cfg.get("n_virtual_rays", 512)
        self.warmup_epochs = mvcr_cfg.get("warmup_epochs", 100)
        self.use_model_renders = mvcr_cfg.get("use_model_renders", False)

        # Load training angles and projections for the MVCR loss
        # These are needed to define virtual view angles and GT interpolation
        with open(cfg["exp"]["datadir"], "rb") as f:
            data = pickle.load(f)

        self.train_angles = np.array(data["train"]["angles"])
        # Store GT projections as tensor for MVCR loss
        self.train_projs_all = torch.tensor(
            data["train"]["projections"], dtype=torch.float32, device=device
        )
        # Access the geometry from the eval dataset
        self.geo = self.eval_dset.geo
        self.near = self.eval_dset.near
        self.far = self.eval_dset.far

        print(
            f"[MV-SAX] λ_mvcr={self.lambda_mvcr}, "
            f"n_virtual_views={self.n_virtual_views}, "
            f"n_virtual_rays={self.n_virtual_rays}, "
            f"warmup_epochs={self.warmup_epochs}"
        )

    def _get_mvcr_weight(self, idx_epoch):
        """
        Curriculum scheduling: ramp up MVCR weight linearly during warm-up,
        then hold constant. This prevents the consistency loss from destabilizing
        early training when the model has not yet learned a reasonable density field.

        Args:
            idx_epoch (int): current epoch

        Returns:
            float: effective MVCR weight for this epoch
        """
        if self.warmup_epochs <= 0:
            return self.lambda_mvcr
        ramp = min(1.0, idx_epoch / self.warmup_epochs)
        return self.lambda_mvcr * ramp

    def compute_loss(self, data, global_step, idx_epoch):
        """
        Compute total training loss = reconstruction loss + MVCR loss.

        Step 1: Standard SAX-NeRF reconstruction loss on the current batch.
        Step 2: MVCR loss on n_virtual_views randomly sampled intermediate angles.
        """
        rays = data["rays"].reshape(-1, 8)
        projs = data["projs"].reshape(-1)

        # --- Step 1: Standard SAX-NeRF reconstruction loss ---
        ret = render(rays, self.net, self.net_fine, **self.conf["render"])
        projs_pred = ret["acc"]

        loss = {"loss": torch.tensor(0.0, device=rays.device)}
        calc_mse_loss(loss, projs, projs_pred)

        # --- Step 2: Multi-View Consistency Regularization ---
        lambda_eff = self._get_mvcr_weight(idx_epoch)
        if lambda_eff > 0:
            calc_mv_consistency_loss(
                loss=loss,
                net=self.net,
                train_angles=self.train_angles,
                train_projs=self.train_projs_all,
                geo=self.geo,
                near=self.near,
                far=self.far,
                render_cfg=self.conf["render"],
                n_virtual_rays=self.n_virtual_rays,
                n_virtual_views=self.n_virtual_views,
                lambda_mvcr=lambda_eff,
                device=self.device,
                use_model_renders=self.use_model_renders,
            )

        # Log all loss components
        for ls in loss.keys():
            self.writer.add_scalar(f"train/{ls}", loss[ls].item(), global_step)

        return loss["loss"]

    def eval_step(self, global_step, idx_epoch):
        """
        Evaluation: compute 2D projection metrics and 3D volume metrics.
        Saves rendered projections and 3D slices for visual inspection.
        """
        # Render all evaluation projections
        projs = self.eval_dset.projs                   # [N_val, H, W]
        rays = self.eval_dset.rays.reshape(-1, 8)
        N, H, W = projs.shape
        projs_pred = []
        for i in tqdm(range(0, rays.shape[0], self.n_rays), desc="Rendering eval"):
            projs_pred.append(
                render(rays[i:i + self.n_rays], self.net, self.net_fine, **self.conf["render"])["acc"]
            )
        projs_pred = torch.cat(projs_pred, 0).reshape(N, H, W)

        # Evaluate 3D density field
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

        # Save best model
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

        # Visualize 3D slices (5 equidistant slices)
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

        # Save projection images
        proj_pred_dir = osp.join(self.expdir, "proj_pred")
        proj_gt_dir = osp.join(self.expdir, "proj_gt")
        os.makedirs(proj_pred_dir, exist_ok=True)
        os.makedirs(proj_gt_dir, exist_ok=True)

        for i in tqdm(range(N), desc="Saving projections"):
            iio.imwrite(
                osp.join(proj_pred_dir, f"proj_pred_{i}.png"),
                ((1 - cast_to_image(projs_pred[i])) * 255).astype(np.uint8)
            )
            iio.imwrite(
                osp.join(proj_gt_dir, f"proj_gt_{i}.png"),
                ((1 - cast_to_image(1 - projs[i])) * 255).astype(np.uint8)
            )

        # Log metrics to TensorBoard
        for ls in loss.keys():
            self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)

        # Save evaluation results to disk
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
