"""
Multi-View Consistency Regularization (MVCR) for Sparse-View CT Reconstruction.

This module implements the core innovation of MV-SAX: enforcing geometric consistency
between views during training of the SAX-NeRF model.

The key idea:
  During training with K sparse input views, we sample "virtual" intermediate views
  at angles between adjacent training views. The model's rendering of these virtual
  views must be geometrically consistent with a linearly-interpolated pseudo ground
  truth derived from the two flanking training views.

  This regularization discourages the model from over-fitting to sparse training
  angles and encourages a globally consistent 3D density representation.

Loss components:
  1. Virtual-view photometric consistency:
       L_vc = ||render(net, θ_v) - interp(I_gt(θ_a), I_gt(θ_b), α)||²
     where θ_v = (1-α)*θ_a + α*θ_b for α ∈ (0,1)

  2. Angular smoothness regularization:
       L_sm = ||render(net, θ_v) - (render(net, θ_a) + render(net, θ_b)) / 2||²
     This variant uses the model's own renders for consistency rather than GT,
     making it applicable even for views without ground truth.

Reference:
  Cai et al., "Structure-Aware Sparse-View X-ray 3D Reconstruction," CVPR 2024.
  MV-SAX extension by Rashed Jafry, Marwan Khayat, Mohammed Ali (ICS483, 2025).
"""

import numpy as np
import torch
import torch.nn.functional as F

from ..render import render
from ..dataset.tigre import TIGREDataset


def _get_virtual_rays(angle, geo, device, near, far):
    """
    Compute rays for a given projection angle using the CT geometry.

    Args:
        angle (float): projection angle in radians
        geo: ConeGeometry object
        device: torch device
        near (float): near plane distance
        far (float): far plane distance

    Returns:
        rays: [H, W, 8] tensor (origin[3] + direction[3] + near[1] + far[1])
    """
    W, H = geo.nDetector
    DSD = geo.DSD

    phi1 = -np.pi / 2
    R1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(phi1), -np.sin(phi1)],
        [0.0, np.sin(phi1), np.cos(phi1)]
    ])
    phi2 = np.pi / 2
    R2 = np.array([
        [np.cos(phi2), -np.sin(phi2), 0.0],
        [np.sin(phi2), np.cos(phi2), 0.0],
        [0.0, 0.0, 1.0]
    ])
    R3 = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0]
    ])
    rot = np.dot(np.dot(R3, R2), R1)
    trans = np.array([geo.DSO * np.cos(angle), geo.DSO * np.sin(angle), 0])
    T = np.eye(4)
    T[:-1, :-1] = rot
    T[:-1, -1] = trans
    pose = torch.Tensor(T).to(device)

    if geo.mode == "cone":
        i, j = torch.meshgrid(
            torch.linspace(0, W - 1, W, device=device),
            torch.linspace(0, H - 1, H, device=device),
            indexing="ij"
        )
        uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
        vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
        dirs = torch.stack([uu / DSD, vv / DSD, torch.ones_like(uu)], -1)
        rays_d = torch.sum(torch.matmul(pose[:3, :3], dirs[..., None]).to(device), -1)
        rays_o = pose[:3, -1].expand(rays_d.shape)
    elif geo.mode == "parallel":
        i, j = torch.meshgrid(
            torch.linspace(0, W - 1, W, device=device),
            torch.linspace(0, H - 1, H, device=device),
            indexing="ij"
        )
        uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
        vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
        dirs = torch.stack(
            [torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)], -1
        )
        rays_d = torch.sum(torch.matmul(pose[:3, :3], dirs[..., None]).to(device), -1)
        rays_o = (
            torch.sum(
                torch.matmul(
                    pose[:3, :3],
                    torch.stack([uu, vv, torch.zeros_like(uu)], -1)[..., None]
                ).to(device), -1
            ) + pose[:3, -1].expand(rays_d.shape)
        )
    else:
        raise NotImplementedError(f"Unknown CT scanner mode: {geo.mode}")

    rays = torch.concat([rays_o, rays_d], dim=-1)  # [H, W, 6]
    near_t = torch.ones_like(rays[..., :1]) * near
    far_t = torch.ones_like(rays[..., :1]) * far
    return torch.cat([rays, near_t, far_t], dim=-1)  # [H, W, 8]


def calc_mv_consistency_loss(
    loss,
    net,
    train_angles,
    train_projs,
    geo,
    near,
    far,
    render_cfg,
    n_virtual_rays=512,
    n_virtual_views=2,
    lambda_mvcr=0.1,
    device="cuda",
    use_model_renders=False
):
    """
    Compute Multi-View Consistency Regularization (MVCR) loss.

    For each sampled virtual view V_mid at angle θ_mid between adjacent training
    views V_a and V_b, we enforce that the model's rendering of V_mid is consistent
    with an interpolated pseudo ground truth.

    Two modes:
      - use_model_renders=False (default): uses GT projections for pseudo GT.
        Provides a strong signal but requires GT to be accurate for intermediate angles.
      - use_model_renders=True: uses the model's own renders of V_a and V_b as
        pseudo GT for V_mid. This is a pure consistency constraint that does not
        require actual GT at intermediate angles.

    Args:
        loss (dict): accumulated loss dictionary, updated in-place
        net: the density network
        train_angles (list or array): training view angles [N_train]
        train_projs (Tensor): GT projections [N_train, H, W]
        geo: ConeGeometry object
        near (float): near plane distance
        far (float): far plane distance
        render_cfg (dict): rendering configuration (n_samples, perturb, netchunk, etc.)
        n_virtual_rays (int): number of randomly sampled rays per virtual view
        n_virtual_views (int): number of virtual views to sample per training step
        lambda_mvcr (float): weight for the consistency loss
        device (str or torch.device): compute device
        use_model_renders (bool): if True, use model renders instead of GT for pseudo GT

    Returns:
        loss (dict): updated with 'loss_mvcr' key
    """
    n_views = len(train_angles)
    loss_mvcr_total = torch.tensor(0.0, device=device)

    for _ in range(n_virtual_views):
        # Sample two adjacent training views
        idx_a = np.random.randint(0, n_views)
        idx_b = (idx_a + 1) % n_views

        theta_a = train_angles[idx_a]
        theta_b = train_angles[idx_b]

        # Interpolation factor: α ∈ (0.2, 0.8) to avoid edge cases
        alpha = np.random.uniform(0.2, 0.8)

        # Virtual view angle
        # Handle wrap-around (e.g., going from ~350° to ~10°)
        delta = theta_b - theta_a
        if abs(delta) > np.pi:
            delta = delta - np.sign(delta) * 2 * np.pi
        theta_v = theta_a + alpha * delta

        # Get rays for the virtual view (subsample for efficiency)
        H, W = geo.nDetector[0], geo.nDetector[1]
        rays_v = _get_virtual_rays(theta_v, geo, device, near, far)  # [H, W, 8]
        rays_v_flat = rays_v.reshape(-1, 8)

        n_total = rays_v_flat.shape[0]
        sample_size = min(n_virtual_rays, n_total)
        select_inds = np.random.choice(n_total, size=sample_size, replace=False)
        rays_v_sub = rays_v_flat[select_inds]

        # Render the virtual view
        ret_v = render(
            rays_v_sub, net, None,
            n_samples=render_cfg["n_samples"],
            n_fine=0,
            perturb=render_cfg["perturb"],
            netchunk=render_cfg["netchunk"],
            raw_noise_std=0.0
        )
        I_pred_v = ret_v["acc"]  # [n_virtual_rays]

        if use_model_renders:
            # Mode: angular smoothness via model's own renders
            # The virtual view rendering should be between the model renders of θ_a and θ_b
            rays_a = _get_virtual_rays(theta_a, geo, device, near, far)
            rays_b = _get_virtual_rays(theta_b, geo, device, near, far)
            rays_a_sub = rays_a.reshape(-1, 8)[select_inds]
            rays_b_sub = rays_b.reshape(-1, 8)[select_inds]

            with torch.no_grad():
                ret_a = render(
                    rays_a_sub, net, None,
                    n_samples=render_cfg["n_samples"],
                    n_fine=0,
                    perturb=False,
                    netchunk=render_cfg["netchunk"],
                    raw_noise_std=0.0
                )
                ret_b = render(
                    rays_b_sub, net, None,
                    n_samples=render_cfg["n_samples"],
                    n_fine=0,
                    perturb=False,
                    netchunk=render_cfg["netchunk"],
                    raw_noise_std=0.0
                )
            I_pseudo = (
                (1 - alpha) * ret_a["acc"].detach()
                + alpha * ret_b["acc"].detach()
            )
        else:
            # Mode: GT-based consistency
            # Use linear interpolation of adjacent GT projections as pseudo GT
            I_gt_a = train_projs[idx_a].flatten()  # [H*W]
            I_gt_b = train_projs[idx_b].flatten()  # [H*W]
            I_gt_a_sub = I_gt_a[select_inds]
            I_gt_b_sub = I_gt_b[select_inds]
            I_pseudo = (1 - alpha) * I_gt_a_sub + alpha * I_gt_b_sub

        # Consistency loss: rendered virtual view vs. pseudo GT
        loss_mvcr = F.mse_loss(I_pred_v, I_pseudo)
        loss_mvcr_total = loss_mvcr_total + loss_mvcr

    loss_mvcr_avg = lambda_mvcr * loss_mvcr_total / n_virtual_views
    loss["loss"] = loss["loss"] + loss_mvcr_avg
    loss["loss_mvcr"] = loss_mvcr_avg

    return loss
