import torch
import pickle
import numpy as np

from torch.utils.data import Dataset
from .tigre import ConeGeometry


def proj_window_partition(x, window_size):
    """Partition projection into windows."""
    h, w = x.shape
    x = x.view(
        h // window_size[0], window_size[0], w // window_size[1], window_size[1]
    )
    windows = x.permute(0, 2, 1, 3).contiguous().view(
        -1, window_size[0], window_size[1]
    )
    return windows


def ray_window_partition(x, window_size):
    """Partition rays into windows."""
    h, w, c = x.shape
    x = x.view(
        h // window_size[0], window_size[0], w // window_size[1], window_size[1], c
    )
    windows = x.permute(0, 2, 1, 3, 4).contiguous().view(
        -1, window_size[0], window_size[1], c
    )
    return windows


class TIGREDataset_MLG(Dataset):
    """
    TIGRE dataset with multi-line group (MLG) window sampling strategy.
    Used as the base dataset for SAX-NeRF (Lineformer) training.
    """
    def __init__(
        self, path, n_rays=1024, type="train",
        window_size=None, window_num=4, device="cuda"
    ):
        super().__init__()
        if window_size is None:
            window_size = [32, 32]

        with open(path, "rb") as handle:
            data = pickle.load(handle)

        self.geo = ConeGeometry(data)
        self.window_size = window_size
        self.window_num = window_num
        self.type = type
        self.n_rays = n_rays
        self.near, self.far = self.get_near_far(self.geo)

        if type == "train":
            self.projs = torch.tensor(
                data["train"]["projections"], dtype=torch.float32, device=device
            )
            angles = data["train"]["angles"]
            rays = self.get_rays(angles, self.geo, device)
            self.rays = torch.cat(
                [rays, torch.ones_like(rays[..., :1]) * self.near,
                 torch.ones_like(rays[..., :1]) * self.far], dim=-1
            )
            self.n_samples = data["numTrain"]
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, self.geo.nDetector[1] - 1, self.geo.nDetector[1], device=device),
                    torch.linspace(0, self.geo.nDetector[0] - 1, self.geo.nDetector[0], device=device),
                    indexing="ij"
                ), -1
            )
            self.coords = torch.reshape(coords, [-1, 2])
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(
                self.get_voxels(self.geo), dtype=torch.float32, device=device
            )
        elif type == "val":
            self.projs = torch.tensor(
                data["val"]["projections"], dtype=torch.float32, device=device
            )
            angles = data["val"]["angles"]
            rays = self.get_rays(angles, self.geo, device)
            self.rays = torch.cat(
                [rays, torch.ones_like(rays[..., :1]) * self.near,
                 torch.ones_like(rays[..., :1]) * self.far], dim=-1
            )
            self.n_samples = data["numVal"]
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(
                self.get_voxels(self.geo), dtype=torch.float32, device=device
            )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if self.type == "train":
            rays = self.rays[index]
            projs = self.projs[index]

            rays_window = ray_window_partition(rays, self.window_size)
            projs_window = proj_window_partition(projs, self.window_size)

            projs_window_valid_indx = (
                (projs_window > 0).sum(dim=-1).sum(dim=-1)
                == self.window_size[0] * self.window_size[1]
            )
            select_inds_window = np.random.choice(
                projs_window_valid_indx.shape[0],
                size=[self.window_num],
                replace=False
            )

            projs_window_select = projs_window[select_inds_window]
            rays_window_select = rays_window[select_inds_window]
            selected_rays_window = rays_window_select.reshape(-1, 8)
            selected_projs_window = projs_window_select.flatten()

            total_inds = list(range(projs_window.shape[0]))
            else_inds = [x for x in total_inds if x not in select_inds_window]
            projs_window_else = projs_window[else_inds]
            rays_window_else = rays_window[else_inds]

            else_inds_pixel_valid = projs_window_else > 0
            rays_else_valid = rays_window_else[else_inds_pixel_valid]
            projs_else_valid = projs_window_else[else_inds_pixel_valid]

            else_valid_select_index = np.random.choice(
                projs_else_valid.shape[0], size=[self.n_rays], replace=False
            )
            selected_rays_else = rays_else_valid[else_valid_select_index]
            selected_projs_else = projs_else_valid[else_valid_select_index]

            selected_rays = torch.concat([selected_rays_window, selected_rays_else], dim=0)
            selected_projs = torch.concat([selected_projs_window, selected_projs_else], dim=0)

            return {"projs": selected_projs, "rays": selected_rays}
        elif self.type == "val":
            return {"projs": self.projs[index], "rays": self.rays[index]}

    def get_voxels(self, geo):
        n1, n2, n3 = geo.nVoxel
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2
        xyz = np.meshgrid(
            np.linspace(-s1, s1, n1),
            np.linspace(-s2, s2, n2),
            np.linspace(-s3, s3, n3),
            indexing="ij"
        )
        return np.asarray(xyz).transpose([1, 2, 3, 0])

    def get_rays(self, angles, geo, device):
        W, H = geo.nDetector
        DSD = geo.DSD
        rays = []
        for angle in angles:
            pose = torch.Tensor(self.angle2pose(geo.DSO, angle)).to(device)
            if geo.mode == "cone":
                i, j = torch.meshgrid(
                    torch.linspace(0, W - 1, W, device=device),
                    torch.linspace(0, H - 1, H, device=device),
                    indexing="ij"
                )
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([uu / DSD, vv / DSD, torch.ones_like(uu)], -1)
                rays_d = torch.sum(
                    torch.matmul(pose[:3, :3], dirs[..., None]).to(device), -1
                )
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
                rays_d = torch.sum(
                    torch.matmul(pose[:3, :3], dirs[..., None]).to(device), -1
                )
                rays_o = (
                    torch.sum(
                        torch.matmul(
                            pose[:3, :3],
                            torch.stack([uu, vv, torch.zeros_like(uu)], -1)[..., None]
                        ).to(device), -1
                    ) + pose[:3, -1].expand(rays_d.shape)
                )
            else:
                raise NotImplementedError("Unknown CT scanner type!")
            rays.append(torch.concat([rays_o, rays_d], dim=-1))
        return torch.stack(rays, dim=0)

    def angle2pose(self, DSO, angle):
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
        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
        T = np.eye(4)
        T[:-1, :-1] = rot
        T[:-1, -1] = trans
        return T

    def get_near_far(self, geo, tolerance=0.005):
        dist1 = np.linalg.norm(
            [geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2]
        )
        dist2 = np.linalg.norm(
            [geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2]
        )
        dist3 = np.linalg.norm(
            [geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2]
        )
        dist4 = np.linalg.norm(
            [geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2]
        )
        dist_max = np.max([dist1, dist2, dist3, dist4])
        near = np.max([0, geo.DSO - dist_max - tolerance])
        far = np.min([geo.DSO * 2, geo.DSO + dist_max + tolerance])
        return near, far
