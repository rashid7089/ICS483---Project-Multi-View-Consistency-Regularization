"""
MV-SAX Training Script.

Train the MV-SAX model (Multi-View Consistency Regularization for Sparse-View
CT Reconstruction) on a TIGRE-format dataset.

Usage:
    python train_mv_sax.py --config config/MV_SAX/chest_50.yaml --gpu_id 0

The MV-SAX model builds on SAX-NeRF (Lineformer, CVPR 2024) and adds
Multi-View Consistency Regularization (MVCR) to improve reconstruction
quality from sparse views.

Authors: Rashed Jafry, Marwan Khayat, Mohammed Ali (ICS483 Course Project, 2025)
Reference: Cai et al., "Structure-Aware Sparse-View X-ray 3D Reconstruction," CVPR 2024.
           GitHub: https://github.com/caiyuanhao1998/SAX-NeRF
"""

import os
import argparse


def config_parser():
    parser = argparse.ArgumentParser(
        description="Train MV-SAX: Multi-View Consistency Regularization for CT Reconstruction"
    )
    parser.add_argument(
        "--config",
        default="./config/MV_SAX/chest_50.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--gpu_id",
        default="0",
        help="GPU device ID (e.g. '0', '1')"
    )
    return parser


parser = config_parser()
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import torch
from src.config.configloading import load_config
from src.trainer_mv_sax import MV_SAX_Trainer


cfg = load_config(args.config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MV_SAX_Run(MV_SAX_Trainer):
    def __init__(self):
        """MV-SAX training run."""
        super().__init__(cfg, device)
        print(f"[Start] exp: {cfg['exp']['expname']}, method: MV-SAX (Lineformer + MVCR)")


trainer = MV_SAX_Run()
trainer.start()
