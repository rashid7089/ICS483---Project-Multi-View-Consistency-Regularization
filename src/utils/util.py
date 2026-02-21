import os
import logging
import torch
import numpy as np
import cv2

from skimage.metrics import structural_similarity
from skimage import img_as_ubyte


get_mse = lambda x, y: torch.mean((x - y) ** 2)


def get_psnr(x, y):
    """Compute PSNR between two images (normalized to [0,1])."""
    if torch.max(x) == 0 or torch.max(y) == 0:
        return torch.zeros(1)
    x_norm = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    y_norm = (y - torch.min(y)) / (torch.max(y) - torch.min(y))
    mse = get_mse(x_norm, y_norm)
    psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).to(x.device))
    return psnr


def get_psnr_3d(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    """
    Compute 3D PSNR.

    Args:
        arr1, arr2: 3D tensors or numpy arrays [D, H, W]
        size_average: whether to return mean or per-sample PSNR

    Returns:
        PSNR value(s)
    """
    if torch.is_tensor(arr1):
        arr1 = arr1.cpu().detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.cpu().detach().numpy()
    arr1 = arr1[np.newaxis, ...]
    arr2 = arr2[np.newaxis, ...]
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    eps = 1e-10
    se = np.power(arr1 - arr2, 2)
    mse = se.mean(axis=1).mean(axis=1).mean(axis=1)
    zero_mse = np.where(mse == 0)
    mse[zero_mse] = eps
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    psnr[zero_mse] = 100

    if size_average:
        return psnr.mean()
    else:
        return psnr


def get_ssim(img1, img2, border=0):
    """
    Compute SSIM between two sets of 2D images.

    Args:
        img1, img2: tensors or arrays [B, H, W]
    """
    if torch.is_tensor(img1):
        img1 = img1.cpu().detach().numpy()
    if torch.is_tensor(img2):
        img2 = img2.cpu().detach().numpy()

    img1 = img_as_ubyte(img1)
    img2 = img_as_ubyte(img2)

    if img1.shape != img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    b, h, w = img1.shape
    img1 = img1[:, border:h - border, border:w - border]
    img2 = img2[:, border:h - border, border:w - border]

    if img1.ndim == 3:
        if b > 1:
            ssims = [_ssim(img1[i], img2[i]) for i in range(b)]
            return np.array(ssims).mean()
        else:
            return _ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def _ssim(img1, img2):
    """Compute SSIM between two 2D images."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = (
        (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    )
    return ssim_map.mean()


def get_ssim_3d(arr1, arr2, size_average=True):
    """
    Compute 3D SSIM (averaged over three orthogonal planes).

    Args:
        arr1, arr2: 3D tensors or numpy arrays
    """
    if torch.is_tensor(arr1):
        arr1 = arr1.cpu().detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.cpu().detach().numpy()
    arr1 = arr1[np.newaxis, ...]
    arr2 = arr2[np.newaxis, ...]
    assert arr1.ndim == 4 and arr2.ndim == 4
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)

    N = arr1.shape[0]

    # Depth slices
    arr1_d = np.transpose(arr1, (0, 2, 3, 1))
    arr2_d = np.transpose(arr2, (0, 2, 3, 1))
    ssim_d = np.array([
        structural_similarity(arr1_d[i], arr2_d[i]) for i in range(N)
    ])

    # Height slices
    arr1_h = np.transpose(arr1, (0, 1, 3, 2))
    arr2_h = np.transpose(arr2, (0, 1, 3, 2))
    ssim_h = np.array([
        structural_similarity(arr1_h[i], arr2_h[i]) for i in range(N)
    ])

    # Width slices
    ssim_w = np.array([
        structural_similarity(arr1[i], arr2[i]) for i in range(N)
    ])

    ssim_avg = (ssim_d + ssim_h + ssim_w) / 3

    if size_average:
        return ssim_avg.mean()
    else:
        return ssim_avg


def cast_to_image(tensor, normalize=True):
    """Convert tensor to displayable numpy image [H, W, 1]."""
    if torch.is_tensor(tensor):
        img = tensor.cpu().detach().numpy()
    else:
        img = tensor
    if normalize:
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
    return img[..., np.newaxis]


def gen_log(model_path):
    """Create a logger that writes to both console and file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = os.path.join(model_path, 'log.txt')
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def time2file_name(time):
    """Convert datetime string to a filename-safe string."""
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    return f"{year}_{month}_{day}_{hour}_{minute}_{second}"
