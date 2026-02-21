import torch


def calc_mse_loss(loss, x, y):
    """Calculate MSE loss and accumulate into loss dict."""
    loss_mse = torch.mean((x - y) ** 2)
    loss["loss"] += loss_mse
    loss["loss_mse"] = loss_mse
    return loss


def calc_mse_loss_raw(loss, x, y, k=1):
    """Calculate MSE loss for raw predictions with optional scaling."""
    loss_mse_raw = torch.mean((x - y) ** 2)
    loss["loss"] += k * loss_mse_raw
    loss["loss_mse_raw"] = loss_mse_raw
    return loss


def calc_tv_loss(loss, x, k):
    """
    Calculate total variation loss on 3D density field.

    Args:
        x: (n1, n2, n3) 3D density field tensor
        k: relative weight for TV loss
    """
    n1, n2, n3 = x.shape
    tv_1 = torch.abs(x[1:, 1:, 1:] - x[:-1, 1:, 1:]).sum()
    tv_2 = torch.abs(x[1:, 1:, 1:] - x[1:, :-1, 1:]).sum()
    tv_3 = torch.abs(x[1:, 1:, 1:] - x[1:, 1:, :-1]).sum()
    tv = (tv_1 + tv_2 + tv_3) / (n1 * n2 * n3)
    loss["loss"] += tv * k
    loss["loss_tv"] = tv * k
    return loss
