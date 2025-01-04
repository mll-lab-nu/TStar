import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple

def gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    """Creates a 1D Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g

def create_window(window_size: int, channel: int) -> torch.Tensor:
    """Creates a 2D Gaussian kernel window."""
    kernel_1d = gaussian_kernel(window_size, sigma=1.5).unsqueeze(1)
    window_2d = kernel_1d @ kernel_1d.T
    window = window_2d.expand(channel, 1, window_size, window_size)
    return window

def ssim_torch(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """Calculates the Structural Similarity Index (SSIM) between two images."""
    channel = img1.size(0)
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1.unsqueeze(0), window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2.unsqueeze(0), window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1.unsqueeze(0) * img1.unsqueeze(0), window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def pairwise_ssim(gt_frames: List[np.ndarray], pred_frames: List[np.ndarray]) -> List[Tuple[Tuple[int, int], float]]:
    """Calculates SSIM for each pair in a list of decoded frames."""
    # Move frames to a list of tensors and place on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gt_frames[0].dtype == np.uint8:
        gt_frames_torch = [torch.tensor(frame, dtype=torch.float32, device=device) / 255.0 for frame in gt_frames]
        pred_frames_torch = [torch.tensor(frame, dtype=torch.float32, device=device) / 255.0 for frame in pred_frames]
    else: # maybe already torch
        gt_frames_torch = gt_frames
        pred_frames_torch = pred_frames

    # List to store SSIM results
    ssim_results = np.zeros((len(gt_frames), len(pred_frames)))
    for i in range(len(gt_frames_torch)):
        for j in range(len(pred_frames_torch)):
            ssim_score = ssim_torch(gt_frames_torch[i], pred_frames_torch[j])
            ssim_results[i][j] = ssim_score.item()
            
    return ssim_results
