import numpy as np
from skimage.metrics import structural_similarity as ssim
from KFSBench.src.evaluation.ssim import pairwise_ssim
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from KFSBench.src.utils import load_image, save_json, load_json
import os
from typing import List, Tuple


def calculate_annd(list_gt: List[np.ndarray], list_pred: List[np.ndarray]) -> List[Tuple[float, float]]:
    """Calculate Average Nearest Neighbor Distance (ANND) between two lists of numpy arrays."""
    annd_list = []
    for gt_array, pred_array in zip(list_gt, list_pred):
        # Skip if either array is empty
        if gt_array.size == 0 or pred_array.size == 0:
            continue

        # Calculate minimum distances from each point in gt_array to pred_array and vice versa
        distances_gt_to_pred = np.min(np.abs(gt_array[:, np.newaxis] - pred_array), axis=1)
        distances_pred_to_gt = np.min(np.abs(pred_array[:, np.newaxis] - gt_array), axis=1)

        # Compute mean distances for precision (gt -> pred) and recall (pred -> gt)
        annd_precision = np.mean(distances_pred_to_gt)
        annd_recall = np.mean(distances_gt_to_pred)
        annd_list.append((annd_precision, annd_recall))

    return annd_list

def calculate_ssim(list_gt: List[np.ndarray], list_pred: List[np.ndarray]) -> List[Tuple[float, float]]:
    """Calculate Structural Similarity Index (SSIM) between two lists of numpy arrays."""
    ssim_list = []
    for gt_array, pred_array in zip(list_gt, list_pred):
        # Skip if either array is empty
        if len(gt_array) == 0 or len(pred_array) == 0:
            continue

        ssim_cur_list = pairwise_ssim(gt_array, pred_array)

        # Compute mean SSIM for precision (gt -> pred) and recall (pred -> gt)
        ssim_precision = np.mean(np.max(ssim_cur_list, axis=0))
        ssim_recall = np.mean(np.max(ssim_cur_list, axis=1))
        ssim_list.append((ssim_precision, ssim_recall))

    return ssim_list
    
def calculate_prf(list_gt: List[np.ndarray], list_pred: List[np.ndarray], threshold: int = 5) -> Tuple[float, float, float]:
    """Calculate Precision, Recall, and F1 Score based on frame coverage."""
    precision_list, recall_list, f1_list = [], [], []
    for gt_array, pred_array in zip(list_gt, list_pred):
        # Skip if either array is empty
        if len(gt_array) == 0 or len(pred_array) == 0:
            continue
        
        distances = np.min(np.abs(gt_array[:, np.newaxis] - pred_array), axis=1)
        covered_frames = np.sum(distances <= threshold)
        total_covered = covered_frames
        total_gt_frames = len(gt_array)
        total_pred_frames = len(pred_array)
        precision = total_covered / total_pred_frames if total_pred_frames > 0 else 0
        recall = total_covered / total_gt_frames if total_gt_frames > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return precision_list, recall_list, f1_list

