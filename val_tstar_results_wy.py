import os
import json
import argparse
import logging
from typing import List, Tuple, Any, Dict
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys
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


# Import custom modules from KFSBench
# Ensure that KFSBench is installed and accessible in PYTHONPATH


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Calculate PRF and SSIM metrics from video analysis results."
    )
    parser.add_argument(
        '--result_path',
        type=str,
        default="/data/guoweiyu/LV-Haystack/results/frame_search/2025-03-22-07-33-52objnew_LVHaystack_gpt4_raw_vid1.json",
        help='Path to the input JSON file containing video analysis results.'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='metrics_results.json',
        help='Path to save the calculated metrics results.'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=1,
        help='Frames per second (FPS) of the videos.'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=5,
        help='Distance threshold for PRF calculation.'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=32,
        help='Maximum number of threads for parallel processing.'
    )
    parser.add_argument(
        '--num',
        type=int,
        default=100,
        help='Num of result entries to process.'
    )
    return parser.parse_args()


def load_json_file(file_path: str) -> Any:
    """
    Load a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Any: Parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not os.path.exists(file_path):
        logger.error(f"JSON file does not exist: {file_path}")
        raise FileNotFoundError(f"JSON file does not exist: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            logger.info(f"Successfully loaded JSON file: {file_path}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON file: {file_path}")
            raise e


def save_json_file(data: Any, file_path: str) -> None:
    """
    Save data to a JSON file.

    Args:
        data (Any): Data to be saved.
        file_path (str): Path to the output JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f"Metrics results saved to: {file_path}")


def calculate_annd(list_gt: List[np.ndarray], list_pred: List[np.ndarray]) -> List[Tuple[float, float]]:
    """
    Calculate Average Nearest Neighbor Distance (ANND) between two lists of numpy arrays.

    Args:
        list_gt (List[np.ndarray]): List of ground truth frame numbers.
        list_pred (List[np.ndarray]): List of predicted frame numbers.

    Returns:
        List[Tuple[float, float]]: List of (precision, recall) tuples.
    """
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


def calculate_ssim_scores(list_gt_images: List[List[np.ndarray]], list_pred_images: List[List[np.ndarray]]) -> List[Tuple[float, float]]:
    """
    Calculate Structural Similarity Index (SSIM) between two lists of image arrays.

    Args:
        list_gt_images (List[List[np.ndarray]]): List of ground truth image arrays per video.
        list_pred_images (List[List[np.ndarray]]): List of predicted image arrays per video.

    Returns:
        List[Tuple[float, float]]: List of (precision_ssim, recall_ssim) tuples.
    """
    ssim_list = []
    for gt_images, pred_images in zip(list_gt_images, list_pred_images):
        # Skip if either list is empty
        if not gt_images or not pred_images:
            continue
        # print(len(gt_images))
        # print(len(pred_images))
        # Pair images: assuming they are in the same order
        paired_gt = []
        paired_pred = []

        
        for i in range(len(gt_images)):
            if gt_images[i].size == 0:
                continue
            paired_gt.append(gt_images[i])
        for i in range(len(pred_images)):
            if pred_images[i].size == 0:
                continue
            paired_pred.append(pred_images[i])

        # Compute pairwise SSIM
        ssim_cur_list = pairwise_ssim(paired_gt, paired_pred)
        # print(ssim_cur_list)
        # Compute mean SSIM for precision and recall
        if ssim_cur_list.size == 0:
            continue
        ssim_precision = np.mean(np.max(ssim_cur_list, axis=0))
        ssim_recall = np.mean(np.max(ssim_cur_list, axis=1))
        ssim_list.append((ssim_precision, ssim_recall))

    return ssim_list


def calculate_prf(list_gt: List[np.ndarray], list_pred: List[np.ndarray], threshold: int = 5) -> Tuple[float, float, float]:
    """
    Calculate Precision, Recall, and F1 Score based on frame coverage.

    Args:
        list_gt (List[np.ndarray]): List of ground truth frame numbers.
        list_pred (List[np.ndarray]): List of predicted frame numbers.
        threshold (int, optional): Distance threshold. Defaults to 5.

    Returns:
        Tuple[float, float, float]: Average Precision, Recall, and F1 Score.
    """
    precision_list, recall_list, f1_list = [], [], []
    
    for gt_array, pred_array in zip(list_gt, list_pred):
        # Skip if either array is empty
        if gt_array.size == 0 or pred_array.size == 0:
            continue

        # Calculate distances from each gt frame to pred frames
        distances_gt_to_pred = np.min(np.abs(gt_array[:, np.newaxis] - pred_array), axis=1)
        # Calculate distances from each pred frame to gt frames
        distances_pred_to_gt = np.min(np.abs(pred_array[:, np.newaxis] - gt_array), axis=1)

        # Frames within the threshold
        covered_gt = np.sum(distances_gt_to_pred <= threshold)
        covered_pred = np.sum(distances_pred_to_gt <= threshold)
        total_gt_frames = len(gt_array)
        total_pred_frames = len(pred_array)

        # Calculate Precision, Recall, F1
        precision = covered_pred / total_pred_frames if total_pred_frames > 0 else 0.0
        recall = covered_gt / total_gt_frames if total_gt_frames > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # Compute average metrics
    avg_precision = np.mean(precision_list) if precision_list else 0.0
    avg_recall = np.mean(recall_list) if recall_list else 0.0
    avg_f1 = np.mean(f1_list) if f1_list else 0.0

    return avg_precision, avg_recall, avg_f1


def load_video_fps(video_path: str) -> float:
    """
    Get the frames per second (FPS) of a video.

    Args:
        video_path (str): Path to the video file.

    Returns:
        float: FPS of the video.

    Raises:
        ValueError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        raise ValueError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    logger.debug(f"Video FPS for {video_path}: {fps}")
    return fps


def extract_frames(video_path: str, frame_indices: List[int]) -> List[np.ndarray]:
    """
    Extract specified frames from a video.

    Args:
        video_path (str): Path to the video file.
        frame_indices (List[int]): List of frame indices to extract.

    Returns:
        List[np.ndarray]: List of extracted frames in RGB format.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        raise ValueError(f"Cannot open video file: {video_path}")

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            logger.debug(f"Extracted frame {idx} from {video_path}")
        else:
            frames.append(np.array([]))  # Append empty array if frame cannot be read
            logger.warning(f"Failed to extract frame {idx} from {video_path}")

    cap.release()
    return frames


def extract_metrics_data(
    result_data: List[Dict[str, Any]],
) -> Tuple[List[str], List[List[float]], List[float]]:
    """
    处理原始 JSON 数据，将 'position' 从帧索引转换为秒，并提取 'video_path' 和 'frame_timestamps'。

    Args:
        result_data (List[Dict[str, Any]]): 从 JSON 文件加载的原始数据。
        fps_override (Optional[Dict[str, float]], optional): 
            覆盖特定视频的 FPS 值。键为 'video_path'，值为 FPS。
            默认为 None，表示自动加载每个视频的 FPS。

    Returns:
        Tuple[List[str], List[List[float]], List[float]]: 
            - video_paths (List[str]): 视频文件路径列表。
            - searching_sec (List[List[float]]): 每个视频预测的关键帧时间戳（秒）的列表。
            - gt_seconds (List[float]): 每个视频真实的关键帧时间（秒）的列表。

    Raises:
        KeyError: 如果某个条目缺少必需的字段。
        ValueError: 如果 FPS 加载失败。
    """
    video_paths = []
    searching_sec = []
    gt_seconds = []

    for idx, entry in enumerate(result_data):
        try:
            if "error" in entry.keys():
                continue
            video_path = entry['video_path']
            frame_timestamps = entry['keyframe_timestamps']
            if 'position' in entry.keys():
                positions = entry['position']
            if 'gt_frame_index' in entry.keys():
                positions = entry['gt_frame_index']

        except KeyError as e:
            logger.error(f"条目 {idx} 缺少必要字段: {e}")
            continue  # 跳过缺少必要字段的条目


        try:
            fps = load_video_fps(video_path)
        except ValueError as e:
            logger.error(f"获取视频 {video_path} 的 FPS 失败: {e}")
            continue  # 跳过无法获取 FPS 的条目

        # 将 'position' 从帧索引转换为秒
        gt_sec = [position / fps for position in positions]
        searching_sec.append(frame_timestamps)
        gt_seconds.append(gt_sec)
        video_paths.append(video_path)

        # logger.debug(f"处理视频 {video_path}: frame_timestamps={frame_timestamps}, gt_sec={gt_sec}")

    return video_paths, searching_sec, gt_seconds
def calculate_scc(list_gt, list_pred):
    total_score = 0.0
    total_frames = 0
    total_empty = 0
    scc=SpatialCorrelationCoefficient()
    # 逐样本处理
    for sample_idx in range(len(list_gt)):
        # 转换当前样本的GT帧 (F_gt, H, W, C)
        break_outer = False  
        
        
        gt_frames = []
        for f in list_gt[sample_idx]:
            
            # 判断空值条件（根据实际情况调整）
            if f.size == 0:  # 例如：numpy数组为空
                total_empty+=1
                break_outer = True
                break       # 跳出内层循环
            
            gt_frames.append(torch.from_numpy(f).float())
    
    # 如果检测到空值，直接终止外部循环
        if break_outer:
            continue  
        if len(gt_frames)==0:
            total_empty+=1
            print("find empty")
            continue
        gt_tensor = torch.stack(gt_frames)  # (F_gt, H, W, 3)
        #print("gt_tensor",gt_tensor.shape)
        
        # 转换当前样本的预测帧 (F_pred, H, W, C)
       #pred_frames = [torch.from_numpy(f).float() for f in list_pred[sample_idx]]
        pred_frames = []
        for f in list_pred[sample_idx]:
            
            # 判断空值条件（根据实际情况调整）
            if f.size == 0:  # 例如：numpy数组为空
                total_empty+=1
                break_outer = True
                break       # 跳出内层循环
            
            pred_frames.append(torch.from_numpy(f).float())

        if break_outer:
            continue 
        #判断preframe是否为空
        if len(pred_frames)==0:
            total_empty+=1
            print("find empty")
            continue
        pred_tensor = torch.stack(pred_frames)  # (F_pred, H, W, 3)
        #print("pred_tensor",pred_tensor.shape)
        
        # 调整维度顺序为通道优先 (F, C, H, W)
        gt_tensor = gt_tensor.permute(0, 3, 1, 2)  # (F_gt, 3, H, W)
        pred_tensor = pred_tensor.permute(0, 3, 1, 2)  # (F_pred, 3, H, W)
        
        # 计算当前样本所有预测帧的最大得分
        sample_score = 0.0
        for pred_frame in pred_tensor:
            # 添加batch维度 (1, C, H, W)
            pred = pred_frame.unsqueeze(0)
            
            # 计算与所有GT帧的相似度
            scores = []
            for gt_frame in gt_tensor:
                gt = gt_frame.unsqueeze(0)  # (1, C, H, W)
                #如果gt和pred的shape不一样，跳过
                if gt.shape!=pred.shape:
                    continue  
                scores.append(scc(preds=pred, target=gt))
                
            if not scores:
                continue
            sample_score += sum(sorted(scores, reverse=True)[:len(gt_frames)])
        
        # 累加总分数
        total_score += sample_score
        total_frames += len(gt_frames)
    print("total_empty",total_empty)
    print("total_frames",total_frames)
    #这里计算的是跟所有gt帧的最大相似度的和除以所有gt帧的数量
    return total_score / total_frames if total_frames > 0 else 0.0

# 使用示例

def calculate_metrics(
    result_data: List[Dict[str, Any]],
    fps: float = 30,
    threshold: int = 5,
    max_workers: int = 4
) -> Dict[str, Any]:
    """
    Calculate PRF and SSIM metrics for all video entries.

    Args:
        result_data (List[Dict[str, Any]]): List of video result entries.
        fps (float): Frames per second of the videos.
        threshold (int, optional): Distance threshold for PRF calculation. Defaults to 5.
        max_workers (int, optional): Number of threads for parallel processing. Defaults to 4.

    Returns:
        Dict[str, Any]: Dictionary containing average PRF and SSIM metrics.
    """
    #result_data = result_data[:10]
    video_paths, searching_sec, gt_seconds = extract_metrics_data(result_data=result_data)
    list_gt = []
    list_pred = []
    list_gt_images = []
    list_pred_images = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, (video_path, pred_timestamps, gt_positions) in enumerate(zip(video_paths, searching_sec, gt_seconds)):
            try:
                # Convert seconds to frame numbers
                gt_frame_nums = np.array(gt_positions, dtype=int)
                #fps = load_video_fps(video_path)
                pred_frame_nums = np.array([int(ts * fps) for ts in pred_timestamps], dtype=int)

                list_gt.append(gt_frame_nums)
                list_pred.append(pred_frame_nums[:8])
                # print(list_gt)
                # print(list_pred)
                # Combine gt and pred frame numbers for extraction
                combined_frames = gt_frame_nums.tolist() + pred_frame_nums.tolist()
                future = executor.submit(extract_frames, video_path, combined_frames)
                future_to_idx[future] = idx

            except Exception as e:
                logger.error(f"Error processing video {video_path}: {e}")
                list_gt.append(np.array([]))
                list_pred.append(np.array([]))
                list_gt_images.append([])
                list_pred_images.append([])

        # Process frame extraction with progress bar
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Extracting Frames"):
            idx = future_to_idx[future]
            try:
                frames = future.result()
                if not frames:
                    logger.warning(f"No frames extracted for video index {idx}")
                    list_gt_images.append([])
                    list_pred_images.append([])
                    continue

                entry = result_data[idx]
                if 'gt_frame_index' in entry.keys():
                    gt_num = len(entry['gt_frame_index'])
                pred_num = len(entry['keyframe_timestamps'])
                gt_images = frames[:gt_num]
                pred_images = frames[gt_num:gt_num + pred_num]

                list_gt_images.append(gt_images)
                
                list_pred_images.append(pred_images)

            except Exception as e:
                logger.error(f"Error extracting frames for video index {idx}: {e}")
                list_gt_images.append([])
                list_pred_images.append([])

    # Calculate PRF Scores
    logger.info("Calculating PRF Scores...")
    avg_precision, avg_recall, avg_f1 = calculate_prf(list_gt, list_pred, threshold=threshold)
    logger.info(f"Average Precision: {avg_precision:.4f}")
    logger.info(f"Average Recall: {avg_recall:.4f}")
    logger.info(f"Average F1 Score: {avg_f1:.4f}")

    # Calculate SSIM Scores
    logger.info("Calculating SSIM Scores...")
    ssim_scores = calculate_ssim_scores(list_gt_images, list_pred_images)
    if ssim_scores:
        avg_ssim_precision = np.mean([s[0] for s in ssim_scores])
        avg_ssim_recall = np.mean([s[1] for s in ssim_scores])
        logger.info(f"Average SSIM Precision: {avg_ssim_precision:.4f}")
        logger.info(f"Average SSIM Recall: {avg_ssim_recall:.4f}")
    else:
        avg_ssim_precision = 0.0
        avg_ssim_recall = 0.0
        logger.warning("No SSIM scores were calculated.")

    metrics = {
        "Average Precision": avg_precision,
        "Average Recall": avg_recall,
        "Average F1 Score": avg_f1,
        "Average SSIM Precision": avg_ssim_precision,
        "Average SSIM Recall": avg_ssim_recall
    }

    return metrics
def calculate_metrics_wy(
    result_data: List[Dict[str, Any]],
    fps: float = 30,
    threshold: int = 5,
    max_workers: int = 4
) -> Dict[str, Any]:
    """
    Calculate PRF and SSIM metrics for all video entries.

    Args:
        result_data (List[Dict[str, Any]]): List of video result entries.
        fps (float): Frames per second of the videos.
        threshold (int, optional): Distance threshold for PRF calculation. Defaults to 5.
        max_workers (int, optional): Number of threads for parallel processing. Defaults to 4.

    Returns:
        Dict[str, Any]: Dictionary containing average PRF and SSIM metrics.
    """
    result_data = result_data[:1]
    video_paths, searching_sec, gt_seconds = extract_metrics_data(result_data=result_data)
    list_gt = []
    list_pred = []
    list_gt_images = []
    list_pred_images = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, (video_path, pred_timestamps, gt_positions) in enumerate(zip(video_paths, searching_sec, gt_seconds)):
            try:
                # Convert seconds to frame numbers
               
                gt_frame_nums = np.array(gt_positions, dtype=int)
                #fps = load_video_fps(video_path)
                pred_frame_nums = np.array([int(ts * fps) for ts in pred_timestamps], dtype=int)

                list_gt.append(gt_frame_nums)
                list_pred.append(pred_frame_nums)
                print(list_gt)
                print(list_pred)
                # Combine gt and pred frame numbers for extraction
                combined_frames = gt_frame_nums.tolist() + pred_frame_nums.tolist()
                # print(combined_frames)
                future = executor.submit(extract_frames, video_path, combined_frames)
                future_to_idx[future] = idx

            except Exception as e:
                logger.error(f"Error processing video {video_path}: {e}")
                list_gt.append(np.array([]))
                list_pred.append(np.array([]))
                list_gt_images.append([])
                list_pred_images.append([])

        # Process frame extraction with progress bar
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Extracting Frames"):
            idx = future_to_idx[future]
            try:
                frames = future.result()
                if not frames:
                    logger.warning(f"No frames extracted for video index {idx}")
                    list_gt_images.append([])
                    list_pred_images.append([])
                    continue

                entry = result_data[idx]
                if 'position' in entry.keys():
                    gt_num = len(entry['position'])
                if 'gt_frame_index' in entry.keys():
                    gt_num = len(entry['gt_frame_index'])
                pred_num = len(entry['keyframe_timestamps'])
                
                gt_images = frames[:gt_num]
                pred_images = frames[gt_num:gt_num + pred_num]

                list_gt_images.append(gt_images)
                
                list_pred_images.append(pred_images)
        
            except Exception as e:
                logger.error(f"Error extracting frames for video index {idx}: {e}")
                list_gt_images.append([])
                list_pred_images.append([])
        # print("list_gt_images",len(list_gt_images))
        # print("list_pred_images",len(list_pred_images))
    # Calculate PRF Scores
    logger.info("Calculating PRF Scores...")
    avg_precision, avg_recall, avg_f1 = calculate_prf(list_gt, list_pred, threshold=threshold)
    logger.info(f"Average Precision: {avg_precision:.4f}")
    logger.info(f"Average Recall: {avg_recall:.4f}")
    logger.info(f"Average F1 Score: {avg_f1:.4f}")

    # Calculate SSIM Scores
    logger.info("Calculating SSIM Scores...")
  

    final_score = calculate_scc(list_gt_images, list_pred_images)
    print(f"Final Average Max Score: {final_score:.4f}")

    ssim_scores = calculate_ssim_scores(list_gt_images, list_pred_images)
    if ssim_scores:
        avg_ssim_precision = np.mean([s[0] for s in ssim_scores])
        avg_ssim_recall = np.mean([s[1] for s in ssim_scores])
        logger.info(f"Average SSIM Precision: {avg_ssim_precision:.4f}")
        logger.info(f"Average SSIM Recall: {avg_ssim_recall:.4f}")
    else:
        avg_ssim_precision = 0.0
        avg_ssim_recall = 0.0
        logger.warning("No SSIM scores were calculated.")

    metrics = {
        "Average Precision": avg_precision,
        "Average Recall": avg_recall,
        "Average F1 Score": avg_f1,
        "Average SSIM Precision": avg_ssim_precision,
        "Average SSIM Recall": avg_ssim_recall
    }

    return metrics

def main():
    """
    Main function to execute metrics calculation.
    """
    args = parse_arguments()

    # Load JSON data
    try:
        result_data = load_json_file(args.result_path)
    except Exception as e:
        logger.error(f"Failed to load JSON data: {e}")
        sys.exit(1)

    # Validate JSON structure
    required_fields = {'video_path', 'keyframe_timestamps'}
    for idx, entry in enumerate(result_data):
        if not required_fields.issubset(entry.keys()):
            logger.warning(f"Entry {idx} is missing required fields. Skipping.")
            continue

    # Calculate metrics
    metrics = calculate_metrics(
        result_data=result_data,
        fps=args.fps,
        threshold=args.threshold,
        max_workers=args.max_workers
    )

    # Save metrics to JSON file
    save_json_file(metrics, args.output_path)

    logger.info("Metrics calculation completed successfully.")


if __name__ == "__main__":
    main()

    #TBD bug on frame index or sec or fps?
