import os
import json
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from TStar.interface_grounding import TStarUniversalGrounder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_video_fps(video_path: str) -> float:
    """
    Retrieve the frames-per-second (FPS) of the video.

    Args:
        video_path: Path to the video file.

    Returns:
        The FPS value.

    Raises:
        ValueError: If the video cannot be opened or FPS cannot be retrieved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Unable to open video file: {video_path}")
        raise ValueError(f"Unable to open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps == 0:
        logger.error(f"Unable to retrieve FPS for video: {video_path}")
        raise ValueError(f"Unable to retrieve FPS for video: {video_path}")
    logger.debug(f"Video {video_path} FPS: {fps}")
    return fps


def extract_frames(
    video_path: str,
    item,
    frame_distribution: Optional[List[float]] = None,
    num_frames: int = 8,
    p_fps: int = 1,
    duration_type: str = "video"
) -> List[Optional[Image.Image]]:
    """
    Extract frames from a video, with optional support for clip-based sampling and 
    score-based distribution (assumed 1 FPS alignment).

    Args:
        video_path: Path to video file.
        item: Metadata dict, must contain `vclip_interval_in_video` if clip mode is used.
        frame_distribution: List of scores (per-second), used to sample frames if given.
        num_frames: Number of frames to sample.
        p_fps: Probability fps (default 1 FPS score-aligned).
        duration_type: "video" or "clip".

    Returns:
        List of PIL Image frames (some may be None if reading fails).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration_sec = total_frames / video_fps

    # Determine sampling interval
    if duration_type == "clip":
        start_sec, end_sec = item.get("vclip_interval_in_video", [0, video_duration_sec])
    else:
        start_sec, end_sec = 0, video_duration_sec

    # Clamp within bounds
    start_sec = max(0, start_sec)
    end_sec = min(video_duration_sec, end_sec)

    # --- Sampling strategy ---
    if frame_distribution is not None:
        dist = np.nan_to_num(np.array(frame_distribution, dtype=np.float32), nan=0.0)
        if dist.sum() == 0:
            dist = np.ones_like(dist)

        # Clip-aware slicing
        clip_start_idx = int(start_sec)
        clip_end_idx = int(end_sec)
        dist_clip = dist[clip_start_idx:clip_end_idx]

        if dist_clip.sum() == 0:
            dist_clip = np.ones_like(dist_clip)

        dist_clip /= dist_clip.sum()

        sampled_secs_in_clip = np.random.choice(
            len(dist_clip), size=num_frames, replace=False, p=dist_clip
        )
        sampled_secs = sampled_secs_in_clip + clip_start_idx
        sampled_secs.sort()

    else:
        # Uniform sampling
        sampled_secs = np.linspace(start_sec, end_sec, num_frames, dtype=int)

    # Convert seconds to frame indices
    frame_indices = [int(sec * video_fps / p_fps) for sec in sampled_secs]
    frame_indices = [min(max(0, idx), total_frames - 1) for idx in frame_indices]

    # --- Frame extraction ---
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        else:
            frames.append(None)

    cap.release()
    return frames


def match_answer(predicted: str, ground_truth: str) -> bool:
    """
    Check whether the predicted answer matches the ground truth answer.
    The check compares the first letter (if available) or the full string.

    Args:
        predicted: The predicted answer.
        ground_truth: The ground truth answer.

    Returns:
        True if the answers match, False otherwise.
    """
    import re
    match = re.match(r"^\s*([A-Fa-f])", predicted)
    if match:
        return match.group(1).lower() == ground_truth.strip().lower()
    else:
        return predicted.strip().lower() == ground_truth.strip().lower()


def _submit_item_task(item: Dict[str, Any],
                      nframes: int,
                      sampling_type: str,
                      p_fps: int = 1) -> Tuple[str, Dict[str, Any]]:
    """
    Prepare frame extraction parameters for a single item.

    Args:
        item: Data dictionary for one item.
        sampling_type: Frame sampling method key (e.g., "uniform" or "TStar").

    Returns:
        Tuple containing the video path and a task info dictionary.
    """
    video_path = item['video_path']
    
    if sampling_type == "uniform":
        keyframe_distribution = None
    elif sampling_type == "TStar":
        keyframe_distribution = item.get("keyframe_distribution", None)
    else:
        raise NotImplementedError(f"Sampling type '{sampling_type}' is not implemented. Choose from [uniform, TStar]")
    # Initialize QA result fields
    item[f"{sampling_type}_pred_answer"] = None
    item["correct"] = None
    return video_path, {"item": item, "num_frames": nframes, "keyframe_distribution": keyframe_distribution, "p_fps": p_fps}


def compute_qa_accuracy(result_data: List[Dict[str, Any]],
                        vlm_model: TStarUniversalGrounder,
                        nframe: int = 8,
                        sampling_type: str = "uniform",
                        duration_type: str = "video",
                        output_file: str = "./Rebuttal.json") -> Tuple[float, List[Dict[str, Any]]]:
    """
    Compute the QA accuracy by processing each item in result_data sequentially.
    
    If the output file exists, load existing results and skip already processed items.
    All results, including previously processed ones, are included in the accuracy calculation.

    Args:
        result_data: List of data items.
        vlm_model: The QA inference model.
        sampling_type: Key for frame sampling (e.g., "uniform" or "TStar").
        output_file: Path to the JSONL file for saving results.

    Returns:
        Tuple containing the accuracy and the list of QA results.
    """
    # Step 1: Load existing results from output_file
    existing_results = {}
    if os.path.exists(output_file):
        logger.info(f"Output file {output_file} already exists. Loading previous results.")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                # Assuming video_path is the unique identifier
                existing_results[item['video_path']] = item

    qa_results = []
    count = {"correct": 0, "total": 0}

    # Create a tqdm progress bar over the result data
    pbar = tqdm(result_data, desc="Processing samples", total=len(result_data))
    
    # Open output file for appending new results
    with open(output_file, "a", encoding="utf-8") as jsonl_file:
        for idx, item in enumerate(pbar):
            video_path = item.get('video_path')
            # Step 2: Skip already processed items and update statistics
            if video_path in existing_results:
                processed_item = existing_results[video_path]
                qa_results.append(processed_item)
                if processed_item.get("correct"):
                    count["correct"] += 1
                count["total"] += 1
                pbar.set_postfix(acc=f"{(count['correct']/count['total']*100):.2f}%" if count["total"] > 0 else "0%")
                continue

            # Submit task for items that have not been processed
            try:
                video_path, task_info = _submit_item_task(item, nframes=nframe, sampling_type=sampling_type)
            except Exception as e:
                logger.error(f"Error preparing item {idx}: {e}")
                continue

            # Sequentially process the item
            try:
                frames = extract_frames(
                    video_path,
                    item,
                    task_info["keyframe_distribution"],
                    task_info["num_frames"],
                    task_info["p_fps"],
                    duration_type=duration_type
                )
            except Exception as e:
                logger.error(f"Error extracting frames for item {idx}: {e}")
                item[f"{sampling_type}_pred_answer"] = "Frame extraction failed."
                item["correct"] = False
                qa_results.append(item)
                json.dump(item, jsonl_file, ensure_ascii=False)
                jsonl_file.write("\n")
                pbar.set_postfix(acc=f"{(count['correct']/count['total']*100):.2f}%" if count["total"] > 0 else "0%")
                continue

            try:
                pred_answer = vlm_model.inference_qa(
                    frames=frames,
                    question=item['question'],
                    options=item['options'],
                    temperature=0.2,
                    max_tokens=1024
                )
                logger.info(f"Item {idx} QA answer: {pred_answer}, GT answer: {item['gt_answer']}")

                gt_answer = item["gt_answer"].strip()
                pred_answer = pred_answer.strip()
                is_correct = match_answer(predicted=pred_answer, ground_truth=gt_answer)

                item[f"{sampling_type}_pred_answer"] = pred_answer
                item["correct"] = is_correct

                if is_correct:
                    count["correct"] += 1
                count["total"] += 1
            except Exception as e:
                logger.error(f"QA inference failed for item {idx}: {e}")
                item[f"{sampling_type}_pred_answer"] = "QA inference failed."
                item["correct"] = False

            qa_results.append(item)
            # Write new result to file
            json.dump(item, jsonl_file, ensure_ascii=False)
            jsonl_file.write("\n")
            pbar.set_postfix(acc=f"{(count['correct']/count['total']*100):.2f}%" if count["total"] > 0 else "0%")

    # Step 4: Calculate overall accuracy
    if count["total"] == 0:
        logger.warning("No QA evaluations were performed.")
        accuracy = 0.0
    else:
        accuracy = count["correct"] / count["total"]

    logger.info(f"QA Accuracy: {accuracy * 100:.2f}% ({count['correct']}/{count['total']})")
    return accuracy, qa_results


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="TStarSearcher: Video Frame Search and QA Tool")
    parser.add_argument('--backend', type=str, default="./pretrained/Qwen2.5-VL-7B-Instruct", help='The backend used for QA. quick start by [gpt4o, Qwen/Qwen2.5-VL-7B-Instruct]')
    parser.add_argument('--json_file', type=str, default="2025-03-22-07-33-52objnew_LVHaystack_gpt4_raw_vid1.json",
                        help='The video dataset used for processing.')
    parser.add_argument('--sampling_type', type=str, default="uniform", help='Frame sampling method.')
    parser.add_argument('--num_frame', type=int, default=8, help='The number of frames fed into QA model.')
    parser.add_argument('--duration_type', type=str, default="video", help='QA on full video or shorter clip.')
    return parser.parse_args()

if __name__ == "__main__":
    np.random.seed(2025)
    args = parse_arguments()

    # Initialize TStarUniversalGrounder
    tstar_grounder = TStarUniversalGrounder(model_name=args.backend)

    # Load result_data from the JSON file
    # frame_search_root = "./results/frame_search"
    data_json_path = args.json_file
    with open(data_json_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    output_root = "./results/last_version"
    os.makedirs(output_root, exist_ok=True)
    backend_name = args.backend.replace("/", "_")
    output_file = os.path.join(
        output_root,
        args.json_file.replace(".json", f"qa_{args.num_frame}frames_{backend_name}_{args.duration_type}_{args.sampling_type}.json")
    )

    # Compute QA accuracy and write results sequentially
    accuracy, qa_results = compute_qa_accuracy(
        result_data=result_data,
        vlm_model=tstar_grounder,
        nframe=args.num_frame,
        sampling_type=args.sampling_type,
        duration_type=args.duration_type,
        output_file=output_file
    )

    print(f"QA Accuracy: {accuracy * 100:.2f}%")
    print(f"Results saved to {output_file}")
