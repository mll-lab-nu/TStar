import os
import json
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    Get the frames-per-second (FPS) of the video.
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


def extract_frames(video_path: str, frame_indices: Optional[List[int]] = None, numframe: int = 8) -> List[Optional[Image.Image]]:
    """
    Extract specified frames from a video and convert them to PIL images.
    If frame_indices is None, uniformly sample 'numframe' frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_indices is None:
        frame_indices = np.linspace(0, total_frames - 1, numframe, dtype=int).tolist()

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

def match_answer(pred_answer, gt_answer):
    import re

    # Extract the first letter from the predicted answer and ground truth.
    pred_letter_match = re.match(r"^\s*([A-Fa-f])", pred_answer)
    gt_letter_match = gt_answer

    if pred_letter_match and gt_letter_match:
        pred_letter = pred_letter_match.group(1).lower()
        gt_letter = gt_letter_match.group(1).lower()
        correct = (pred_letter == gt_letter)
    else:
        # Fall back to full string comparison if regex fails.
        correct = (pred_answer.strip().lower() == gt_answer.strip().lower())

    return correct
def _submit_item_task(item: Dict[str, Any], fps_cache: Dict[str, float], frame_key: str) -> Tuple[str, Dict[str, Any]]:
    """
    Prepare the information for a frame extraction task for one item.
    Returns the video path and a dictionary with task details.
    """
    video_path = item['video_path']
    keyframe_timestamps = item.get("keyframe_timestamps", [])
    if frame_key == "uniform":
        num_frames = len(keyframe_timestamps)
        frame_indices = None
    else:
        keyframe_timestamps.sort()
        frame_indices = [int(ts * load_video_fps(video_path)) for ts in keyframe_timestamps]
        num_frames = len(frame_indices)

    # Initialize QA result fields
    item[f"{frame_key}_pred_answer"] = None
    item["correct"] = None
    return video_path, {"item": item, "num_frames": num_frames, "frame_indices": frame_indices}


def _process_future_result(future, qa_results: List[Dict[str, Any]], tstar_grounder: TStarUniversalGrounder,
                           result_data: List[Dict[str, Any]], frame_key: str, count: Dict[str, int]) -> None:
    """
    Process a completed future: extract frames, perform QA inference, update qa_results.
    The count dict is used to update the total and correct counts.
    """
    qa_idx = future.qa_idx  # Attached to the future
    try:
        frames = future.result()
    except Exception as e:
        logger.error(f"Error extracting frames for item {qa_idx}: {e}")
        qa_results[qa_idx][f"{frame_key}_pred_answer"] = "Frame extraction failed."
        qa_results[qa_idx]["correct"] = False
        return

    if not frames or len(frames) < 1:
        logger.warning(f"No frames extracted for item {qa_idx}.")
        return

    gt_image = frames[0]
    pred_images = frames[1:]
    if gt_image is None:
        logger.warning(f"GT frame extraction failed for item {qa_idx}.")
        qa_results[qa_idx]["pred_answer"] = "GT frame extraction failed."
        qa_results[qa_idx]["correct"] = False
        return

    try:
        pred_answer = tstar_grounder.inference_qa(
            frames=pred_images,
            question=result_data[qa_idx]['question'],
            options=result_data[qa_idx]['options'],
            temperature=0.2,
            max_tokens=1024
        )
        logger.info(f"item {qa_idx} QA answer: {pred_answer}, GT answer: {result_data[qa_idx]['gt_answer']}")

        gt_answer = qa_results[qa_idx]["gt_answer"].strip()
        pred_answer = pred_answer.strip()
        correct = match_answer(pred_answer=pred_answer, gt_answer=gt_answer)

        qa_results[qa_idx][f"{frame_key}_pred_answer"] = pred_answer
        qa_results[qa_idx]["correct"] = correct

        if correct:
            count["correct"] += 1
        count["total"] += 1
    except Exception as e:
        logger.error(f"QA inference failed for item {qa_idx}: {e}")
        qa_results[qa_idx][f"{frame_key}_pred_answer"] = "QA inference failed."
        qa_results[qa_idx]["correct"] = False


def compute_qa_accuracy(
    result_data: List[Dict[str, Any]],
    tstar_grounder: TStarUniversalGrounder,
    frame_key: str = "uniform",
    max_workers: int = 4,
    output_file: str = "./Rebuttal.json",
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Process result_data, perform QA inference, and compute QA accuracy.
    The results for each item are saved to a JSONL file whose name is derived from the input file name plus a marker.
    If the output file already exists, the function loads the existing results and computes accuracy from them.
    """


    # If output file exists, assume processing is done; load results and compute accuracy.
    if os.path.exists(output_file):
        logger.info(f"Output file {output_file} already exists. Loading previous results.")
        qa_results = []
        correct_count = 0
        total_count = 0
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                qa_results.append(item)
                if item.get("correct"):
                    correct_count += 1
                total_count += 1
        if total_count == 0:
            accuracy = 0.0
        else:
            accuracy = correct_count / total_count
        logger.info(f"QA Accuracy (loaded): {accuracy*100:.2f}% ({correct_count}/{total_count})")
        return accuracy, qa_results

    # Otherwise, process the result_data
    qa_results = []
    correct_count_dict = {"correct": 0, "total": 0}
    fps_cache = {}
    futures = []

    with open(output_file, "a", encoding="utf-8") as jsonl_file, ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, item in enumerate(result_data):
            try:
                video_path, task_info = _submit_item_task(item, fps_cache, frame_key)
                future = executor.submit(extract_frames, video_path, task_info["frame_indices"], task_info["num_frames"])
                future.qa_idx = idx
                # Initialize QA result fields in the item
                item[f"{frame_key}_pred_answer"] = None
                item["correct"] = None
                qa_results.append(item)
                futures.append(future)
            except KeyError as e:
                logger.error(f"item {idx} is missing key: {e}. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}. Skipping.")
                continue

        # Process futures as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting frames and performing QA"):
            qa_idx = getattr(future, "qa_idx", None)
            if qa_idx is None:
                continue
            try:
                _process_future_result(future, qa_results, tstar_grounder, result_data, frame_key, correct_count_dict)
            except Exception as e:
                logger.error(f"Error processing future for item {qa_idx}: {e}")
                qa_results[qa_idx][f"{frame_key}_pred_answer"] = "Processing failed."
                qa_results[qa_idx]["correct"] = False

            json.dump(qa_results[qa_idx], jsonl_file, ensure_ascii=False)
            jsonl_file.write("\n")

    if correct_count_dict["total"] == 0:
        logger.warning("No QA evaluations were performed.")
        accuracy = 0.0
    else:
        accuracy = correct_count_dict["correct"] / correct_count_dict["total"]

    logger.info(f"QA Accuracy: {accuracy*100:.2f}% ({correct_count_dict['correct']}/{correct_count_dict['total']})")
    return accuracy, qa_results


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="TStarSearcher: Video Frame Search and QA Tool")

    # Data meta processing arguments
    parser.add_argument('--backend', type=str, default="gpt-4o", help='The backend used for question qa.')
    parser.add_argument('--json_file', type=str, default="2025-03-22-07-33-52objnew_LVHaystack_gpt4_raw_vid1.json", help='The video dataset used for processing.')
    parser.add_argument('--frame_key', type=str, default="keyframe_timestamps", help='Frame sampling method.')
    parser.add_argument('--num_frame', type=int, default=8, help='The number of frames fed into qa model.')
    return parser.parse_args()

if __name__ == "__main__":
    np.random.seed(2025)
    args = parse_arguments()
    # 初始化 TStarUniversalGrounder
    tstar_grounder = TStarUniversalGrounder(
        model_name=args.backend,
    )

    # 加载 result_data 从 JSON 文件
    frame_search_root = "./results/frame_search"
    data_json_path = os.path.join(frame_search_root, args.json_file)
    with open(data_json_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)[0:4]
    
    output_root = "./results/last_version"
    os.makedirs(output_root, exist_ok=True)

    # Compute QA accuracy.
    # The output file name is generated from the input file name and saved in args.output_dir.
    accuracy, qa_results = compute_qa_accuracy(
        result_data=result_data,
        tstar_grounder=tstar_grounder,
        frame_key=args.frame_key,
        max_workers=1,
        output_file=os.path.join(output_root, args.json_file.replace(".json", "qa_" + str(args.num_frame) + "frames_" + args.backend + "_" + args.frame_key + ".json"))
    )

    print(f"QA Accuracy: {accuracy*100:.2f}%")
