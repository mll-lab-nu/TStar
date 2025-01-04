import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from TStar.interface_llm import TStarUniversalGrounder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_video_fps(video_path: str) -> float:
    """
    获取视频的帧率（FPS）。

    Args:
        video_path (str): 视频文件的路径。

    Returns:
        float: 视频的帧率。

    Raises:
        ValueError: 如果无法打开视频文件或无法获取 FPS。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频文件: {video_path}")
        raise ValueError(f"无法打开视频文件: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps == 0:
        logger.error(f"无法获取视频帧率（FPS）: {video_path}")
        raise ValueError(f"无法获取视频帧率（FPS）: {video_path}")
    logger.debug(f"视频 {video_path} 的 FPS: {fps}")
    return fps

def extract_frames(video_path: str, frame_indices: List[int]) -> List[Optional[Image.Image]]:
    """
    从视频中提取指定的帧，并转换为 PIL 图像。

    Args:
        video_path (str): 视频文件的路径。
        frame_indices (List[int]): 要提取的帧索引列表。

    Returns:
        List[Optional[Image.Image]]: 提取的帧图像列表（PIL 格式）。如果无法提取某帧，则为 None。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频文件: {video_path}")
        raise ValueError(f"无法打开视频文件: {video_path}")

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            logger.debug(f"提取视频 {video_path} 的帧 {idx}")
        else:
            frames.append(None)  # 使用 None 表示无法读取帧
            logger.warning(f"无法提取视频 {video_path} 的帧 {idx}")
    cap.release()
    return frames

def compute_qa_accuracy(
    result_data: List[Dict[str, Any]],
    tstar_grounder: TStarUniversalGrounder,
    fps_override: Optional[Dict[str, float]] = None,
    ground_truth_key: str = "gt_answer",
    max_workers: int = 4
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    处理 result_data，执行 QA 推理，并计算 QA 准确率。

    Args:
        result_data (List[Dict[str, Any]]): 视频结果条目列表。每个条目应包含：
            - 'video_path': str
            - 'frame_timestamps': List[float]
            - 'position': int (帧索引)
            - 'question': str
            - 'options': str
            - 'gt_answer': str (真实答案)
        tstar_grounder (TStarUniversalGrounder): 用于执行 QA 推理的实例。
        fps_override (Optional[Dict[str, float]], optional): 
            覆盖特定视频的 FPS 值。键为 'video_path'，值为 FPS。默认为 None。
        ground_truth_key (str, optional): 条目中真实答案的键名。默认为 "gt_answer"。
        max_workers (int, optional): 并行处理的最大线程数。默认为 4。

    Returns:
        Tuple[float, List[Dict[str, Any]]]: 
            - QA 准确率（0 到 1 之间的浮点数）。
            - 包含每个视频的结果列表，每个结果为一个字典，包含：
                - 'video_path': str
                - 'pred_answer': str
                - 'gt_answer': str
                - 'correct': bool
    """
    qa_results = []
    correct_count = 0
    total_count = 0

    # 缓存 FPS 以避免重复加载
    fps_cache = {}

    # 初始化 ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_qa_idx = {}
        for idx, entry in enumerate(result_data):
            try:
                video_path = entry['video_path']
                frame_timestamps = entry['frame_timestamps']
                position = entry['position']
                question = entry['question']
                options = entry['options']
                gt_answer = entry.get(ground_truth_key)

                # 验证必要字段
                if not all([video_path, frame_timestamps, position is not None, question, options, gt_answer]):
                    logger.warning(f"条目 {idx} 缺少必要字段。跳过。")
                    continue

                # 获取 FPS
                if fps_override and video_path in fps_override:
                    fps = fps_override[video_path]
                    logger.debug(f"使用覆盖的 FPS {fps} for 视频: {video_path}")
                elif video_path in fps_cache:
                    fps = fps_cache[video_path]
                else:
                    try:
                        fps = load_video_fps(video_path)
                        fps_cache[video_path] = fps
                    except ValueError as e:
                        logger.error(f"获取视频 {video_path} 的 FPS 失败: {e}")
                        continue

                # 将 'position' 从帧索引转换为秒
                gt_sec = position / fps

                # 将 'frame_timestamps'（秒）转换为帧索引
                pred_frame_nums = [int(ts * fps) for ts in frame_timestamps]

                # 合并 gt 帧和 pred 帧索引
                combined_frame_indices = [position] + pred_frame_nums

                # 提交帧提取任务
                future = executor.submit(extract_frames, video_path, combined_frame_indices)
                future_to_qa_idx[future] = len(qa_results)  # 映射到 qa_results 列表的索引

                # 初始化 qa_results 条目
                qa_results.append({
                    "video_path": video_path,
                    "pred_answer": None,
                    "gt_answer": gt_answer,
                    "correct": False
                })

            except KeyError as e:
                logger.error(f"条目 {idx} 缺少键: {e}。跳过。")
                continue
            except Exception as e:
                logger.error(f"处理条目 {idx} 时发生错误: {e}。跳过。")
                continue

        # 处理帧提取和 QA 推理
        for future in tqdm(as_completed(future_to_qa_idx), total=len(future_to_qa_idx), desc="提取帧并执行 QA"):
            qa_idx = future_to_qa_idx[future]
            try:
                frames = future.result()
                if not frames or len(frames) < 1:
                    logger.warning(f"无法提取帧用于条目 {qa_idx}。")
                    continue

                gt_image = frames[0]
                pred_images = frames[1:]

                if gt_image is None:
                    logger.warning(f"条目 {qa_idx} 的 gt 帧提取失败。")
                    qa_results[qa_idx]["pred_answer"] = "无法提取 gt 帧。"
                    qa_results[qa_idx]["correct"] = False
                    continue

                # 执行 QA 推理
                try:
                    # 使用预测帧执行 QA 推理
                    pred_answer = tstar_grounder.inference_qa(
                        frames=pred_images,
                        question=result_data[qa_idx]['question'],
                        options=result_data[qa_idx]['options'],
                        temperature=0.2,
                        max_tokens=30
                    )
                    logger.debug(f"条目 {qa_idx} 的 QA 答案: {pred_answer}")

                    # 比较预测答案与真实答案（忽略大小写和前后空格）
                    gt_answer_clean = qa_results[qa_idx]["gt_answer"].strip().lower()
                    pred_answer_clean = pred_answer.strip().lower()

                    correct = (pred_answer_clean == gt_answer_clean)
                    qa_results[qa_idx]["pred_answer"] = pred_answer
                    qa_results[qa_idx]["correct"] = correct

                    if correct:
                        correct_count += 1
                    total_count += 1

                except Exception as e:
                    logger.error(f"条目 {qa_idx} 的 QA 推理失败: {e}")
                    qa_results[qa_idx]["pred_answer"] = "QA 推理失败。"
                    qa_results[qa_idx]["correct"] = False

            except Exception as e:
                logger.error(f"提取帧或执行 QA 推理时发生错误 for 条目 {qa_idx}: {e}")
                qa_results[qa_idx]["pred_answer"] = "处理失败。"
                qa_results[qa_idx]["correct"] = False

    if total_count == 0:
        logger.warning("没有执行任何 QA 评估。")
        accuracy = 0.0
    else:
        accuracy = correct_count / total_count

    logger.info(f"QA 准确率: {accuracy*100:.2f}% ({correct_count}/{total_count})")

    return accuracy, qa_results


import json
from TStar.interface_llm import TStarUniversalGrounder
if __name__ == "__main__":
    
    # 初始化 TStarUniversalGrounder
    tstar_grounder = TStarUniversalGrounder(
        backend="llava",
        model_path="/path/to/llava_model",
        model_base="model_base_version",
        num_frames=8
    )

    # 加载 result_data 从 JSON 文件
    with open("result_data.json", "r", encoding="utf-8") as f:
        result_data = json.load(f)

    # 可选：加载 FPS 覆盖数据
    fps_override = {}
    if os.path.exists("fps_override.json"):
        with open("fps_override.json", "r", encoding="utf-8") as f:
            fps_override = json.load(f)

    # 计算 QA 准确率
    accuracy, qa_results = compute_qa_accuracy(
        result_data=result_data,
        tstar_grounder=tstar_grounder,
        fps_override=fps_override,
        ground_truth_key="gt_answer",  # 根据实际 JSON 结构调整
        max_workers=4
    )

    # 打印准确率
    print(f"QA Accuracy: {accuracy*100:.2f}%")

    # 可选：保存详细的 QA 结果
    with open("qa_results.json", "w", encoding="utf-8") as f:
        json.dump(qa_results, f, indent=4, ensure_ascii=False)
