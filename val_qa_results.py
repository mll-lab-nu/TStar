import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
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

def extract_frames(video_path: str, frame_indices: List[int] = None, numframe: int = 8) -> List[Optional[Image.Image]]:
    """
    从视频中提取指定的帧，并转换为 PIL 图像。如果没有提供帧索引，则均匀地采样指定数量的帧。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数

    # 如果没有提供 frame_indices，进行均匀采样
    if frame_indices is None:
        frame_indices = np.linspace(0, total_frames - 1, numframe, dtype=int).tolist()

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        else:
            frames.append(None)  # 使用 None 表示无法读取帧

    cap.release()
    return frames

def compute_qa_accuracy(
    result_data: List[Dict[str, Any]],
    tstar_grounder: TStarUniversalGrounder,
    frame_key: str ="uniform",
    ground_truth_key: str = "gt_answer",
    max_workers: int = 4,
    output_file: str = "Rebuttal/qa_results.jsonl"
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    处理 result_data，执行 QA 推理，并计算 QA 准确率，动态保存每个条目的 QA 结果为 JSONL 格式。
    """
    qa_results = []
    correct_count = 0
    total_count = 0

    # 缓存 FPS 以避免重复加载
    fps_cache = {}

    # 打开 JSONL 文件准备追加
    with open(output_file, "a", encoding="utf-8") as jsonl_file, ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_qa_idx = {}

        for idx, entry in enumerate(result_data):
            try:
                video_path = entry['video_path']
                frame_timestamps = entry.get("frame_timestamps", [])
                question = entry['question']
                options = entry['options']
                gt_answer = entry.get(ground_truth_key, "None")

                if video_path in fps_cache:
                    fps = fps_cache[video_path]
                else:
                    try:
                        fps = load_video_fps(video_path)
                        fps_cache[video_path] = fps
                    except ValueError as e:
                        logger.error(f"获取视频 {video_path} 的 FPS 失败: {e}")
                        continue

                # 提交帧提取任务
                if frame_key == "uniform":
                    future = executor.submit(extract_frames, video_path, None)
                else: 
                    frame_timestamps.sort()
                    pred_frame_nums = [int(ts * fps) for ts in frame_timestamps]
                    future = executor.submit(extract_frames, video_path, pred_frame_nums)

                future_to_qa_idx[future] = len(qa_results)  # 映射到 qa_results 列表的索引

                # 初始化 qa_results 条目
                entry[f"{frame_key}_pred_answer"] = None
                entry["correct"] = None
                qa_results.append(entry)

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
                    pred_answer = tstar_grounder.inference_openend_qa(
                        frames=pred_images,
                        question=result_data[qa_idx]['question'],
                        # options=result_data[qa_idx]['options'],
                        temperature=0.2,
                        max_tokens=1024
                    )
                    print(f"条目 {qa_idx} 的 QA 答案: {pred_answer}")

                    # 比较预测答案与真实答案（忽略大小写和前后空格）
                    gt_answer_clean = qa_results[qa_idx]["gt_answer"].strip().lower()
                    pred_answer_clean = pred_answer.strip().lower()

                    correct = (pred_answer_clean == gt_answer_clean)
                    qa_results[qa_idx][f"{frame_key}_pred_answer"] = pred_answer
                    qa_results[qa_idx]["correct"] = correct

                    if correct:
                        correct_count += 1
                    total_count += 1

                except Exception as e:
                    logger.error(f"条目 {qa_idx} 的 QA 推理失败: {e}")
                    qa_results[qa_idx][f"{frame_key}_pred_answer"] = "QA 推理失败。"
                    qa_results[qa_idx]["correct"] = False

            except Exception as e:
                logger.error(f"提取帧或执行 QA 推理时发生错误 for 条目 {qa_idx}: {e}")
                qa_results[qa_idx][f"{frame_key}_pred_answer"] = "处理失败。"
                qa_results[qa_idx]["correct"] = False

            # 将每个条目的结果以 JSON 格式写入文件
            json.dump(qa_results[qa_idx], jsonl_file, ensure_ascii=False)
            jsonl_file.write("\n")

    if total_count == 0:
        logger.warning("没有执行任何 QA 评估。")
        accuracy = 0.0
    else:
        accuracy = correct_count / total_count

    logger.info(f"QA 准确率: {accuracy*100:.2f}% ({correct_count}/{total_count})")

    return accuracy, qa_results


if __name__ == "__main__":
    # 初始化 TStarUniversalGrounder
    tstar_grounder = TStarUniversalGrounder(
        backend="gpt4",
        num_frames=8
    )

    # 加载 result_data 从 JSON 文件
    data_json_path = "test_data.csv.json"
    with open(data_json_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)[0:20]

    # 计算 QA 准确率
    accuracy, qa_results = compute_qa_accuracy(
        result_data=result_data,
        tstar_grounder=tstar_grounder,
        ground_truth_key="answer",  # 根据实际 JSON 结构调整
        frame_key="uniform",
        max_workers=1,
        output_file="Rebuttal/all_openqa_results.jsonl"
    )

    # 打印准确率
    print(f"QA Accuracy: {accuracy*100:.2f}%")
