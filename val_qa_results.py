import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
from TStar.interface_grounding import TStarUniversalGrounder
import argparse
import datetime
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


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

def extract_frames(video_path: str, frame_indices: List[int] = None, numframe: int = 16) -> List[Optional[Image.Image]]:
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
            if numframe > 8:
                w, h = pil_image.size
                pil_image = pil_image.resize((int(w/2), int(h/2)), Image.Resampling.LANCZOS)
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
) -> Tuple[Dict, List[Dict[str, Any]]]:
    """
    处理 result_data，执行 QA 推理，并计算 QA 准确率，动态保存每个条目的 QA 结果为 JSONL 格式。
    """
    qa_results = []
    correct_count = 0
    total_count = 0
    # 缓存 FPS 以避免重复加载
    fps_cache = {}

    # 打开 JSONL 文件准备追加
    # with open(output_file, "w", encoding="utf-8") as jsonl_file, ThreadPoolExecutor(max_workers=max_workers) as executor:
    with open(output_file, "w", encoding="utf-8") as jsonl_file:
        future_to_qa_idx = {}
          
        # 分类别统计准确率
        for idx, item in tqdm(enumerate(result_data), desc="提取帧并执行 QA"):
            try:


                video_path = item['video_path']
                
                frame_timestamps = item.get("keyframe_timestamps", [])[:16]
                question = item['question']
                options = item['options']
                gt_answer = item.get(ground_truth_key, "None")

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
                    # future = executor.submit(extract_frames, video_path, None)
                    frames = extract_frames(video_path, None)

                else: 
                    frame_timestamps.sort()
                    
                    pred_frame_nums = [int(ts * fps) for ts in frame_timestamps]
                    
                    # future = executor.submit(extract_frames, video_path, pred_frame_nums)
                    frames = extract_frames(video_path, pred_frame_nums, numframe=len(pred_frame_nums))
                    
                # future_to_qa_idx[future] = len(qa_results)  # 映射到 qa_results 列表的索引
                

                # 初始化 qa_results 条目
                item[f"{frame_key}_pred_answer"] = None
                item["correct"] = None
                score_distribution = item.pop("score_distribution")
                if not frames or len(frames) < 1:
                    logger.warning(f"无法提取帧用于条目 {idx}。")
                    item["correct"] = False
                else:
                    try:
                        # 使用预测帧执行 QA 推理 
                        pred_answer = tstar_grounder.inference_qa(
                            frames=frames,
                            question=question,
                            options=options,
                            temperature=0.2,
                        )
                        print(f"条目 {idx} 的 QA 答案: {pred_answer}")

                        # 比较预测答案与真实答案（忽略大小写和前后空格）
                        gt_answer_clean = gt_answer.strip().lower()
                        pred_answer_clean = pred_answer.strip().lower()

                        correct = (pred_answer_clean == gt_answer_clean)
                        item[f"{frame_key}_pred_answer"] = pred_answer
                        item["correct"] = correct

                        if correct:
                            correct_count += 1
                        total_count += 1
                    except Exception as e:
                        logger.error(f"条目 {idx} 的 QA 推理失败: {e}")
                        item[f"{frame_key}_pred_answer"] = "QA 推理失败。"
                        item["correct"] = False

            except Exception as e:
                logger.error(f"提取帧或执行 QA 推理时发生错误 for 条目 {idx}: {e}")
                item[f"{frame_key}_pred_answer"] = "处理失败。"
                item["correct"] = False

            qa_results.append(item)
            if (idx + 1) % 50 == 0 or idx == (len(result_data) - 1):
                json.dump(qa_results, jsonl_file, ensure_ascii=False)

        
    if total_count == 0:
        logger.warning("没有执行任何 QA 评估。")
        accuracy = 0.0
    else:
        accuracy = correct_count / total_count

    logger.info(f"QA 准确率: {accuracy*100:.2f}% ({correct_count}/{total_count})")

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
    parser.add_argument('--json_file', type=str, default="LongVideoHaystack_tiny.json", help='The video dataset used for processing.')
    parser.add_argument('--frame_key', type=str, default="keyframe_", help='Frame sampling method.')
    parser.add_argument('--num_frame', type=int, default=8, help='The number of frames fed into qa model.')
    return parser.parse_args()

if __name__ == "__main__":
    
    np.random.seed(2025)
    args = parse_arguments()
    # 初始化 TStarUniversalGrounder
    tstar_grounder = TStarUniversalGrounder(
        model_name=args.backend
    )


    # 加载 result_data 从 JSON 文件
    frame_search_root = "./results/frame_search"
    data_json_path = os.path.join(frame_search_root, args.json_file)
    with open(data_json_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)[0:20]

    output_root = "./results/last_version"
    os.makedirs(output_root, exist_ok=True)
    # 计算 QA 准确率
    accuracy, qa_results = compute_qa_accuracy(
        result_data=result_data,
        tstar_grounder=tstar_grounder,
        ground_truth_key="gt_answer",  # 根据实际 JSON 结构调整
        frame_key=args.frame_key,  
        max_workers=1,
        output_file=os.path.join(output_root, args.json_file.replace(".json", "qa_" + str(args.num_frame) + "frames_" + args.backend + "_" + args.frame_key + ".json"))
    )

    # # 打印准确率
    print(f"QA Accuracy: {accuracy*100:.2f}%")


      