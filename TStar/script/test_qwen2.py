import json
import cv2
from PIL import Image
from tqdm import tqdm
import random
import numpy as np

def get_sampled_frame_indexes(vclip_interval, num_samples, frame_distribution=None):
    """
    如果提供了 frame_distribution（每个位置是帧的打分），
    则在 [start, end] 区间内做带权采样；
    否则使用均匀采样。
    """
    start, end = int(vclip_interval[0]), int(vclip_interval[1])

    if frame_distribution:
        # 获取在 clip 区间内的 frame indexes 和对应 scores
        frame_indexes = list(range(start, end))
        scores = frame_distribution[start:end]

        # 如果所有分数为0，fallback
        if sum(scores) == 0:
            print("Warning: All scores are zero in frame_distribution, fallback to uniform sampling.")
            return uniform_sample_frame_indexes(vclip_interval, num_samples)

        # 转成概率分布
        prob_dist = np.array(scores) / np.sum(scores)

        # 带权采样，不能重复 -> replace=False
        if len(frame_indexes) <= num_samples:
            return frame_indexes
        else:
            sampled = np.random.choice(frame_indexes, size=num_samples, replace=False, p=prob_dist)
            return sorted(sampled.tolist())

    # fallback: 均匀采样
    return uniform_sample_frame_indexes(vclip_interval, num_samples)

def uniform_sample_frame_indexes(vclip_interval, num_samples):
    """
    给定 vclip_interval_in_video（假定为 [start, end]，单位为帧数）和期望采样的帧数，
    返回在该区间内均匀采样的帧索引列表。
    """
    start, end = vclip_interval
    if num_samples <= 1:
        return [int(round(start))]
    step = (end - start) / (num_samples - 1)
    indexes = [int(round(start + i * step)) for i in range(num_samples)]
    return indexes


def extract_frames(video_path, frame_indexes):
    """
    根据视频路径和帧索引列表，返回对应的 PIL 图像列表。
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx in frame_indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 将 BGR 转换为 RGB 格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        else:
            print(f"Unable to read frame {idx} from {video_path}")
    cap.release()
    return frames
from datasets import load_dataset
def align2clip(jsondata):
    # load 
        # # Load the dataset from the given source
    dataset_meta: str = "LVHaystack/LongVideoHaystack"
    split="test_tiny"
    dataset = load_dataset(dataset_meta) #, download_mode="force_redownload"
    
    # Extract the 'test' split from the dataset
    LVHaystact_testset = dataset[split]

    # # List to hold the transformed data
    TStar_format_data = []

    # Iterate over each row in the dataset
    # build dict
    video_question2clip = {}
    for idx, item in enumerate(LVHaystact_testset):
        video_id = item["video_id"]
        question =  item["question"]  
        video_metadata = item["video_metadata"]     
        vclip_interval_in_video = video_metadata["vclip_interval_in_video"]
        id = f"{video_id}_{question}"
        video_question2clip[id] = vclip_interval_in_video
        pass

    for idx, item in enumerate(jsondata):

        video_id = item["video_id"]
        question =  item["question"]
        id = f"{video_id}_{question}"
        item["vclip_interval_in_video"] = video_question2clip[id]
    
    return jsondata



def test_qa_accuracy_uniform_vclip(json_path, num_samples, qwen_interface, output_json_path):
    """
    加载 JSON 数据，对于每个样本：
      - 从 vclip_interval_in_video 区间内均匀采样 num_samples 帧，
      - 构造包含相应 <image> 占位符的 prompt，
      - 调用模型接口进行推理，
      - 计算并输出整体准确率。
    
    同时，将每个样本的预测结果和正确性标记写回 JSON，并保存至 output_json_path。
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        data = align2clip(data)[:200]
    
    correct = 0
    total = len(data)
    
    for idx, item in enumerate(tqdm(data, desc="Processing samples", unit="sample")):
        video_path = item.get("video_path", "")
        question = item.get("question", "")
        gt_answer = item.get("gt_answer", "")
        
        # 从 item 中获取 vclip_interval_in_video，假定格式为 [start, end]
        vclip_interval = item.get("vclip_interval_in_video", None)
        if vclip_interval is None:
            print("vclip_interval_in_video not found for sample, skipping.")
            continue
        
        # 均匀采样帧索引
        frame_distribution  = item.get("keyframe_distribution", None)
        
        # vclip_interval = [0, len(frame_distribution)]
        # frame_distribution  = None
        frame_indexes = get_sampled_frame_indexes(vclip_interval, num_samples, frame_distribution)
        frame_indexes = item.get("keyframe_timestamps", [])[:8]
        frame_indexes.sort()
        # 构造 prompt，<image> 标签的数量与采样帧数量一致
        prompt = (
            "Answer the following question briefly based on the video.\n" +
            "\n".join(["<image>"] * len(frame_indexes)) +
            f"\nQuestion: {question}\nPlease answer directly with the letter of the option."
        )
        
        # 根据采样帧索引提取对应帧图像
        frames = extract_frames(video_path, frame_indexes)
        
        # 使用模型接口进行推理
        pred = qwen_interface.inference(prompt, frames=frames)
        pred_clean = pred.strip().upper()
        gt_clean = gt_answer.strip().upper()
        
        # 将预测结果及正确性写入当前样本
        item["predicted_answer"] = pred_clean
        item["is_correct"] = (pred_clean == gt_clean)
        
        if pred_clean == gt_clean:
            correct += 1
        
        current_accuracy = (correct / (idx + 1)) * 100
        print(f"Video ID: {item.get('video_id', 'N/A')}, GT: {gt_answer}, Pred: {pred_clean}, Current Accuracy: {current_accuracy:.2f}%")
    
    overall_accuracy = (correct / total * 100) if total > 0 else 0
    print("Overall Accuracy: {:.2f}%".format(overall_accuracy))
    
    # 总结信息
    results_summary = {
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy_percent": overall_accuracy
    }
    
    output_data = {
        "results": data,
        "summary": results_summary
    }
    
    with open(output_json_path, 'w') as f_out:
        json.dump(output_data, f_out, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_json_path}")

if __name__ == "__main__":
    from TStar.interface_grounding_plus import QwenInterface
    # 初始化 QwenInterface 对象
    qwen_interface = QwenInterface(
        model_name="./pretrained/Qwen2.5-VL-72B-Instruct",
    )
    
    # 定义输入输出 JSON 文件路径（确保路径正确）
    json_path = "results/frame_search/yolo-World_TStar_LongVideoHaystack_test.json"
    json_path = "/data/guoweiyu/LV-Haystack/results/frame_search/2025-03-22-07-33-52objnew_LVHaystack_gpt4_raw_vid1.json"
    
    output_json_path = "./results/frame_search/TStar_LongVideoHaystack_tiny_with_uniform_predictions.json"
    
    # 指定在 vclip_interval_in_video 区间内采样的帧数量（可根据需要调整）
    num_samples = 8
    
    test_qa_accuracy_uniform_vclip(json_path, num_samples, qwen_interface, output_json_path)
