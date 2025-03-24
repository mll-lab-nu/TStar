import json
import cv2
from PIL import Image

def extract_frames(video_path, frame_indexes):
    """
    根据给定视频路径和帧索引列表，返回对应的 PIL 图像列表。
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx in frame_indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # OpenCV读取的帧为 BGR 格式，转换为 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        else:
            print(f"无法从 {video_path} 中读取帧 {idx}")
    cap.release()
    return frames

def test_qa_accuracy(json_path, qwen_interface, output_json_path):
    """
    加载 JSON 数据，对每个样本提取视频帧，构造查询文本，
    调用模型推理，并计算总体准确率。
    
    同时，将每个样本的预测答案及是否正确加入 JSON 中，最后保存到 output_json_path 文件中。
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    correct = 0
    total = len(data)
    
    for item in data:
        video_path = item.get("video_path", "")
        question = item.get("question", "")
        options = item.get("options", "")
        gt_answer = item.get("gt_answer", "")
        # 使用 keyframe_timestamps 列表作为帧的索引
        frame_indexes = item.get("keyframe_timestamps", [])
        
        # 构造系统提示：包含对应数量的 <image> 标签、问题以及选项说明
        prompt = (
            "Answer the following question briefly based on the video.\n" +
            "\n".join(["<image>"] * len(frame_indexes)) +
            f"\nQuestion: {question}\n请直接用选项的字母回答"
        )
        
        # 提取视频中对应帧
        frames = extract_frames(video_path, frame_indexes)
        
        # 调用模型推理得到预测答案
        pred = qwen_interface.inference(prompt, frames=frames)
        pred_clean = pred.strip().upper()
        gt_clean = gt_answer.strip().upper()
        
        # 将预测答案和正确标志写入当前样本
        item["predicted_answer"] = pred_clean
        item["is_correct"] = (pred_clean == gt_clean)
        
        if pred_clean == gt_clean:
            correct += 1

        print(f"视频ID: {item.get('video_id', 'N/A')}\n真实答案: {gt_answer}\n预测答案: {pred}\n")
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print("总体准确率: {:.2f}%".format(accuracy))
    
    # 添加总体统计信息
    results_summary = {
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy_percent": accuracy
    }
    
    # 将更新后的数据和统计信息一起保存到新的 JSON 文件中
    output_data = {
        "results": data,
        "summary": results_summary
    }
    with open(output_json_path, 'w') as f_out:
        json.dump(output_data, f_out, indent=2, ensure_ascii=False)
    print(f"结果已保存到 {output_json_path}")

if __name__ == "__main__":
    from TStar.interface_grounding_plus import QwenInterface
    # 实例化 QwenInterface 对象
    qwen_interface = QwenInterface(
        model_name="./pretrained/Qwen2.5-VL-7B-Instruct",
    )
    json_path = "/data/guoweiyu/LV-Haystack/results/frame_search/yolo-World_TStar_LVHaystack_tiny.json"  # 请确保路径正确
    output_json_path = "./results/frame_search/TStar_LongVideoHaystack_tiny_with_predictions.json"
    test_qa_accuracy(json_path, qwen_interface, output_json_path)

