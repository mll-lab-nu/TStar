import json
import cv2
from PIL import Image
from tqdm import tqdm

def extract_frames(video_path, frame_indexes):
    """
    Given a video path and a list of frame indexes, returns a list of corresponding PIL images.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx in frame_indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert frame from BGR (OpenCV format) to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        else:
            print(f"Unable to read frame {idx} from {video_path}")
    cap.release()
    return frames

def test_qa_accuracy(json_path, frame_key, qwen_interface, output_json_path):
    """
    Load JSON data, extract video frames for each sample, construct a prompt,
    run inference with the model, and calculate overall accuracy.
    
    Also, append each sample's predicted answer and correctness flag to the JSON,
    and save the updated results to output_json_path.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    correct = 0
    total = len(data)
    
    for idx, item in enumerate(tqdm(data, desc="Processing samples", unit="sample")):
        video_path = item.get("video_path", "")
        question = item.get("question", "")
        options = item.get("options", "")
        gt_answer = item.get("gt_answer", "")
        # Use keyframe_timestamps as frame indexes
        frame_indexes = item.get(frame_key, [])
        
        # Construct the system prompt with the required number of <image> tags, question and instruction
        prompt = (
            "Answer the following question briefly based on the video.\n" +
            "\n".join(["<image>"] * len(frame_indexes)) +
            f"\nQuestion: {question}\nPlease answer directly with the letter of the option."
        )
        
        # Extract corresponding frames from the video
        frames = extract_frames(video_path, frame_indexes)
        
        # Run inference to get the predicted answer
        pred = qwen_interface.inference(prompt, frames=frames)
        pred_clean = pred.strip().upper()
        gt_clean = gt_answer.strip().upper()
        
        # Append the predicted answer and correctness flag to the current sample
        item["predicted_answer"] = pred_clean
        item["is_correct"] = (pred_clean == gt_clean)
        
        if pred_clean == gt_clean:
            correct += 1
        
        current_accuracy = (correct / (idx + 1)) * 100
        print(f"Video ID: {item.get('video_id', 'N/A')}, GT: {gt_answer}, Pred: {pred_clean}, Current Accuracy: {current_accuracy:.2f}%")
    
    overall_accuracy = (correct / total * 100) if total > 0 else 0
    print("Overall Accuracy: {:.2f}%".format(overall_accuracy))
    
    # Append summary information
    results_summary = {
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy_percent": overall_accuracy
    }
    
    # Save the updated results along with summary information to a new JSON file
    output_data = {
        "results": data,
        "summary": results_summary
    }
    with open(output_json_path, 'w') as f_out:
        json.dump(output_data, f_out, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_json_path}")

if __name__ == "__main__":
    from TStar.interface_grounding_plus import QwenInterface
    # Instantiate the QwenInterface object
    qwen_interface = QwenInterface(
        model_name="./pretrained/Qwen2.5-VL-7B-Instruct",
    )
    json_path = "/data/guoweiyu/LV-Haystack/results/frame_search/yolo-World_TStar_LongVideoHaystack_tiny.json"  # Ensure the path is correct
    output_json_path = "./results/frame_search/TStar_LongVideoHaystack_tiny_with_predictions.json"
    frame_key = "32keyframe_indices"
    frame_key = "keyframe_timestamps"
    test_qa_accuracy(json_path, frame_key, qwen_interface, output_json_path)
