import json
import argparse
import numpy as np
from tqdm import tqdm

# Import or define calculate_prf function below

def load_json(filepath):
    """Load JSON data from a given file path."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_prf_for_one(data, threshold):
    # Extract frame indexes and positions from the input data
    try:
        total_covered, total_gt_frames, total_pred_frames = 0, 0, 0
        frame_indexes = data.get("frame_indexes", [])
        positions = data.get("position", [])
        gt_frames = np.array(positions)
        pred_frames = np.array(frame_indexes)
        
        distances = np.min(np.abs(gt_frames[:, np.newaxis] - pred_frames), axis=1)
        covered_frames = np.sum(distances <= threshold)
        total_covered += covered_frames
        total_gt_frames += len(gt_frames)
        total_pred_frames += len(pred_frames)
        precision = total_covered / total_pred_frames if total_pred_frames > 0 else 0
        recall = total_covered / total_gt_frames if total_gt_frames > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        # print(gt_frames, pred_frames, recall)

        # 另一种算法：precision是每个pred最近的gt距离取平均， recall是每个gt最近的pred距离取平均， f1是precision和recall的调和平均
        # precision = np.mean(np.min(np.abs(gt_frames[:, np.newaxis] - pred_frames), axis=1))
        # recall = np.mean(np.min(np.abs(pred_frames[:, np.newaxis] - gt_frames), axis=1))
        # f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1
    except:
        breakpoint()
        return 0, 0, 0
        
def calculate_prf(predictions, fps_dict, threshold=5, group=None):
    """Calculate Precision, Recall, and F1 Score based on frame coverage."""
    total_precision, total_recall, total_f1 = 0, 0, 0
    for data in predictions:
        if group is not None and data['duration_group'] != group:
            continue
        if data['video_path'] not in fps_dict:
            continue
        fps = eval(fps_dict.get(data['video_path']))
        precision, recall, f1 = get_prf_for_one(data, threshold * fps)
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    return total_precision / len(predictions), total_recall / len(predictions), total_f1 / len(predictions)

def directly_eval_based_on_json(pred_path, fps_dict_path, group, threshold=5, baseline=False, num_frames=8):
    """
    Evaluate predictions based on JSON data for precision, recall, and F1 score.
    """
    all_data = json.load(open("data/lvbench/datasets/lvb_val.json"))
    all_data_dict = {i['question']: i for i in all_data}
    fps_dict = load_json(fps_dict_path)
    if pred_path.endswith(".jsonl"):
        predictions = [json.loads(i) for i in open(pred_path).readlines()]
    else:
        predictions = load_json(pred_path)
    for i in range(len(predictions)):
        data = predictions[i]
        data = {**data, **all_data_dict[data['question']]}
        if 'frame_indexes' not in data:
            data['frame_indexes'] = sum(data['frames'], [])
        predictions[i] = data
    # predictions = load_json(pred_path)
    if baseline:
        for data in predictions:
            all_frames = int(data['duration'] * eval(fps_dict[data['video_path']])) - 1
            data['frame_indexes'] = list(range(0, all_frames, all_frames // num_frames))
    # Processing and evaluating predictions here (adapt if necessary)
    precision, recall, f1 = calculate_prf(predictions, fps_dict, threshold=threshold, group=group)
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct evaluation based on JSON results.")
    parser.add_argument("--fps_dict_path", required=True, help="Path to FPS dictionary JSON file")
    parser.add_argument("--group", type=int, help="Grouping parameter")
    parser.add_argument("--threshold", type=int, default=5, help="Threshold seconds for frame coverage")
    parser.add_argument("--pred_path", required=True, help="Path to prediction JSON file")
    parser.add_argument("--baseline", action="store_true", help="Use baseline evaluation")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to evaluate")

    args = parser.parse_args()
    directly_eval_based_on_json(args.pred_path, args.fps_dict_path, args.group, args.threshold, args.baseline, args.num_frames)
