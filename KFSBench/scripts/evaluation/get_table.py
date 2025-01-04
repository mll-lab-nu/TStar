import argparse
import os
from kfs.utils import get_nested_dict, load_json
from kfs.evaluation import calculate_prf, calculate_annd, calculate_ssim
import numpy as np
import json

def format_value(value):
    """Round the value to 2 decimal places for cleaner output."""
    return round(value, 3)

def calculate_scores(precision_values, recall_values, scale=1):
    """
    Calculate Precision, Recall, Average, and F1 Score.
    """
    precision_mean = np.mean(precision_values) / scale
    recall_mean = np.mean(recall_values) / scale
    avg_score = (precision_mean + recall_mean) / 2
    f1_score = 2 * precision_mean * recall_mean / (precision_mean + recall_mean) if (precision_mean + recall_mean) != 0 else 0

    return {
        "Precision": format_value(precision_mean),
        "Recall": format_value(recall_mean),
        "Average": format_value(avg_score),
        "F1 Score": format_value(f1_score)
    }

def flatten_dict(nested_dict):
    flattened_dict = {}
    for key1, sub_dict in nested_dict.items():
        if isinstance(sub_dict, dict):
            for key2, value in sub_dict.items():
                flattened_dict[(key1, key2)] = value
        else:
            flattened_dict[key1] = sub_dict
    return flattened_dict


def main(args):
    gt_folder_path, pred_folder_path = args.gt_folder_path, args.pred_folder_path
    json_path, group = args.json_path, args.group
    question_json = load_json(json_path)
    fps_dict = json.load(open(args.fps_dict_path)) 
    
    """Main function to compute comparison metrics between ground truth and predicted data."""
    nested_dict_1 = get_nested_dict(gt_folder_path)
    nested_dict_2 = get_nested_dict(pred_folder_path)
    dict_1 = flatten_dict(nested_dict_1)
    dict_2 = flatten_dict(nested_dict_2)
    dict_1 = {k: v for k, v in dict_1.items() if v}
    dict_2 = {k: v for k, v in dict_2.items() if v}
    # 交集
    dict_1 = {k: v for k, v in dict_1.items() if k in dict_2}
    dict_2 = {k: v for k, v in dict_2.items() if k in dict_1}
    breakpoint()

    if group == "60":
        dict_1 = {k: v for k, v in dict_1.items() if question_json[k[0]]['durations'][0] <= 60}
        dict_2 = {k: v for k, v in dict_2.items() if question_json[k[0]]['durations'][0] <= 60}
    elif group == "600":
        dict_1 = {k: v for k, v in dict_1.items() if question_json[k[0]]['durations'][0] > 60 and question_json[k[0]]['durations'][0] <= 600}
        dict_2 = {k: v for k, v in dict_2.items() if question_json[k[0]]['durations'][0] > 60 and question_json[k[0]]['durations'][0] <= 600}
    elif group == "3600":
        dict_1 = {k: v for k, v in dict_1.items() if question_json[k[0]]['durations'][0] > 600 and question_json[k[0]]['durations'][0] <= 3600}
        dict_2 = {k: v for k, v in dict_2.items() if question_json[k[0]]['durations'][0] > 600 and question_json[k[0]]['durations'][0] <= 3600}

    breakpoint()
    video_list = list(set([i[0] for i in dict_1]))
    video_len_in_d2 = [question_json[i]['durations'][0] for i in video_list]
    print("extracted video length:", round(sum(video_len_in_d2) / len(video_len_in_d2), 2))
    dict_1_num = {k: [int(i.split("_")[1].split(".")[0]) / float(fps_dict[question_json[k[0]]['video_name']]) for i in v['files']] for k, v in dict_1.items()}
    dict_2_num = {k: [int(i.split("_")[1].split(".")[0]) / float(fps_dict[question_json[k[0]]['video_name']]) for i in v['files']] for k, v in dict_2.items()}
    prf_score = calculate_prf(dict_1_num, dict_2_num, threshold=5)
    prf_formatted = tuple(map(format_value, prf_score))
    print("PRECISION, RECALL, F1:", prf_score)
    # print("Precision, Recall, F1 Score (frame hit threshold=60):", prf_formatted)
    # for k, v in nested_dict_1_numberlized.items():
    #     print(v, nested_dict_2_numberlized[k])

    # Calculate Average Nearest Neighbor Distance (ANND)
    # annd_scores = calculate_annd(dict_1_num, dict_2_num)
    # annd_precision, annd_recall = zip(*annd_scores)
    # annd_results = calculate_scores(annd_precision, annd_recall)
    # print("Average Nearest Neighbor Distance (ANND, frames):")
    # for key, value in annd_results.items():
    #     print(f"  {key}:", value)
    # print("Available item count:", len(annd_scores))


    # Calculate Structural Similarity Index (SSIM)
    # ssim_scores = calculate_ssim(nested_dict_1, nested_dict_2, gt_folder_path, pred_folder_path)
    # ssim_precision, ssim_recall = zip(*ssim_scores)
    # ssim_results = calculate_scores(ssim_precision, ssim_recall)
    # print("Structural Similarity Index (SSIM):")
    # for key, value in ssim_results.items():
    #     print(f"  {key}:", value)
    # print("Available item count:", len(ssim_scores))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute comparison metrics between ground truth and predicted data.")
    parser.add_argument("--gt_folder_path", type=str, required=True, help="Path to the ground truth folder.")
    parser.add_argument("--pred_folder_path", type=str, required=True, help="Path to the predicted folder.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file with video frame positions.")
    parser.add_argument("--fps_dict_path", type=str, help="Dictionary mapping video names to FPS values.")
    parser.add_argument("--group", type=str, required=False, help="Group of the JSON file with video frame positions.")
    args = parser.parse_args()

    if not os.path.exists(args.gt_folder_path) or not os.path.exists(args.pred_folder_path):
        print("Error: One or both of the specified folder paths do not exist.")
        exit()

    main(args)