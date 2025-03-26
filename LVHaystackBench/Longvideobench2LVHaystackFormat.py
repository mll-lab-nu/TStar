import json
import os
from typing import List, Dict
from collections import Counter, defaultdict

def LongVideoBench2TStarFormat(dataset_path: str, video_root: str, output_path: str) -> List[Dict]:
    """Load and transform the dataset into the required format for T*.

    Args:
        dataset_path (str): Path to the input dataset JSON file.
        video_root (str): Root directory where video files are stored.
        output_path (str): Path to save the transformed JSON dataset.

    Returns:
        List[Dict]: Transformed dataset formatted for T*.
    """
    with open(dataset_path, 'r', encoding='utf-8') as file:
        lvb_dataset = json.load(file)

    TStar_format_data = []
    num2letter = ['A', 'B', 'C', 'D', 'E']
    
    question_category_counts = Counter()
    video_question_counts = defaultdict(int)
    
    for idx, entry in enumerate(lvb_dataset):
        try:
            video_id = entry.get("video_id")
            video_path = entry.get("video_path")
            question = entry.get("question")
            answer = entry.get("correct_choice", "")
            answer = num2letter[answer]
            question_category = entry.get("question_category", "Unknown")
            duration_group = entry.get("duration_group", "Unknown")
            position = entry.get("position", [])
            options_list = entry.get("candidates", [])

            # Filter out subtitle questions based on question category
            if 'T' in question_category:
                continue
            # Only keep entries with duration group 3600
            if duration_group != 3600:
                continue

            if not video_id or not question or not options_list:
                raise ValueError(f"Missing required fields in entry {idx+1}. Skipping entry.")

            options = "\n".join(f"{num2letter[i]}) {opt}" for i, opt in enumerate(options_list))

            transformed_entry = {
                "video_id": video_id,
                "video_path": os.path.join(video_root, video_path),
                "question": question,
                "options": options,
                "answer": answer,
                "duration_group": duration_group,
                "gt_frame_index": position,
            }
            
            TStar_format_data.append(transformed_entry)
            
            question_category_counts[question_category] += 1
            video_question_counts[video_id] += 1

        except ValueError as e:
            print(f"Skipping entry {idx+1}, reason: {str(e)}")
        except Exception as e:
            print(f"Error processing entry {idx+1}: {str(e)}")

    print("Remaining question category counts:", dict(question_category_counts))
    print("Number of questions per video:", len(video_question_counts))
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(TStar_format_data, f, indent=4)
    print(f"Transformed dataset saved to {output_path}")
    
    return TStar_format_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transform LongVideoBench dataset to T* format.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset JSON file.")
    parser.add_argument("--video_root", type=str, required=True, help="Root directory for video files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the transformed JSON file.")
    args = parser.parse_args()
    
    LongVideoBench2TStarFormat(args.dataset_path, args.video_root, args.output_path)
