import pandas as pd
import os
import string
import json
import os
import sys
import cv2
import torch
import copy
import logging
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from decord import VideoReader, cpu
from scipy.interpolate import UnivariateSpline

# Import custom TStar interfaces
from TStar.interface_llm import TStarUniversalGrounder
from TStar.interface_yolo import YoloInterface
from TStar.interface_searcher import TStarSearcher
from TStar.TStarFramework import TStarFramework, initialize_yolo  # better to keep interfaces separate for readability



import os
import ast
from datasets import load_dataset
from typing import List

def LVHaystack2TStar_json(dataset_meta: str = "LVHaystack/LongVideoHaystack", 
                          split="test",
                          video_root: str = "Datasets/Ego4D_videos") -> List[dict]:
    """Load and transform the dataset into the required format for T*.

    The output JSON structure is like:
    [
        {
            "video_path": "path/to/video1.mp4",
            "question": "What is the color of my couch?",
            "options": "A) Red\nB) Black\nC) Green\nD) White\n",
            // More user-defined keys...
        },
        // More entries...
    ]
    """
    # Load the dataset from the given source
    dataset = load_dataset(dataset_meta)
    
    # Extract the 'test' split from the dataset
    LVHaystact_testset = dataset[split]

    # List to hold the transformed data
    TStar_format_data = []

    # Iterate over each row in the dataset
    for idx, entry in enumerate(LVHaystact_testset[:2]):
        try:
            # Extract necessary fields from the entry
            video_id = entry.get("video_id")
            question = entry.get("question")
            gt_answer = entry.get("answer", "")

            options_dict = entry.get("options", "")
            gt_frame_index = entry.get("frame_indexes", []) #gt frame index for quetion

            # Validate required fields
            if not video_id or not question:
                raise ValueError(f"Missing required question or video id in entry {idx+1}. Skipping entry.")

            # Parse the options string into a dictionary
            if options_dict != "":
                # Format the options with letter prefixes (A, B, C, D...)
                options = ""
                for i, (key, value) in enumerate(options_dict.items()):
                    options += f"{key}) {value}\n"
                options = options.rstrip('\n')  # Remove the trailing newline

            # Construct the transformed dictionary for the entry
            transformed_entry = {
                "video_id": video_id,
                "video_path": os.path.join(video_root, f"{video_id}.mp4"),  # Build the full video path
                "question": question,
                "options": options,
                "gt_answer": gt_answer,
                "gt_frame_index": gt_frame_index,
            }

            # Add the transformed entry to the result list
            TStar_format_data.append(transformed_entry)

        except ValueError as e:
            print(f"Skipping entry {idx+1}, reason: {str(e)}")
        except Exception as e:
            print(f"Error processing entry {idx+1}: {str(e)}")

    return TStar_format_data



def process_TStar_onVideo(args, data_item,
        yolo_scorer: YoloInterface,
        grounder: TStarUniversalGrounder,) -> dict:
    """
    Process a single video search and QA.

    Args:
        args (argparse.Namespace): Parsed arguments.
        entry (dict): Dictionary containing 'video_path', 'question', and 'options'.
        yolo_scorer (YoloV5Interface): YOLO interface instance.
        grounder (TStarUniversalGrounder): Universal Grounder instance.

    Returns:
        dict: Results containing 'video_path', 'grounding_objects', 'frame_timestamps', 'answer'.
    """
 

    # Initialize VideoSearcher
    TStar_framework = TStarFramework(
        grounder=grounder,
        yolo_scorer=yolo_scorer,
        video_path=data_item['video_path'],
        question=data_item['question'],
        options=data_item['options'],
        search_nframes=args.search_nframes,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        search_budget=args.search_budget,
        prefix=args.prefix,
        device=args.device
    )

    # Use Grounder to get target and cue objects
    target_objects, cue_objects = TStar_framework.get_grounded_objects()

    # Initialize Searching Targets to TStar Seacher
    video_searcher = TStar_framework.set_searching_targets(target_objects, cue_objects)


    # Perform search
    all_frames, time_stamps = TStar_framework.perform_search(video_searcher)



    # Output the results
    print("Final Results:")
    print(f"Grounding Objects: {TStar_framework.results['Searching_Objects']}")
    print(f"Frame Timestamps: {TStar_framework.results['timestamps']}")

    # Collect the results
    result = {
        "video_path": data_item['video_path'],
        "grounding_objects": TStar_framework.results.get('Searching_Objects', []),
        "keyframe_timestamps": TStar_framework.results.get('timestamps', []),
        "frame_distribution": video_searcher.P_history[-1]
    }

    return result


def main():
    """
    Main function to execute TStarSearcher.
    """

    parser = argparse.ArgumentParser(description="TStarSearcher: Video Frame Search and QA Tool")

    # Data meta processing arguments
    parser.add_argument('--dataset_meta', type=str, default="LVHaystack/LongVideoHaystack", help='Path to the input JSON file for batch processing.')
    parser.add_argument('--video_root', type=str, default='./Datasets/ego4d_data/ego4d_data/v1/256p', help='Root directory where the input video files are stored.')
    parser.add_argument('--output_json', type=str, default='./Datasets/LongVideoHaystack.json', help='Path to save the batch processing results.')
    
    # Common arguments
    parser.add_argument('--config_path', type=str, default="./YOLO-World/configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py", help='Path to the YOLO configuration file.')
    parser.add_argument('--checkpoint_path', type=str, default="./pretrained/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth", help='Path to the YOLO model checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device for model inference (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--search_nframes', type=int, default=8, help='Number of top frames to return.')
    parser.add_argument('--grid_rows', type=int, default=4, help='Number of rows in the image grid.')
    parser.add_argument('--grid_cols', type=int, default=4, help='Number of columns in the image grid.')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='YOLO detection confidence threshold.')
    parser.add_argument('--search_budget', type=float, default=1.0, help='Maximum ratio of frames to process during search.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save outputs.')
    parser.add_argument('--prefix', type=str, default='stitched_image', help='Prefix for output filenames.')
    
    args=  parser.parse_args()


    if args.dataset_meta:
        # Batch processing
        dataset = LVHaystack2TStar_json(dataset_meta=args.dataset_meta, video_root=args.video_root)
        

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

       # Initialize Grounder
    grounder = TStarUniversalGrounder(
        backend="gpt4",
        gpt4_model_name="gpt-4o"
    )

    # Initialize YOLO interface
    yolo_interface = initialize_yolo(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device
    )

    results = []


    for idx, data_item in enumerate(dataset):      


        print(f"Processing {idx+1}/{len(dataset)}: {data_item['video_id']}")
        try:
            result = process_TStar_onVideo(args, data_item=data_item, grounder=grounder, yolo_scorer=yolo_interface)
            
            print(f"Completed: {data_item['video_id']}\n")
        except Exception as e:
            print(f"Error processing {data_item['video_id']}: {e}")
            result = {
                "video_id": data_item.get('video_id', ''),
                "grounding_objects": [],
                "frame_timestamps": [],
                "answer": "",
                "error": str(e)
            }
        data_item.update(result)
        results.append(data_item)
    
    # Save batch results to output_json
    output_json = args.output_json
    with open(output_json, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, indent=4, ensure_ascii=False)
    
    print(f"Batch processing completed. Results saved to {output_json}")



if __name__ == "__main__":
    """
    TStarSearcher: Comprehensive Video Frame Search Tool

    This script input videos and questions (with / without options) and frame buget K
    
    then return the the K keyframe indexs to response the given question. 

    Usage:
        Batch Processing:
            python tstar_searcher.py --dataset_meta path/to/dateset meta --output_json path/to/out_json with the attaching rearching frames.

        input json
        [
            {
                "video_path": "path/to/video1.mp4",
                "question": "What is the color of my couch?",
                "options": "A) Red\nB) Black\nC) Green\nD) White\n"
            },
            // 更多条目...
        ]
        output: 
        [
            {
                "video_path": "path/to/video1.mp4",
                "question": "What is the color of my couch?",
                "options": "A) Red\nB) Black\nC) Green\nD) White\n"

                "keyframe_distribution": []
                "keyframe_timestamps: []
            },
            // 更多条目...
        ]
        
        ]
    """
    
    main()

    
