import string
import json
from typing import List, Dict


def transform2Tstar_Json(entry: dict) -> dict:
    """
    Transforms a complex entry dictionary into a simplified format containing
    only 'video_path', 'question', and 'options'.

    Args:
        entry (dict): The original dictionary with multiple fields.

    Returns:
        input dict: A simplified dictionary with 'video_path', 'question', and 'options'.
        
    """
    # Extract necessary fields
    #@TBD Jinhui
    video_path = entry.get('video_id')
    video_path = "./38737402-19bd-4689-9e74-3af391b15feb.mp4"
    question = entry.get('question') or entry.get('question_wo_referring_query')
    candidates = entry.get('candidates', [])

    # Validate extracted fields
    if not video_path:
        raise ValueError("The entry is missing the 'video_path' field.")
    if not question:
        raise ValueError("The entry is missing the 'question' field.")
    if not candidates:
        raise ValueError("The entry is missing the 'candidates' field or it is empty.")

    # Generate options string with letter prefixes
    options = ""
    for idx, candidate in enumerate(candidates):
        if idx < 26:
            option_label = string.ascii_uppercase[idx]
        else:
            # For options beyond 'Z', use double letters (e.g., AA, AB, ...)
            first = (idx // 26) - 1
            second = idx % 26
            option_label = string.ascii_uppercase[first] + string.ascii_uppercase[second]
        options += f"{option_label}) {candidate}\n"

    # Remove the trailing newline character
    options = options.rstrip('\n')

    return {
        "video_path": video_path,
        "question": question,
        "options": options
   }


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


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="TStarSearcher: Video Frame Search and QA Tool")

    # Batch processing arguments
    parser.add_argument('--json_path', type=str, default="llava/VL-Haystack/Datasets/Haystack-Bench/KFS_lvbench_XL_allinone.json", help='Path to the input JSON file for batch processing.')
    parser.add_argument('--output_json', type=str, default='./batch_output.json', help='Path to save the batch processing results.')

    # Common arguments
    parser.add_argument('--config_path', type=str, default="./YOLOWorld/configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py", help='Path to the YOLO configuration file.')
    parser.add_argument('--checkpoint_path', type=str, default="./pretrained/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth", help='Path to the YOLO model checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device for model inference (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--search_nframes', type=int, default=8, help='Number of top frames to return.')
    parser.add_argument('--grid_rows', type=int, default=4, help='Number of rows in the image grid.')
    parser.add_argument('--grid_cols', type=int, default=4, help='Number of columns in the image grid.')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='YOLO detection confidence threshold.')
    parser.add_argument('--search_budget', type=float, default=1.0, help='Maximum ratio of frames to process during search.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save outputs.')
    parser.add_argument('--prefix', type=str, default='stitched_image', help='Prefix for output filenames.')
    return parser.parse_args()


def process_single_search(args, searching_entry,
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
    searcher = TStarFramework(
        grounder=grounder,
        yolo_scorer=yolo_scorer,
        video_path=searching_entry['video_path'],
        question=searching_entry['question'],
        options=searching_entry['options'],
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
    target_objects, cue_objects = searcher.get_grounded_objects()

    # Initialize TStarSearcher
    video_searcher = TStarSearcher(
        video_path=searcher.video_path,
        target_objects=target_objects,
        cue_objects=cue_objects,
        search_nframes=searcher.search_nframes,
        image_grid_shape=(args.grid_rows, args.grid_cols),
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        search_budget=args.search_budget,
        prefix=args.prefix,
        yolo_scorer=yolo_scorer
    )

    # Perform search
    all_frames, time_stamps = searcher.perform_search(video_searcher)

    # # Save retrieved frames
    # searcher.save_frames(all_frames, time_stamps)

    # Perform question-answering on retrieved frames
    answer = searcher.perform_qa(all_frames)
    
    # Collect the results
    result = {
        "video_path": searching_entry['video_path'],
        "grounding_objects": searcher.results.get('Searching_Objects', []),
        "frame_timestamps": searcher.results.get('timestamps', []),
        "answer": searcher.results.get('answer', "")
    }

    return result


def main():
    """
    Main function to execute TStarSearcher.
    """
    args = parse_arguments()

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

    if args.json_path:
        # Batch processing
        with open(args.json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)[:2] #@Debug
        
        for idx, sample in enumerate(dataset):
            print(f"Processing {idx+1}/{len(dataset)}: {sample['video_id']}")
            searching_json = transform2Tstar_Json(sample)
            try:
                result = process_single_search(args, searching_entry=searching_json, grounder=grounder, yolo_scorer=yolo_interface)
                
                print(f"Completed: {searching_json['video_path']}\n")
            except Exception as e:
                print(f"Error processing {searching_json['video_path']}: {e}")
                result = {
                    "video_path": searching_json.get('video_path', ''),
                    "grounding_objects": [],
                    "frame_timestamps": [],
                    "answer": "",
                    "error": str(e)
                }
            sample.update(result)
            results.append(sample)
        
        # Save batch results to output_json
        with open(args.json_path+".addTStarResults", 'w', encoding='utf-8') as f_out:
            json.dump(results, f_out, indent=4, ensure_ascii=False)
        
        print(f"Batch processing completed. Results saved to {args.json_path}")

    

if __name__ == "__main__":
    """
    TStarSearcher: Comprehensive Video Frame Search Tool

    This script allows searching for specific objects within a video using YOLO object detection and GPT-4 for question-answering. It leverages the TStar framework's universal Grounder, YOLO interface, and video searcher to identify relevant frames and answer questions based on the detected objects.

    Usage:
        Batch Processing:
            python tstar_searcher.py --json_path path/to/input.json --output_json path/to/output.json

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

            },
            // 更多条目...
        ]
    """
    
    main()

    
