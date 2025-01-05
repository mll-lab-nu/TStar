
import os
import sys
import cv2
import torch
import copy
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from decord import VideoReader, cpu
from scipy.interpolate import UnivariateSpline

# Import custom TStar interfaces

from TStar.TStarFramework import TStarFramework, initialize_yolo #if framework, we might not put interfance here? --> better to put for more readable
from TStar.interface_llm import TStarUniversalGrounder

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="TStarSearcher: Video Frame Search and QA Tool")
    parser.add_argument('--video_path', type=str, default="./Datasets/ego4d/ego4d_data/v1/256p/38737402-19bd-4689-9e74-3af391b15feb.mp4", help='Path to the input video file.')
    parser.add_argument('--question', type=str, default="What is the color of my couch?", help='Question for video content QA.')
    parser.add_argument('--options', type=str, default="A) Red\nB) Black\nC) Green\nD) White\n", help='Multiple-choice options for the question, e.g., "A) Option1\nB) Option2\nC) Option3\nD) Option4"')
    parser.add_argument('--config_path', type=str, default="./YOLO-World/configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py", help='Path to the YOLO configuration file.')
    parser.add_argument('--checkpoint_path', type=str, default="./pretrained/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth", help='Path to the YOLO model checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device for model inference (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--search_nframes', type=int, default=8, help='Number of top frames to return.')
    parser.add_argument('--grid_rows', type=int, default=4, help='Number of rows in the image grid.')
    parser.add_argument('--grid_cols', type=int, default=4, help='Number of columns in the image grid.')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='YOLO detection confidence threshold.')
    parser.add_argument('--search_budget', type=float, default=0.5, help='Maximum ratio of frames to process during search.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save outputs.')
    parser.add_argument('--prefix', type=str, default='stitched_image', help='Prefix for output filenames.')
    return parser.parse_args()


def main():
    """
    Main function to execute TStarSearcher.
    """
    args = parse_arguments()

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

    # Initialize VideoSearcher
    searcher = TStarFramework(
        grounder=grounder,
        yolo_scorer=yolo_interface,
        video_path=args.video_path,
        question=args.question,
        options=args.options,
        search_nframes=args.search_nframes,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        search_budget=args.search_budget,
        prefix=args.prefix,
        device=args.device
    )

    # Run the search and QA process
    searcher.run()

    # Output the results
    print("Final Results:")
    print(f"Grounding Objects: {searcher.results['Searching_Objects']}")
    print(f"Frame Timestamps: {searcher.results['timestamps']}")
    print(f"Answer: {searcher.results['answer']}")




if __name__ == "__main__":
    """
    TStarSearcher: Comprehensive Video Frame Search Tool

    This script allows searching for specific objects within a video using YOLO object detection and GPT-4 for question-answering. It leverages the TStar framework's universal Grounder, YOLO interface, and video searcher to identify relevant frames and answer questions based on the detected objects.

    Usage:
        python tstar_searcher.py --video_path path/to/video.mp4 --question "Your question here" --options "A) Option1\nB) Option2\nC) Option3\nD) Option4"
    """
    main()