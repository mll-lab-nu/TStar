import pandas as pd
import os
import string
import json


def LVHaystack2Tstar_Json(input_csv: str, video_root: str) -> list:
    """
    读取 filtered_questions.csv 文件并将每一行转换为包含 'video_path'，'question' 和 'options' 的字典格式，
    然后返回转换后的字典列表。

    Args:
        input_csv (str): CSV 文件路径，例如 'filtered_questions.csv'。
        video_root (str): 视频文件的根路径，用于构造完整的视频路径。

    Returns:
        list: 转换后的字典列表，每个字典包含 'video_path'，'question' 和 'options'。
    """
    import ast
    # 读取 CSV 文件
    from datasets import load_dataset
    dataset = load_dataset("LVHaystack/LongVideoHaystack")
    print(dataset)
    category_1_df = dataset



    # 构建 vclipid 到 videoid 的映射字典
    # 使用示例
    json_file_path = './Datasets/Haystack-Bench/ego4d_nlq_val.json'
    # 读取 JSON 文件
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    vclipid_to_videoid = {}
    # 遍历 JSON 数据，提取 vclipid 和 videoid
    for entry in data:
        vclipid = entry.get('source_clip_uid')
        videoid = entry.get('source_video_uid')

        if vclipid and videoid:
            vclipid_to_videoid[vclipid] = videoid


    # 定义结果列表
    transformed_data = []

    # 遍历 CSV 文件中的每一行
    for idx, row in category_1_df.iterrows():
        entry = row.to_dict()  # 将当前行转换为字典

        try:
            # 提取必要字段
            video_id = vclipid_to_videoid[entry.get('vclip_id', '')]  # 假设视频ID为 'video_id'
            question = entry.get('question', '')  # 假设问题文本为 'question'
            candidates_str = entry.get('choices', '')  # 假设选项是通过 '|' 分隔的字符串


            answer = entry.get('answer', 'None') 
            gt_frame_index = entry.get('frame_indexes', []) 


            # 验证字段是否存在
            if not video_id:
                raise ValueError(f"缺少 'video_id' 字段 (第 {idx+1} 行)")
            if not question:
                raise ValueError(f"缺少 'question' 字段 (第 {idx+1} 行)")
            if not candidates_str:
                raise ValueError(f"缺少 'options' 字段或选项为空 (第 {idx+1} 行)")

            # 将字符串解析为字典
            candidates_dict = ast.literal_eval(candidates_str)
            # 生成选项字符串，带有字母前缀（A, B, C, D...）
            options = ""
            for i, key in enumerate(candidates_dict):

                options += f"{key}) {candidates_dict[key]}\n"

            options = options.rstrip('\n')  # 去掉末尾的换行符

            # 构建转换后的字典并加入结果列表
            transformed_entry = {
                "video_id": video_id,
                "video_path": os.path.join(video_root, video_id + ".mp4"),  # 构建完整的视频路径
                "question": question,
                "options": options,
                "answer": answer,
                'gt_frame_index': gt_frame_index,
            }
            transformed_data.append(transformed_entry)

        except ValueError as e:
            print(f"跳过条目 {idx+1}，原因：{str(e)}")

    return transformed_data


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

    # Data meta processing arguments
    parser.add_argument('--json_path', type=str, default="./Datasets/charades_annotation/test.caption_coco_format.json", help='Path to the input JSON file for batch processing.')
    parser.add_argument('--output_json', type=str, default='./batch_output.json', help='Path to save the batch processing results.')
    parser.add_argument('--video_root', type=str, default='./Datasets/CharadesVideo_v1', help='Root directory where the input video files are stored.')
    
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
    TStar_framework = TStarFramework(
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
        "video_path": searching_entry['video_path'],
        "grounding_objects": TStar_framework.results.get('Searching_Objects', []),
        "frame_timestamps": TStar_framework.results.get('timestamps', []),
        "frame_distribution": video_searcher.P_history[-1]
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
        dataset = Charades2Tstar_Json(json_file_path=args.json_path, video_root=args.video_root)
        print(len(dataset), "%"*30)
        for idx, searching_json in enumerate(dataset):      

            # if searching_json['video_id'] != "38737402-19bd-4689-9e74-3af391b15feb":
            #     continue

            print(f"Processing {idx+1}/{len(dataset)}: {searching_json['video_id']}")
            try:
                result = process_single_search(args, searching_entry=searching_json, grounder=grounder, yolo_scorer=yolo_interface)
                
                print(f"Completed: {searching_json['video_id']}\n")
            except Exception as e:
                print(f"Error processing {searching_json['video_id']}: {e}")
                result = {
                    "video_id": searching_json.get('video_id', ''),
                    "grounding_objects": [],
                    "frame_timestamps": [],
                    "answer": "",
                    "error": str(e)
                }
            searching_json.update(result)
            results.append(searching_json)
        
        # Save batch results to output_json
        output_json = args.json_path+".json"
        with open(output_json, 'w', encoding='utf-8') as f_out:
            json.dump(results, f_out, indent=4, ensure_ascii=False)
        
        print(f"Batch processing completed. Results saved to {output_json}")

    

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
                "video_path": "path/to/video1.mp4",
                "question": "What is the color of my couch?",
                "options": "A) Red\nB) Black\nC) Green\nD) White\n"

                "keyframe_distribution": []
                "keyframe_sec: []
            },
            // 更多条目...
        ]
        
        ]
    """
    
    main()

    
