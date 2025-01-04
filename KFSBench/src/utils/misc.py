import os
import cv2
import json
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from collections import defaultdict

def get_nested_dict(folder_path):
    """Generate a nested dictionary for all image files in the folder structure."""
    nested_dict = {}
    for root, _, files in os.walk(folder_path):
        relative_path = os.path.relpath(root, folder_path)
        path_parts = relative_path.split(os.sep)
        current_level = nested_dict

        for part in path_parts:
            part = int(part) if part.isdigit() else part
            if part == '.':
                continue
            current_level = current_level.setdefault(part, {})
        
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            current_level['files'] = image_files
    return nested_dict

def format_result_data(json_result_path, fps_dict_path):
    """Format JSON result data for frame extraction."""
    with open(json_result_path, 'r', encoding='utf-8') as f:
        json_result = json.load(f)
    fps_dict = load_json(fps_dict_path)

    formatted_result = defaultdict(lambda: defaultdict(list))
    for item in json_result:
        video_path = item['video_path']
        question = item['question']
        fps = eval(fps_dict.get(video_path))
        frame_index = item['frame_indexes']
        formatted_result[video_path][question] = frame_index

    return formatted_result

def load_image(path):
    """Load image from path."""
    return cv2.imread(path)

def save_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f)

def load_json(file_path):
    """Load data from a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: File not found at {file_path}, returning empty json.")
        return {}

def load_questions(json_path):
    """Load questions from a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

