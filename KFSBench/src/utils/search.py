import os
import cv2
import json
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from .video_processing import fallback_video_processing

def extract_oracle_frames(video_path, positions, dry_run=False):
    """Extract frames from the video using the specified positions with decord."""
    if dry_run:
        return [(pos, np.zeros((1, 1, 3), dtype=np.uint8)) for pos in positions]
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        return [(pos, vr[pos].asnumpy()) for pos in positions if pos < len(vr)]
    except Exception as e:
        print(f"Decord failed for {video_path}, using fallback. Error: {e}")
        return fallback_video_processing(video_path, positions)

def extract_linear_frames(video_path, num_frames, dry_run=False):
    if dry_run:
        len_video = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)
        frame_indices = np.linspace(0, len_video - 2, num_frames, dtype=int)
        return [(idx, np.zeros((1, 1, 3), dtype=np.uint8)) for idx in frame_indices]
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        frame_indices = np.linspace(0, len(vr) - 2, num_frames, dtype=int)
        return [(idx, vr[idx].asnumpy()) for idx in frame_indices]
    except:
        return fallback_video_processing(video_path, num_frames)
