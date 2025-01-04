import os
import cv2
import json
import numpy as np
from PIL import Image
from decord import VideoReader, cpu

def get_video_fps(video_path):
    """Return frames per second (FPS) of the given video."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video {video_path}")
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_all_videos_fps(folder_path):
    """Retrieve FPS for all videos in a folder."""
    fps_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv')):
                video_path = os.path.join(root, file)
                fps = get_video_fps(video_path)
                if fps is not None:
                    fps_list.append(fps)
                else:
                    print(f"Could not get FPS for {video_path}")
    return fps_list

class VideoTime:
    def __init__(self, video_length, fps, current_position, position_in_frames=True):
        """
        Parameters:
        - video_length (float): Total length of the video in seconds.
        - fps (float): Frames per second of the video.
        - current_position (float): Current position in the video (in seconds or frames).
        - position_in_frames (bool): If True, the current position is given in frames. If False, it is in seconds.
        """
        self.video_length = video_length
        self.fps = fps
        self.current_position = current_position / fps if position_in_frames else current_position

    def rel(self):
        return min(max(self.current_position / self.video_length, 0), 1)

    def frame(self):
        return int(self.current_position * self.fps)

    def second(self):
        return self.current_position


def fallback_video_processing(video_path, positions):
    """Fallback frame extraction using OpenCV if decord fails."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return []
    print(f"Warning: Using fallback video processing for {video_path}")

    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frames.append((pos, frame))
        else:
            print(f"Warning: Failed to retrieve frame {pos} from {video_path}")
    cap.release()
    return frames

def extract_frames(video_path, positions, output_dir, use_decord=True, fig_scale=1, quality=50):
    """Extract frames from a video at the specified positions and save them as JPEG files."""
    os.makedirs(output_dir, exist_ok=True)

    if use_decord:
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            frames = [(pos, vr[pos].asnumpy()) for pos in positions]
        except Exception as e:
            print(f"Error: {e}")
            frames = fallback_video_processing(video_path, positions)
    else:
        frames = fallback_video_processing(video_path, positions)

    for i, frame in frames:
        frame_path = os.path.join(output_dir, f"frame_{i}.jpg")
        frame_resized = cv2.resize(frame, (0, 0), fx=fig_scale, fy=fig_scale)
        Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)).save(frame_path, "JPEG", quality=quality)

def save_frames(frames, output_dir):
    """Save frames to the output directory as JPEG files with specified quality."""
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in frames:
        frame_path = os.path.join(output_dir, f"frame_{i}.jpg")
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(frame_path, "JPEG", quality=50)
