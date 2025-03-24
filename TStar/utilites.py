import math
import base64
import io
import os
from typing import List
import numpy as np
from PIL import Image
import cv2

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v3 as imageio


def encode_image_to_base64(image) -> str:
    """
    Convert an image (PIL.Image or numpy.ndarray) to a Base64 encoded string.
    
    Args:
        image: A PIL.Image or numpy.ndarray representing the image.
    
    Returns:
        A Base64 encoded string of the image.
    
    Raises:
        ValueError: If the input is neither a PIL.Image nor a numpy.ndarray.
    """
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL.Image or numpy.ndarray")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        raise ValueError(f"Error encoding image: {str(e)}")


def load_video_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """
    Load a specified number of frames from a video as PIL.Image objects.
    
    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to extract.
    
    Returns:
        A list of PIL.Image objects representing the extracted frames.
    
    Raises:
        ImportError: If OpenCV is not installed.
        ValueError: If the video cannot be opened or has zero frames.
    """
    if cv2 is None:
        raise ImportError("OpenCV is not installed, cannot load video frames.")
    
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError("Video has zero frames or could not retrieve frame count.")
    
    num_frames = min(num_frames, total_frames)
    step = total_frames / num_frames

    for i in range(num_frames):
        frame_index = int(math.floor(i * step))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames


def save_as_gif(images, output_gif_path):
    """
    Save a list of images as an animated GIF.
    
    Args:
        images: A list of image arrays.
        output_gif_path: Path to save the resulting GIF.
    """
    fps = 1  # Frames per second
    duration = int(1000 / fps)  # Duration per frame in milliseconds
    pil_images = [Image.fromarray(img.astype('uint8')) for img in images]
    pil_images[0].save(
        output_gif_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0
    )
    print(f"Saved GIF: {output_gif_path}")




def extract_frames(video_path, output_dir, fps=1):
    """
    Extract frames from a video at a specified frame rate.
    
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save extracted frames.
        fps (int): Frames per second to extract.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps // fps)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved: {frame_filename}")
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"Total frames saved: {saved_count}")


def extract_frames_from_gif(input_gif_path, output_dir):
    """
    Extract frames from a GIF file and save them as individual PNG files.
    
    Args:
        input_gif_path (str): Path to the input GIF.
        output_dir (str): Directory where frames will be saved.
    """
    base_name = os.path.basename(input_gif_path).split('.')[0]
    output_subdir = os.path.join(output_dir, base_name)
    os.makedirs(output_subdir, exist_ok=True)

    with imageio.imopen(input_gif_path, "r", plugin="pillow") as gif:
        i = 0
        for frame in gif.iter():
            frame_image = Image.fromarray(frame)
            frame_filename = os.path.join(output_subdir, f"frame_{i + 1}.png")
            frame_image.save(frame_filename)
            print(f"Saved frame {i + 1} to {frame_filename}")
            i += 1

    print(f"All frames have been extracted to {output_subdir}")


if __name__ == "__main__":
    # Example usage for extracting frames from a GIF
    input_gif_path = "output/38737402-19bd-4689-9e74-3af391b15feb/Who did I talk to in  the living room_score_distribution.gif"
    output_dir = "output/38737402-19bd-4689-9e74-3af391b15feb"
    extract_frames_from_gif(input_gif_path, output_dir)
