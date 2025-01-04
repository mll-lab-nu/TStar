

from typing import List
import math
from typing import List, Dict
from PIL import Image
import base64
import io
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: OpenCV is not installed, video frame extraction will not work.")



def encode_image_to_base64(image) -> str:
    """
    Convert an image (PIL.Image or numpy.ndarray) to a Base64 encoded string.
    """
    try:
        # If the input is a numpy array, convert it to a PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Ensure it's a PIL Image before proceeding
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL.Image or numpy.ndarray")

        # Encode the image to Base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        raise ValueError(f"Error encoding image: {str(e)}")

def load_video_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """
    从视频中读取 num_frames 帧并返回 PIL.Image 列表。
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
    from PIL import Image
    import os

    fps = 1  # 设置帧率为 1
    duration = int(1000 / fps)  # GIF 每帧显示时间，单位为毫秒

    # 将每一帧图像转换为 PIL 图像
    pil_images = [Image.fromarray(img.astype('uint8')) for img in images]
    
    # 保存为 GIF
    pil_images[0].save(
        output_gif_path, 
        save_all=True, 
        append_images=pil_images[1:], 
        duration=duration, 
        loop=0  # 设置循环播放（0 为无限循环）
    )
    print(f"Saved GIF: {output_gif_path}")

