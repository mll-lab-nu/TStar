

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



import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def process_video_frames(video_path, output_path, num_frames=10, x_angle=290, y_angle=20, z_angle=10):
    """
    Uniformly sample frames from a video, apply 3D rotation to each frame, and stitch them together.

    Args:
        video_path (str): Path to the input video.
        num_frames (int): Number of frames to sample uniformly from the video.
        x_angle (float): Rotation around the X-axis in degrees.
        y_angle (float): Rotation around the Y-axis in degrees.
        z_angle (float): Rotation around the Z-axis in degrees.

    Returns:
        PIL.Image: The final stitched image.
    """
    def get_rotation_matrix(x_angle, y_angle, z_angle):
        # Convert angles to radians
        x_rad = np.deg2rad(x_angle)
        y_rad = np.deg2rad(y_angle)
        z_rad = np.deg2rad(z_angle)

        # Define rotation matrices
        rx = np.array([[1, 0, 0],
                       [0, np.cos(x_rad), -np.sin(x_rad)],
                       [0, np.sin(x_rad), np.cos(x_rad)]])
        ry = np.array([[np.cos(y_rad), 0, np.sin(y_rad)],
                       [0, 1, 0],
                       [-np.sin(y_rad), 0, np.cos(y_rad)]])
        rz = np.array([[np.cos(z_rad), -np.sin(z_rad), 0],
                       [np.sin(z_rad), np.cos(z_rad), 0],
                       [0, 0, 1]])

        # Combined rotation matrix
        return np.dot(np.dot(rz, ry), rx)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize frame for easier processing
        frame = cv2.resize(frame, (160, 120))  # Resize to 160x120
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames extracted from the video.")

    # Process each frame with 3D rotation
    processed_frames = []
    rotation_matrix = get_rotation_matrix(x_angle, y_angle, z_angle)

    for frame in frames:
        h, w, _ = frame.shape
        corners = np.array([[0, 0, 0],
                            [w, 0, 0],
                            [0, h, 0],
                            [w, h, 0]])
        rotated_corners = np.dot(corners, rotation_matrix.T)
        projected_corners = rotated_corners[:, :2]
        min_x, min_y = projected_corners.min(axis=0)
        projected_corners -= [min_x, min_y]
        max_x, max_y = projected_corners.max(axis=0)
        scale = min(w / max_x, h / max_y)
        projected_corners *= scale

        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst_pts = projected_corners.astype(np.float32)
        transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        rotated_frame = cv2.warpPerspective(frame, transform_matrix, (int(max_x * scale), int(max_y * scale)))
        processed_frames.append(rotated_frame)

    # Stitch all rotated frames together
    stitched_image = np.hstack(processed_frames)

    # Save the stitched image
    cv2.imwrite(output_path, cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving

    print(f"Stitched image saved to: {output_path}")

    # Convert to PIL image for display
    final_image = Image.fromarray(stitched_image)

    # Display the result
    plt.figure(figsize=(15, 5))
    plt.imshow(final_image)
    plt.axis('off')
    plt.show()


    return final_image


def render_frames_in_3d(video_path, output_path, num_frames=10, x_angle=290, y_angle=20, z_angle=10):
    """
    Render uniformly sampled video frames as 3D boards with adjustable angles and save the result.

    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the rendered 3D plot.
        num_frames (int): Number of frames to sample from the video.
        x_angle (float): Rotation around the X-axis in degrees.
        y_angle (float): Rotation around the Y-axis in degrees.
        z_angle (float): Rotation around the Z-axis in degrees.
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize frame for easier processing
        frame = cv2.resize(frame, (160, 120))  # Resize to 160x120
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames extracted from the video.")

    # Create a 3D plot
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Helper function to draw a frame as a board
    def draw_frame(ax, img, x_offset, z_offset):
        h, w, _ = img.shape
        x = np.array([0, w, w, 0]) + x_offset
        y = np.array([0, 0, h, h]) - h / 2
        z = np.array([0, 0, 0, 0]) + z_offset
        vertices = [list(zip(x, y, z))]
        
        poly = Poly3DCollection(vertices, alpha=0.8, facecolors=plt.cm.viridis(np.random.rand()))
        ax.add_collection3d(poly)

    # Adjust angles
    ax.view_init(elev=y_angle, azim=z_angle)

    # Render each frame
    for i, frame in enumerate(frames):
        draw_frame(ax, frame, x_offset=i * 180, z_offset=i * 5)

    # Set axis limits and labels
    ax.set_xlim(0, num_frames * 200)
    ax.set_ylim(-100, 100)
    ax.set_zlim(0, num_frames * 10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save with high resolution
    print(f"3D plot saved to: {output_path}")

    # Display the result
    plt.show()


if __name__ == "__main__":
    # Example usage
    video_path = "./Datasets/ego4d/ego4d_data/v1/256p/38737402-19bd-4689-9e74-3af391b15feb.mp4" # Replace with your video file path
    output_path = "output_3d_strip.png"
    # process_video_frames(video_path, output_path)
    render_frames_in_3d(video_path, output_path, num_frames=10, x_angle=10, y_angle=20, z_angle=10)

