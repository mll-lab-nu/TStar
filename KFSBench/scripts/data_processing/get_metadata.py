import cv2
import ffmpeg
import os
import json
import argparse
from tqdm import tqdm

def get_video_metadata(video_path):
    # 使用 OpenCV 获取基础元数据
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / frame_rate if frame_rate > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = f"{width}x{height}"
    frame_dimensions = (width, height)
    
    # 使用 ffmpeg 获取编码格式和比特率
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
        codec = video_stream.get('codec_name', 'N/A')
        bitrate = int(video_stream.get('bit_rate', 0))
        key_frame_count = sum(1 for frame in video_stream.get('tags', {}).get('keyframe', []))  # 统计关键帧数量
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        codec = 'N/A'
        bitrate = 0
        key_frame_count = 0
    
    cap.release()
    
    return {
        'file_name': os.path.basename(video_path),
        'frame_count': frame_count,
        'frame_rate': frame_rate,
        'duration': duration,
        'resolution': resolution,
        'frame_dimensions': frame_dimensions,
        'codec': codec,
        'bitrate': bitrate,
        'key_frame_count': key_frame_count
    }

def summarize_video_metadata(videos_path, output_file):
    # 获取目录下所有视频文件的路径
    video_files = [os.path.join(videos_path, f) for f in os.listdir(videos_path) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    
    # 提取并保存所有视频的元数据
    with open(output_file, 'w') as f:
        for video_path in tqdm(video_files, desc="Processing videos"):
            metadata = get_video_metadata(video_path)
            f.write(json.dumps(metadata) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract metadata from a directory of videos")
    parser.add_argument("--videos_path", type=str, required=True, help="Path to the directory containing videos")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON Lines file")

    args = parser.parse_args()
    
    summarize_video_metadata(args.videos_path, args.output_file)
