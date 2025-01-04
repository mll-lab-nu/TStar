import os
import json
import subprocess
from collections import defaultdict

# 路径配置
questions_file = "data/questions-all.json"
video_folder = "../Jinhui/ValSubSet"
output_folder = "../Jinhui/clip_videos"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# Step 1: 读取JSON文件
with open(questions_file, 'r') as f:
    data = json.load(f)

# Step 2: 解析每一个entry，提取必要的信息，并去重
video_clips = defaultdict(list)  # 使用字典来保存每个视频的剪辑信息

for entry in data:
    source_video_uid = entry["source_video_uid"]
    source_clip_uid = entry["source_clip_uid"]
    start_sec = round(entry["source_clip_video_start_sec"], 3)  # 将开始时间保留三位小数
    end_sec = round(entry["source_clip_video_end_sec"], 3)  # 将结束时间保留三位小数

    # 将剪辑信息加入对应视频的列表中
    if source_clip_uid not in [clip[0] for clip in video_clips[source_video_uid]]:
        video_clips[source_video_uid].append((source_clip_uid, start_sec, end_sec))

# Step 3: 遍历所有视频并提取片段
processed_clips = set()  # 记录已处理的片段

if os.path.exists("processed_clips.txt"):
    with open("processed_clips.txt", "r") as f:
        processed_clips = set(f.read().splitlines())

for idx, (video_uid, clips) in enumerate(video_clips.items()):
    input_video_path = os.path.join(video_folder, f"{video_uid}.mp4")
    
    # 检查源视频文件是否存在
    if not os.path.exists(input_video_path):
        print(f"Warning: Source video {input_video_path} not found.")
        continue
    for clip_uid, start_sec, end_sec in clips:
        if clip_uid in processed_clips:
            print(f"Skipping already processed clip: {clip_uid}")
            continue

        output_clip_path = os.path.join(output_folder, f"{clip_uid}.mp4")
        
        # 使用FFmpeg提取片段
        ffmpeg_command = [
            "ffmpeg",
            "-i", input_video_path,
            "-ss", str(start_sec),
            "-to", str(end_sec),
            "-c", "copy",  # 直接复制，不重新编码
            output_clip_path
        ]
        
        # 执行FFmpeg命令
        try:
            subprocess.run(ffmpeg_command, check=True)
            print(f"Successfully created clip: {output_clip_path}")
            with open("processed_clips.txt", "a") as f:
                f.write(f"{clip_uid}\n")
        except subprocess.CalledProcessError as e:
            print(f"Error creating clip {output_clip_path}: {e}")