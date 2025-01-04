import os
import cv2
import json

# 初始化总时长
total_duration = 0

# 获取当前目录路径
current_directory = "../Jinhui/ValSubSet"

# 获取视频文件列表
video_files = [filename for filename in os.listdir(current_directory) if filename.endswith(".mp4")]

# 遍历当前目录中的所有文件
for filename in video_files:
    try:
        # 使用 cv2 获取视频时长
        video_path = os.path.join(current_directory, filename)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法读取文件 {filename}: 文件可能已损坏或者不支持的视频编码格式。")
            continue
        
        # 获取视频的帧数和帧率
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        # 计算视频时长
        duration = frame_count / fps
        total_duration += duration
        
        # 释放资源
        cap.release()
    except Exception as e:
        print(f"无法读取文件 {filename}: {e}")

# 计算 questions-all.json 中的时长和
json_file_path = "data/questions-all.json"
unique_durations = {}
try:
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        questions_data = json.load(json_file)
        for item in questions_data:
            source_clip_uid = item.get('source_clip_uid')
            if source_clip_uid is not None:
                video_start_sec = item.get('source_clip_video_start_sec', 0)
                video_end_sec = item.get('source_clip_video_end_sec', 0)
                duration = video_end_sec - video_start_sec
                if source_clip_uid not in unique_durations:
                    unique_durations[source_clip_uid] = duration
                else:
                    pass
except Exception as e:
    print(f"无法读取文件 {json_file_path}: {e}")

# 计算去重后的总时长
questions_total_duration = sum(unique_durations.values())

# 将总时长转换为小时、分钟和秒
hours = int(total_duration // 3600)
minutes = int((total_duration % 3600) // 60)
seconds = int(total_duration % 60)

questions_hours = int(questions_total_duration // 3600)
questions_minutes = int((questions_total_duration % 3600) // 60)
questions_seconds = int(questions_total_duration % 60)

# 打印结果
print(f"所有mp4文件的总时长为: {hours}小时 {minutes}分钟 {seconds}秒")
print(f"去重后的 JSON 文件中所有片段的总时长为: {questions_hours}小时 {questions_minutes}分钟 {questions_seconds}秒")
