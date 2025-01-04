import os
import json
import torch
from torchvision.io import read_image
import numpy as np
from PIL import Image
from datasets import Dataset, DatasetDict, Features, Value, Array3D, Sequence

# 定义根目录
root_dir = "data/annotations"
questions_file = "data/questions-annotated.json"
video_metadata_file = "../Jinhui/metadata.jsonl"
video_metadata = {k['file_name']: k for k in [json.loads(i) for i in open(video_metadata_file).readlines()]}


# 加载 questions-annotated.json 文件，创建 video_id 到 video_name 的映射
with open(questions_file, "r") as f:
    questions_data = json.load(f)
video_id_to_name = {idx: entry["video_name"] for idx, entry in enumerate(questions_data)}



def resize_frame(frame, target_width, target_height):
    """Resize a frame to a target height and width."""
    frame = frame.unsqueeze(0)  # shape: (1, C, W, H)
    frame_resized = torch.nn.functional.interpolate(
        frame.float(), size=(target_width, target_height), mode="bilinear", align_corners=False
    )
    frame_resized = frame_resized.squeeze(0).byte().numpy()
    return frame_resized

# 初始化列表以存储 metadata 和 frame 数据
metadata_records = []
metadata_records_textonly = []

# 遍历文件夹并处理文件
for first_level in os.listdir(root_dir):
    first_level_path = os.path.join(root_dir, first_level)
    if not os.path.isdir(first_level_path):
        continue

    for second_level in os.listdir(first_level_path):
        second_level_path = os.path.join(first_level_path, second_level)
        if not os.path.isdir(second_level_path):
            continue
        video_id = int(first_level)
        video_name = video_id_to_name.get(video_id, "unknown")

        # 构建 metadata
        metadata = {k: v[int(second_level)] for k, v in questions_data[video_id].items() if type(v) == list}
        metadata.update({
            "question_id": int(second_level),
            "video_name": video_name, 
            "frames": [],
            "frame_indexes": [],
            "video_metadata": video_metadata[video_name]
        })
        frame_files = [i for i in os.listdir(os.path.join(second_level_path, 'snapshots')) if i.endswith("jpg")]
        for frame_file in frame_files:
            filepath = os.path.join(second_level_path, 'snapshots', frame_file)
            frame = read_image(filepath) # resize to 3*340*256
            frame_resized = resize_frame(frame, 340, 256)
            metadata["frames"].append(frame_resized.tobytes())
            metadata["frame_indexes"].append(int(frame_file.split(".")[0].strip("frame_")))
            metadata["video_metadata"]["frame_dimensions_resized"] = [340, 256]
            metadata["video_metadata"]["resolution_resized"] = "340x256"
        metadata_records.append(metadata)
        metadata_records_textonly.append({
            k:v for k, v in metadata.items() if k not in ["frames"]
        })

features_textonly = Features({
    "video_name": Value("string"),
    "questions": Value("string"),
    "CLIP-result-relative": Sequence(Value("float32")),
    "positions": Sequence(Value("int32")),
    "answers": Value("string"),
    "question_id": Value("int32"),
    "frame_indexes": Sequence(Value("int32")),
    "video_metadata": {
        "file_name": Value("string"),
        "frame_count": Value("int32"),
        "frame_rate": Value("float32"),
        "duration": Value("float32"),
        "resolution": Value("string"),
        "frame_dimensions": Sequence(Value("int32")),
        "codec": Value("string"),
        "bitrate": Value("int32"),
        "frame_dimensions_resized": Sequence(Value("int32")),
        "resolution_resized": Value("string"),
    }
})


# 构建数据集
dataset_textonly = Dataset.from_list(metadata_records_textonly, features=features_textonly)
dataset_dict_textonly = DatasetDict({"test": dataset_textonly})
dataset_dict_textonly.save_to_disk("kfs-bench-textonly")

features = Features({
    "video_name": Value("string"),
    "questions": Value("string"),
    "CLIP-result-relative": Sequence(Value("float32")),
    "positions": Sequence(Value("int32")),
    "answers": Value("string"),
    "question_id": Value("int32"),
    "frames": Sequence(Value("binary")),  
    "frame_indexes": Sequence(Value("int32")),
    "video_metadata": {
        "file_name": Value("string"),
        "frame_count": Value("int32"),
        "frame_rate": Value("float32"),
        "duration": Value("float32"),
        "resolution": Value("string"),
        "frame_dimensions": Sequence(Value("int32")),
        "codec": Value("string"),
        "bitrate": Value("int32"),
        "frame_dimensions_resized": Sequence(Value("int32")),
        "resolution_resized": Value("string"),
    }
})

dataset = Dataset.from_list(metadata_records, features=features)
dataset_dict = DatasetDict({"test": dataset})
dataset_dict.save_to_disk("kfs-bench")
