from datasets import load_from_disk
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple
from kfs.evaluation.metrics import calculate_ssim, calculate_prf

dataset = load_from_disk("kfs-bench-textonly")
# preprocess data: decode frames and put back into the dataset
def preprocess_dataset(dataset, textonly=False):
    dataset = dataset.map(
        lambda sample: {
            "indexes_seconds": np.array([i / sample['video_metadata']['frame_rate'] for i in sample['frame_indexes']]),
            **sample
        }
    )
    if not textonly:
        dataset = dataset.map(
            lambda sample: {
                "decoded_frames": [
                    torch.from_numpy(np.copy(np.frombuffer(frame, dtype=np.uint8).reshape(3, *sample["video_metadata"]["frame_dimensions_resized"])))
                    .float() / 255.0  # Normalize to [0, 1]
                    for frame in sample["frames"]
                ],
                **sample
            },
            remove_columns=["frames"]
        )
    return dataset

dataset = preprocess_dataset(dataset['test'], textonly=True)
# decoded_frames = dataset["test"][0]["decoded_frames"]
# indexes_seconds = dataset["test"][0]["indexes_seconds"]

# 1. calculate prf
gt_indexes_seconds = dataset['indexes_seconds']

# ours_data_path = "data/kfsbench/distributionsearch_Ego_KFS.json" # ours, 8 frames
# ours_lvbench_path = "data/lvbench/lvbench_XL_4methods_probs.json"
ours_lvbench_path = "data/lvbench/KFS_lvbench_XL_allinone.json"
# A[0].keys()
# dict_keys(['video_id', 'question', 'question_wo_referring_query', 'candidates', 'correct_choice', 'position', 'topic_category', 'question_category', 'level', 'id', 'video_path', 'subtitle_path', 'duration_group', 'starting_timestamp_for_subtitles', 'duration', 'view_count', 'outputs', 'baseline_answer', 'search_predict_answer', 'best_predict_answer', 'priority_objects', 'yolo_score', 'distributions_history', 'keyframe_distributions', 'iteration', 'KSF_sec_32', 'KSF_sec_8', 'TStar_RL_8frames_sec', 'TStar_YOLO_8frames_sec', 'TStar_Attention_8frames_sec', 'TStar_YOLO_32frames_sec', 'Uniform_32frames_sec', 'Uniform_8frames_sec'])
breakpoint()
prf_scores = calculate_prf([indexes_seconds], [indexes_seconds])



ssim_scores = calculate_ssim([decoded_frames], [decoded_frames])
prf_scores = calculate_prf([indexes_seconds], [indexes_seconds])

