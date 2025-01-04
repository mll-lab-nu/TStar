## get_metadata.py

python scripts/data_processing/get_metadata.py \
  --videos_path "../Jinhui/clip_videos" \
  --output_file "../Jinhui/metadata.jsonl"

python scripts/data_processing/get_metadata.py \
  --videos_path "../LongVideoBench" \
  --output_file "../Jinhui/metadata.jsonl"

## sec_to_frame.py

files=(
    # TSearching_max_resource.json
    # searching_distribution_shift.json
    # 3600_TSearching_8frames.json
    # 3600_RL_16frames.json
    # 600_TSearching_8frames.json
    # 60_TSearching.json
    # 600_RLSearching_8.json
    15_Tsearching_8.json
    60_TSearching_32.json
)

for file in "${files[@]}" ; do
  python scripts/data_processing/sec_to_frame.py \
    --input_file data/lvbench/json_results/"$file" \
    --output_file data/lvbench/json_results/new/"$file" \
    --fps_file "data/lvbench/datasets/fps.json"
done

# DATASET EXAMPLE
# {
#     'video_name': 'cb3bf9d7-7f6b-4567-9446-45c6b493d721.mp4',
#     'questions': 'what did I put in the black dustbin?',
#     'CLIP-result-relative': [420.0, 435.0],
#     'positions': [],
#     'answers': '吸尘器里的垃圾',
#     'question_id': 0,
#     'frames': [...],  # 图片数据列表，每张为3D numpy数组 (C, H, W)
#     'frame_indexes': [12845, 12875, 12917],
#     'video_metadata': {
#         'file_name': 'cb3bf9d7-7f6b-4567-9446-45c6b493d721.mp4',
#         'frame_count': 14274,
#         'frame_rate': 30.0,
#         'duration': 475.8,
#         'resolution': '454x256',
#         'frame_dimensions': [454, 256],
#         'codec': 'h264',
#         'bitrate': 518542
#     }
# }
