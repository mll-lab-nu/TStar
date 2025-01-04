python eval_metrics.py \
    --video_dir ~/Jinhui/clip_videos --metadata_path ~/Jinhui/metadata.jsonl \
    --gt_path xxx [optional, since we have baseline] \
    --pred_path xxx [optional] \
    --frames_key xxx / timestamp_key xxx [optional, least choose one] \
    --eval_temporal --eval_visual