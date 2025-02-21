

python HaystackBench/run_TStar_onDataset.py \
    --input_json ./Datasets/Haystack-Bench/annotations.json \
    --output_json ./outputs/add_TStar_results.json \
    --video_dir ./Data/Haystack-Bench/videos \
    --yolo_config_path ./YOLOWorld/configs/pretrain/yolo_world_v2_xl_vlpan_bn.py \
    --yolo_checkpoint_path ./pretrained/yolo_checkpoint.pth \
    --search_nframes 8 \
    --image_grid_shape 8 8 \
    --confidence_threshold 0.5 \
    --device cuda:0

python HaystackBench/val_kfs_results.py \
    --result_path ./outputs/add_TStar_results.json \
    --video_dir ./Data/Haystack-Bench/videos \


python HaystackBench/val_qa_results.py \
    --result_path ./outputs/add_TStar_results.json \
    --video_dir ./Data/Haystack-Bench/videos \
    --output_json ./outputs/add_qa_results.json \