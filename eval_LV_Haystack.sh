

python HaystackBench/run_TStar_onDataset.py \
    --dataset_meta longvideobench/LongVideoBench \
    --output_json ./Datasets/HaystackBench/LongVideoHaystack_TStar_results.json \
    --video_dir ./Datasets/HaystackBench/videos \
    --yolo_config_path ./YOLO-World/configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py \
    --yolo_checkpoint_path ./pretrained/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth \
    --search_nframes 8 \
    --image_grid_shape 8 8 \
    --confidence_threshold 0.5 \
    --device cuda:0

python HaystackBench/val_kfs_results.py \
    --result_path ./Datasets/HaystackBench/LongVideoHaystack.json \
    --video_dir ./Datasets/HaystackBench/videos \

python HaystackBench/val_qa_results.py \
    --result_path ./Datasets/HaystackBench/LongVideoHaystack.json \
    --video_dir ./Datasets/HaystackBench/videos \
    --output_json ./Datasets/HaystackBench/LongVideoHaystack_qa_results.json \