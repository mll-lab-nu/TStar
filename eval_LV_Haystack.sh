
./hfd.sh Qwen/Qwen2.5-VL-7B-Instruct

export https_proxy="http://10.120.16.212:20000"; export http_proxy="http://10.120.16.212:20000"; export all_proxy="socks5://10.120.16.212:20000"

export OPENAI_API_KEY=open_ai_key
export PYTHONPATH=/data/guoweiyu/LV-Haystack:/data/guoweiyu/LV-Haystack/YOLO-World:${PYTHONPATH}
# gpt-4o as grounder, owl as Searcher
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
python ./run_TStar_onDataset.py \
    --dataset_meta LVHaystack/LongVideoHaystack \
    --split val \
    --video_root ./Datasets/ego4d_data/ego4d_data/v1/256p \
    --output_json_name TStar_LVHaystack_tiny.json \
    --grounder gpt-4o \
    --heuristic yolo-World \
    --search_nframes 8 \
    --grid_rows 4 \
    --grid_cols 4 


python HaystackBench/val_kfs_results.py \
    --result_path ./Datasets/HaystackBench/LongVideoHaystack.json \
    --video_dir ./Datasets/HaystackBench/videos \

conda activate haystack
export https_proxy="http://10.120.16.212:20000"; export http_proxy="http://10.120.16.212:20000"; export all_proxy="socks5://10.120.16.212:20000"

export OPENAI_API_KEY=open_ai_key

./results/last_version/yolo-World_TStar_LongVideoHaystack_testqa_8frames_gpt-4o_uniform.json

results/

python val_qa_results_jinhui.py \
    --json_file LongVideoHaystack_test.json \
    --num_frame 8 \
    --sampling_type TStar \
    --duration_type video


python val_qa_results_jinhui.py \
    --json_file LongVideoHaystack_test.json \
    --num_frame 8 \
    --sampling_type uniform \
    --duration_type video



python val_tstar_results.py \
    --search_result_path results/frame_search/2025-03-22-07-33-52objnew_LVHaystack_gpt4_raw_vid1.json \
    --pred_index_key 32keyframe_indices 



