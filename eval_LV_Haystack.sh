
export https_proxy="http://10.120.16.212:20000"; export http_proxy="http://10.120.16.212:20000"; export all_proxy="socks5://10.120.16.212:20000"

export OPENAI_API_KEY=open_ai_key

# gpt-4o as grounder, owl as Searcher

python ./run_TStar_onDataset.py \
    --dataset_meta LVHaystack/LongVideoHaystack \
    --split test \
    --video_root ./Datasets/ego4d_data/ego4d_data/v1/256p \
    --output_json ./Datasets/HaystackBench/LongVideoHaystack_tiny_check.json \
    --grounder gpt-4o \
    --heuristic owl-vit \
    --search_nframes 8 \
    --grid_rows 4 \
    --grid_cols 4 \
    --search_budget 0.1 \
    --confidence_threshold 0.6 


python HaystackBench/val_kfs_results.py \
    --result_path ./Datasets/HaystackBench/LongVideoHaystack.json \
    --video_dir ./Datasets/HaystackBench/videos \

python HaystackBench/val_qa_results.py \
    --result_path ./Datasets/HaystackBench/LongVideoHaystack.json \
    --video_dir ./Datasets/HaystackBench/videos \
    --output_json ./Datasets/HaystackBench/LongVideoHaystack_qa_results.json \