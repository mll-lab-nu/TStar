# # firstly, put all videos as a symlink under static/lvbench_videos
# # evaluation/lvbench/format_questions has already been run. Ignore it.
echo videoagent
python scripts/evaluation/eval_json.py \
    --pred_path=data/lvbench/json_results/new/videoagent.jsonl \
    --fps_dict_path data/lvbench/datasets/fps.json \
    --threshold 5 \
    --group 3600


names=(
    "3600_TSearching_8frames" 
    "3600_RL_32frames"
)
frames=(
    8, 32
)

groups=(
    3600 3600
)
i=-1
for name in "${names[@]}"
do
    i=$((i+1))
    frame=${frames[$i]}
    group=${groups[$i]}
    echo "Processing frame ${frame} group ${group}"
    echo ours
    python scripts/evaluation/eval_json.py \
        --pred_path=data/kfsbench/json_results/new/${name}.json \
        --fps_dict_path ~/Jinhui/fps.json \
        --threshold 5
    
    echo random
    python scripts/evaluation/eval_json.py \
        --pred_path=data/kfsbench/json_results/new/${name}.json \
        --fps_dict_path  ~/Jinhui/fps.json \
        --num_frames ${frame} \
        --baseline \
        --threshold 5
done

# # firstly, put all videos as a symlink under static/lvbench_videos
# # evaluation/lvbench/format_questions has already been run. Ignore it.
# python evaluation/scripts/json_oracle_to_frames.py \
#     --json_path data/lvbench/datasets/questions-all-formatted-lvbench.json \
#     --output_dir data/lvbench/frame_results/lvbench_oracle

# # this would create model prediction file frames under data/lvbench/frame_results/XXX given files such as data/lvbench/json_results/linear_search.json
# python evaluation/scripts/json_result_to_frames.py \
#     --json_result_path data/lvbench/json_results/linear_search.json \
#     --output_dir data/lvbench/frame_results/linear_search 

# python scripts/evaluation/get_table.py \
#     --gt_folder_path=log/results/uniform_baseline_8 \
#     --pred_folder_path=log/results/random_baseline
