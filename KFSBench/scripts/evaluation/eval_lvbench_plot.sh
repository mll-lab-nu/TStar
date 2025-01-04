# firstly, put all videos as a symlink under static/lvbench_videos
# evaluation/lvbench/format_questions has already been run. Ignore it.
python evaluation/scripts/json_oracle_to_frames.py \
    --json_path data/lvbench/datasets/questions-all-formatted-lvbench.json \
    --output_dir data/lvbench/frame_results/lvbench_oracle


# python scripts/evaluation/get_plot.py \
#     --pred_json data/lvbench/json_results/zoom_in_search.json \
#     --group 600


python scripts/evaluation/get_plot_bi.py \
    --linear_json data/lvbench/json_results/linear_search.json \
    --zoom_in_json data/lvbench/json_results/zoom_in_search.json \
    --fps_dict_path data/lvbench/datasets/fps.json \
    --group 600
