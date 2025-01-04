# # firstly, put all videos as a symlink under static/lvbench_videos
# # evaluation/lvbench/format_questions has already been run. Ignore it.
# python scripts/evaluation/json_to_frames.py \
#     --json_path data/lvbench/datasets/questions-all-formatted-lvbench.json \
#     --video_dir static/lvbench_videos \
#     --output_dir data/lvbench/frame_results/lvbench_oracle \
#     --mode oracle

# python scripts/evaluation/json_to_frames.py \
#     --json_path data/questions-annotated.json \
#     --video_dir static/clip_videos \
#     --output_dir data/kfsbench_oracle.json \
#     --mode oracle



# #### processing lvbench results

# names=("linear_search" "zoom_in_search" "tzoom_search")
# # this would create model prediction file frames under data/lvbench/frame_results/XXX given files such as data/lvbench/json_results/linear_search.json
# for name in "${names[@]}"
# do
#     python scripts/evaluation/json_to_frames.py \
#         --json_path data/lvbench/datasets/questions-all-formatted-lvbench.json \
#         --json_result_path data/lvbench/json_results/${name}.json \
#         --output_dir data/lvbench/frame_results/${name} \
#         --fps_dict_path data/lvbench/datasets/fps.json \
#         --mode result
# done

# # if the json result path uses seconds, then we need to convert it to frame index

# for framenum in 8 32
# do
#     python scripts/evaluation/search_frames.py \
#         --json_path data/lvbench/datasets/questions-all-formatted-lvbench.json \
#         --strategy uniform \
#         --num_frames ${framenum} \
#         --output_dir data/lvbench/frame_results/uniform_baseline_${framenum}frames
# done



# get the table for each model

names=("linear_search" "zoom_in_search" "tzoom_search" "beam_search" "uniform_baseline_4frames"  "uniform_baseline_8frames" "uniform_baseline_32frames")
for name in "${names[@]}"
do
    python scripts/evaluation/get_table.py \
        --gt_folder_path=data/lvbench/frame_results/lvbench_oracle \
        --pred_folder_path=data/lvbench/frame_results/${name} \
        --json_path data/lvbench/datasets/questions-all-formatted-lvbench.json \
        --fps_dict_path data/lvbench/datasets/fps.json \
        --group 3600
done


### processing kfsbench results

