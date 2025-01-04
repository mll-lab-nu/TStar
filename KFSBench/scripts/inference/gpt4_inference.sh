files=(
#     TSearching_max_resource.json
#     60_TSearching.json
#     600_TSearching_8frames.json
#     600_RLSearching_8.json
#     searching_distribution_shift.json
#     3600_TSearching_8frames.json
#     3600_RL_16frames.json
    # 15_Tsearching_8.json
    # 60_TSearching_32.json
)

for file in "${files[@]}" ; do
  name=$(basename "$file" .json)
  python scripts/inference/gpt4_inference.py \
    --json_file "$file" \
    --json_path "data/lvbench/json_results/new" \
    --videos_path "../LongVideoBench" \
    --output_file "log/gpt4_result_lvbench_${name}.jsonl" \
    --model "gpt-4o-2024-05-13"
done


files=(
    # 60_TSearching.json
    # 600_TSearching_8frames.json
    # 3600_TSearching_8frames.json
    # TSearching_max_resource.json
    # 600_RLSearching_8.json
    # searching_distribution_shift.json
    # 3600_RL_16frames.json
    # 15_Tsearching_8.json
    60_TSearching_32.json
)

# for file in "${files[@]}" ; do
#   name=$(basename "$file" .json)
#   python scripts/inference/gpt4_inference.py \
#     --json_file "$file" \
#     --json_path "data/lvbench/json_results/new" \
#     --videos_path "../LongVideoBench" \
#     --use_oracle \
#     --output_file "log/gpt4_result_lvbench_${name}_oracle.jsonl" \
#     --model "gpt-4o-2024-05-13"
# done

for file in "${files[@]}" ; do
  name=$(basename "$file" .json)
  python scripts/inference/gpt4_inference.py \
    --json_file "$file" \
    --json_path "data/lvbench/json_results/new" \
    --videos_path "../LongVideoBench" \
    --use_uniform \
    --output_file "log/gpt4_result_lvbench_${name}_uniform.jsonl" \
    --model "gpt-4o-2024-05-13"
done
