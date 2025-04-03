#!/bin/bash
# =============================================================================
# This script runs the complete TStar pipeline:
# 1. Runs TStar on the dataset to generate search results.
# 2. Validates the TStar results.
# 3. Validates the QA results.
#
# Make sure that the required Python scripts and environment are set up before
# running this script.
# =============================================================================

# -----------------------------------------------------------------------------
# Step 1: Run TStar on the Dataset
# -----------------------------------------------------------------------------
# This command processes the dataset using the TStar pipeline.
# Parameters:
#   --dataset_meta: Specifies the dataset metadata (LVHaystack/LongVideoHaystack).
#   --split: The dataset split (test_tiny, val, test) to use (here, "test_tiny").
#   --video_root: Root directory containing the video files.
#   --output_json_name: The name of the output JSON file for search results.
#   --grounder: The model backend used for QA (here, "gpt-4o").
#   --heuristic: The heuristic used (e.g., "yolo-World").
#   --search_nframes: Number of frames to use for search (here, 8).
python ./LVHaystackBench/run_TStar_onDataset.py \
    --dataset_meta LVHaystack/LongVideoHaystack \
    --split test_tiny \
    --video_root ./Datasets/ego4d_data/ego4d_data/v1/256p \
    --output_json_name TStar_LVHaystack_tiny.json \
    --grounder gpt-4o \
    --heuristic yolo-World \
    --search_nframes 8

# -----------------------------------------------------------------------------
# Step 2: Validate TStar Results
# -----------------------------------------------------------------------------
# This command validates the TStar search results.
# Parameters:
#   --search_result_path: Path to the JSON file containing search results.
#   --frame_index_key: The key in the JSON file that indicates the frame indexes/timestamps.
python ./LVHaystackBench/val_tstar_results.py \ 
    --search_result_path ./results/frame_search/yolo-World_TStar_LVHaystack_tiny.json \
    --frame_index_key keyframe_timestamps


# -----------------------------------------------------------------------------
# Step 3: Validate QA Results
# -----------------------------------------------------------------------------
# This command validates the QA results based on the JSON search results.
# Parameters:
#   --json_file: Path to the JSON file with search results to be used for QA evaluation.
#   --num_frame: Number of frames fed into the QA model.
#   --sampling_type: The frame sampling method (here, "TStar").
#   --duration_type: Whether QA is performed on the full video or a shorter clip (here, "video").
python ./LVHaystackBench/val_qa_results.py \
    --json_file ./results/frame_search/yolo-World_TStar_LVHaystack_tiny.json \
    --backend gpt-4o \
    --num_frame 8 \
    --sampling_type TStar \
    --duration_type video

# =============================================================================
# End of pipeline
# =============================================================================
