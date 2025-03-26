import os
import json
import argparse
from typing import List
from datasets import load_dataset

# Import custom TStar interfaces
from TStar.interface_grounding import TStarUniversalGrounder
from TStar.interface_heuristic import HeuristicInterface
from TStar.TStarFramework import TStarFramework, initialize_heuristic  # better to keep interfaces separate for readability



def LVHaystack2TStarFormat(dataset_meta: str = "LVHaystack/LongVideoHaystack", 
                          split="tiny",
                          video_root: str = "Datasets/Ego4D_videos") -> List[dict]:
    """Load and transform the dataset into the required format for T*.

    The output JSON structure is like:
    [
        {
            "video_path": "path/to/video1.mp4",
            "question": "What is the color of my couch?",
            "options": "A) Red\nB) Black\nC) Green\nD) White\n",
            // More user-defined keys...
        },
        // More entries...
    ]
    """
    # # Load the dataset from the given source
    dataset = load_dataset(dataset_meta) #, download_mode="force_redownload"
    
    # Extract the 'test' split from the dataset
    LVHaystact_testset = dataset[split]

    # # List to hold the transformed data
    TStar_format_data = []

    
    # Iterate over each row in the dataset
    for idx, item in enumerate(LVHaystact_testset):
        try:
            # Extract necessary fields from the entry
            video_id = item.get("video_id")
            question = item.get("question")
            gt_answer = item.get("answer")

            options_dict = item.get("options", "")
            gt_frame_index = item.get("frame_indexes", []) #gt frame index for quetion

            # Validate required fields
            if not video_id or not question:
                raise ValueError(f"Missing required question or video id in entry {idx+1}. Skipping entry.")

            # Parse the options string into a dictionary
            if options_dict != "":
                # Format the options with letter prefixes (A, B, C, D...)
                options = ""
                for i, (key, value) in enumerate(options_dict.items()):
                    options += f"{key}) {value}\n"
                options = options.rstrip('\n')  # Remove the trailing newline

            video_metadata = item["video_metadata"]
            vclip_interval_in_video = video_metadata["vclip_interval_in_video"]
        
            # Construct the transformed dictionary for the entry
            transformed_item = {
                "video_id": video_id,
                "video_path": os.path.join(video_root, f"{video_id}.mp4"),  # Build the full video path
                "question": question,
                "options": options,
                "gt_answer": gt_answer,
                "gt_frame_index": gt_frame_index,
                "vclip_interval_in_video": vclip_interval_in_video
            }

            # Add the transformed entry to the result list
            TStar_format_data.append(transformed_item)

        except ValueError as e:
            print(f"Skipping entry {idx+1}, reason: {str(e)}")
        except Exception as e:
            print(f"Error processing item {idx+1}: {str(e)}")

    return TStar_format_data[:200] #[0:10] # [:2] for debug



def get_TStar_search_results(args, data_item,
        grounder: TStarUniversalGrounder,
        heurisiticFuncion: HeuristicInterface,
        ) -> dict:
    """
    Process a single video search and QA.

    Args:
        args (argparse.Namespace): Parsed arguments.
        entry (dict): Dictionary containing 'video_path', 'question', and 'options'.
        yolo_scorer (YoloV5Interface): YOLO interface instance.
        grounder (TStarUniversalGrounder): Universal Grounder instance.

    Returns:
        dict: Results containing 'video_path', 'grounding objects', 'frame_timestamps', 'answer'.
    """
 

    # Initialize VideoSearcher
    TStar_searcher = TStarFramework(
        video_path=data_item['video_path'],
        question=data_item['question'],
        options=data_item['options'],
        grounder=grounder,
        heuristic=heurisiticFuncion,
        search_nframes=args.search_nframes,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        search_budget=args.search_budget,
    )

    # Use Grounder to get target and cue objects
    target_objects, cue_objects = TStar_searcher.get_grounded_objects()
    # Initialize Searching Targets to TStar Seacher
    video_searcher = TStar_searcher.initialize_videoSearcher(target_objects, cue_objects)
    # Perform search
    all_frames, time_stamps = TStar_searcher.perform_search(video_searcher, visualization=True)
    time_stamps.sort()

    # Output the results
    print("#"*20+" Original Inputs "  + "#"*20)
    print(f"Input Quetion: {data_item['question']}")
    print(f"Input Options: {data_item['options']}")
    print("#"*20+" T* Searching Results "  + "#"*20)
    print(f"Grounding Objects: target_objects: {target_objects}, cue_objects: {cue_objects}")
    print(f"Frame Timestamps: {time_stamps}")

    # Collect the results
    result = {
        "video_path": data_item['video_path'],
        "grounding_objects": {"target_objects": target_objects, "cue_objects": cue_objects},
        "keyframe_timestamps": time_stamps,
        "keyframe_distribution": video_searcher.P_history[-1]
    }

    return result


def main():
    """
    Main function to execute TStarSearcher.
    """

    parser = argparse.ArgumentParser(description="TStarSearcher: Video Frame Search and QA Tool")

    # Data meta processing arguments
    parser.add_argument('--dataset_meta', type=str, default="LVHaystack/LongVideoHaystack", help='Path to the input JSON file for batch processing.')
    parser.add_argument('--split', type=str, default="test_tiny", help='Path to the input JSON file for batch processing.')    
    parser.add_argument('--video_root', type=str, default='./Datasets/ego4d_data/ego4d_data/v1/256p', help='Root directory where the input video files are stored.')
    parser.add_argument('--output_json_name', type=str, default='TStar_LongVideoHaystack_tiny.json', help='Path to save the batch processing results.')
    
    # search tools
    parser.add_argument('--grounder', type=str, default='gpt-4o', help='Directory to save outputs.')
    parser.add_argument('--heuristic', type=str, default='owl-vit', help='Directory to save outputs.')
    ## if yolo as detector
    parser.add_argument('--config_path', type=str, default="./YOLO-World/configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py", help='Path to the YOLO configuration file.')
    parser.add_argument('--checkpoint_path', type=str, default="./pretrained/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth", help='Path to the YOLO model checkpoint.')
    
    # Common arguments
    parser.add_argument('--device', type=str, default="auto", help='Device for model inference (e.g., "cuda" or "cpu").')
    parser.add_argument('--search_nframes', type=int, default=8, help='Number of top frames to return.')
    parser.add_argument('--grid_rows', type=int, default=4, help='Number of rows in the image grid.')
    parser.add_argument('--grid_cols', type=int, default=4, help='Number of columns in the image grid.')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='YOLO detection confidence threshold.')
    parser.add_argument('--search_budget', type=float, default=1.0, help='Maximum ratio of frames to process during search.')
    parser.add_argument('--output_dir', type=str, default='./results/frame_search', help='Directory to save outputs.')

    args = parser.parse_args()


    if args.dataset_meta:
        dataset = LVHaystack2TStarFormat(dataset_meta=args.dataset_meta, split=args.split, video_root=args.video_root)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Search tools
    grounder = TStarUniversalGrounder(model_name=args.grounder)
    TStarHeuristic = initialize_heuristic(
        heuristic_type=args.heuristic
    )

    results = []

    for idx, data_item in enumerate(dataset):      
        print(f"Processing {idx+1}/{len(dataset)}: {data_item['video_id']}")
        try:
            result = get_TStar_search_results(args, data_item=data_item, grounder=grounder, heurisiticFuncion=TStarHeuristic)
            print(f"Completed: {data_item['video_id']}\n")
        except Exception as e:
            print(f"Error processing {data_item['video_id']}: {e}")
            continue

        data_item.update(result)
        results.append(data_item)
    
    # Save batch results to output_json
    frame_search_root = "./results/frame_search"
    output_json = os.path.join(frame_search_root, f"{args.heuristic}_{args.output_json_name}")
    with open(output_json, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, indent=4, ensure_ascii=False)
    
    print(f"Batch processing completed. Results saved to {output_json}")

if __name__ == "__main__":

    """
    TStarSearcher: Comprehensive Video Frame Search and Question Answering Tool

    This script processes videos and associated questions (with or without multiple-choice options),
    and returns the top-K keyframe indices that best respond to the question using visual grounding
    and heuristic-based search.

    Core Features:
    - Utilizes the TStarFramework to conduct multi-stage visual reasoning over long videos.
    - Supports flexible grounding models (e.g., GPT-4, LLaVA) and visual heuristics (e.g., OWL-ViT, YOLO).
    - Enables batch processing of datasets and saves results in a structured JSON format.

    Usage:
    1. Data Preprocessing:
    Use a custom converter (e.g., `LVHaystack2TStarFormat`) to transform datasets such as 
    LVHaystack from HuggingFace into the required T* input format. 
    This involves aligning keys such as `video_id`, `question`, `options`, etc.

    2. Run the batch inference:
    ```bash
    python run_TStar_onDataset.py \
        --dataset_meta path/to/dataset_meta \
        --output_json path/to/output.json \
        --video_root path/to/video_folder

    The output JSON will include:
        Original fields (e.g., video_id, question, options, answer, etc.)
        Predicted keyframe timestamps (keyframe_timestamps)
        Grounded objects (grounding_objects)
        Frame-wise score distribution (frame_distribution)
    """
    main()

    
