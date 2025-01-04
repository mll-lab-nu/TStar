import os
import argparse
from tqdm import tqdm
from kfs.utils import load_questions, extract_oracle_frames, save_frames, format_result_data

def process_video_entry(entry, idx_v, args):
    output_dir_base, video_dir = args.output_dir, args.video_dir
    """Process a single video entry from the questions data."""
    video_path = os.path.join(video_dir, entry['video_name'])
    video_output_dir = os.path.join(output_dir_base, str(idx_v))

    for idx_q, position_list in enumerate(entry['positions']):
        frame_output_dir = os.path.join(video_output_dir, str(idx_q))
        # if not os.path.exists(frame_output_dir):  # Only process if directory does not exist
        frames = extract_oracle_frames(video_path, position_list, args.dry_run)
        save_frames(frames, frame_output_dir)


def prepare_data(args):
    """Prepare data based on the selected mode (oracle or result)."""
    data = load_questions(args.json_path)

    if args.mode == 'result':
        assert args.fps_dict_path, "FPS dictionary is required for result mode."
        formatted_result = format_result_data(args.json_result_path, args.fps_dict_path)
        
        # Update each entry with frame positions from the result JSON
        for entry in data:
            video_name = entry['video_name']
            questions = entry['questions']
            entry['positions'] = [
                formatted_result.get(video_name, {}).get(question, [])
                for question in questions
            ]
    elif args.mode == 'oracle':
        # For oracle mode, assume positions are directly available in 'positions' field of `data`
        pass
    else:
        raise ValueError("Unsupported mode. Use 'oracle' or 'result'.")

    return data

def main(args):
    data = prepare_data(args)
    # Process each video entry
    for idx_v, entry in enumerate(tqdm(data, desc="Processing videos")):
        import time
        process_video_entry(entry, idx_v, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos based on provided positions.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file with video frame positions.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing the videos.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted frames.")
    parser.add_argument("--mode", type=str, choices=["oracle", "result"], required=True, help="Mode for frame extraction: 'oracle' or 'result'.")
    parser.add_argument("--json_result_path", type=str, default="", help="Path to the JSON result file (required if mode is 'result').")
    parser.add_argument("--fps_dict_path", type=str, help="Dictionary mapping video names to FPS values.")
    parser.add_argument("--dry_run", action="store_true", help="Dry run mode: do not actually save extracted frames. Just put a blank file there.")

    args = parser.parse_args()
    
    if args.mode == 'result' and not args.json_result_path:
        parser.error("--json_result_path is required when mode is 'result'")

    main(args)
