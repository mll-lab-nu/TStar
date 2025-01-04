import os
import argparse
from tqdm import tqdm
from kfs.utils import load_questions, save_frames, format_result_data
from kfs.utils.search import extract_linear_frames

def process_video_entry(entry, idx_v, args):
    output_dir_base, num_frames, video_dir = args.output_dir, args.num_frames, args.video_dir
    """Process a single video entry from the questions data."""
    video_path = os.path.join(video_dir, entry['video_name'])


    video_output_dir = os.path.join(output_dir_base, str(idx_v))

    for idx_q, position_list in enumerate(entry['positions']):
        frame_output_dir = os.path.join(video_output_dir, str(idx_q))
        if not os.path.exists(frame_output_dir):  # Only process if directory does not exist
            frames = extract_linear_frames(video_path, num_frames, args.dry_run)
            save_frames(frames, frame_output_dir)

def prepare_data(args):
    """Prepare data based on the selected mode (oracle or result)."""
    data = load_questions(args.json_path)
    return data

def main(args):
    data = prepare_data(args)
    
    # Process each video entry
    for idx_v, entry in enumerate(tqdm(data, desc="Processing videos")):
        process_video_entry(entry, idx_v, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos based on provided positions.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file with video frame positions.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted frames.")
    parser.add_argument("--strategy", type=str, default="uniform", help="Frame extraction strategy: 'uniform' or others.")
    parser.add_argument("--num_frames", type=int, default=2, help="Number of frames to extract per question.")
    parser.add_argument("--video_dir", type=str, default="data/videos", help="Directory with the video files.")
    parser.add_argument("--dry_run", action="store_true", help="Dry run mode: do not actually save extracted frames. Just put a blank file there.")

    args = parser.parse_args()
    
    main(args)
