import os
import argparse
import warnings
from TStar.interface_grounding import TStarUniversalGrounder
from TStar.TStarFramework import TStarFramework, run_tstar



def main():
    """
    TStarSearcher: Simplified Video Frame Search Tool
    
    Example usage:
        python searcher.py --video_path path/to/video.mp4 --question "Your question here" --options "A) Option1\nB) Option2\nC) Option3\nD) Option4"
    """
    parser = argparse.ArgumentParser(description="TStarSearcher: Simplified Video Frame Search and QA Tool")
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--question', type=str, required=True, help='Question for video content QA.')
    parser.add_argument('--options', type=str, required=True, help='Multiple-choice options for the question.')
    
    # search tools
    parser.add_argument('--grounder', type=str, default='gpt-4o', help='Directory to save outputs.')
    parser.add_argument('--heuristic', type=str, default='owl-vit', help='Directory to save outputs.')
    
    parser.add_argument('--device', type=str, default="cuda:0", help='Device for model inference (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--search_nframes', type=int, default=8, help='Number of top frames to return.')
    parser.add_argument('--grid_rows', type=int, default=4, help='Number of rows in the image grid.')
    parser.add_argument('--grid_cols', type=int, default=4, help='Number of columns in the image grid.')
    parser.add_argument('--confidence_threshold', type=float, default=0.6, help='YOLO detection confidence threshold.')
    parser.add_argument('--search_budget', type=float, default=0.5, help='Maximum ratio of frames to process during search.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save outputs.')
    
    args = parser.parse_args()

    # Run the TStar search process
    results = run_tstar(
        video_path=args.video_path,
        question=args.question,
        options=args.options,
        grounder=args.grounder,
        heuristic=args.heuristic,
        search_nframes=args.search_nframes,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        confidence_threshold=args.confidence_threshold,
        search_budget=args.search_budget,
        output_dir=args.output_dir,
    )

    # Display the results
    print("#"*20)
    print(f"Input Quetion: {args.question}")
    print(f"Input Options: {args.question}")
    print("#"*20)
    print("T* Results:")
    print(f"Grounding Objects: {results['Grounding Objects']}")
    print(f"Frame Timestamps: {results['Frame Timestamps']}")
    print(f"Answer: {results['Answer']}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore all warnings
        main()
