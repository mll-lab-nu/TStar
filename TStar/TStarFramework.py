
"""
TStarSearcher: Comprehensive Video Frame Search Tool

This script allows searching for specific objects within a video using YOLO object detection and GPT-4 for question-answering. It leverages the TStar framework's universal Grounder, YOLO interface, and video searcher to identify relevant frames and answer questions based on the detected objects.

Usage:
    python tstar_searcher.py --video_path path/to/video.mp4 --question "Your question here" --options "A) Option1\nB) Option2\nC) Option3\nD) Option4"
"""

import os
import sys
import cv2
import torch
import copy
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from decord import VideoReader, cpu
from scipy.interpolate import UnivariateSpline
from PIL import Image
# Import custom TStar interfaces
from TStar.interface_llm import TStarUniversalGrounder
from TStar.interface_yolo import YoloWorldInterface, YoloInterface
from TStar.interface_searcher import TStarSearcher
from TStar.utilites import save_as_gif
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TStarFramework:
    """
    Main class for performing object-based frame search and question-answering in a video.
    """

    def __init__(
        self,
        video_path: str,
        yolo_scorer: YoloInterface,
        grounder: TStarUniversalGrounder,
        question: str,
        options: str,
        search_nframes: int = 8,
        grid_rows: int = 4,
        grid_cols: int = 4,
        output_dir: str = './output',
        confidence_threshold: float = 0.6,
        search_budget: int = 1000,
        prefix: str = 'stitched_image',
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda:0"
    ):
        """
        Initialize VideoSearcher.

        Args:
            video_path (str): Path to the input video file.
            yolo_scorer (YoloV5Interface): YOLO interface instance.
            grounder (TStarUniversalGrounder): Universal Grounder instance.
            question (str): The question for question-answering.
            options (str): Multiple-choice options for the question.
            search_nframes (int, optional): Number of top frames to return. Default is 8.
            grid_rows (int, optional): Number of rows in the image grid. Default is 4.
            grid_cols (int, optional): Number of columns in the image grid. Default is 4.
            output_dir (str, optional): Directory to save outputs. Default is './output'.
            confidence_threshold (float, optional): YOLO detection confidence threshold. Default is 0.6.
            search_budget (int, optional): Maximum number of frames to process during search. Default is 1000.
            prefix (str, optional): Prefix for output filenames. Default is 'stitched_image'.
            config_path (str, optional): Path to the YOLO configuration file. Default is None.
            checkpoint_path (str, optional): Path to the YOLO model checkpoint. Default is None.
            device (str, optional): Device for model inference (e.g., "cuda:0" or "cpu"). Default is "cuda:0".
        """
        self.video_path = video_path
        self.yolo_scorer = yolo_scorer
        self.grounder = grounder
        self.question = question
        self.options = options
        self.search_nframes = search_nframes
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.search_budget = search_budget
        self.prefix = prefix
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device

        self.video_id=self.video_path.split("/")[-1].split(".")[0]
        self.output_dir = os.path.join(self.output_dir, self.video_id)  # 视频保存路径
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("VideoSearcher initialized successfully.")

        self.results = {}

    def run(self):
        """
        Execute the complete video search and question-answering process.
        """
        # Use Grounder to get target and cue objects
        target_objects, cue_objects = self.get_grounded_objects()

        # Initialize TStarSearcher
        video_searcher  = self.set_searching_targets(target_objects, cue_objects)

        
        logger.info(f"TStarSearcher initialized successfully for video {self.video_path}.")

        # Perform search
        all_frames, time_stamps = self.perform_search(video_searcher)

        # Save retrieved frames
        self.save_frames(all_frames, time_stamps)
        self.save_searching_iters(video_searcher)
        # Plot and save score distribution
        self.plot_and_save_scores(video_searcher)

        # Perform question-answering on retrieved frames
        answer = self.perform_qa(all_frames)
        print("QA Answer:", answer)

        logger.info("VideoSearcher completed successfully.")

    def get_grounded_objects(self) -> Tuple[List[str], List[str]]:
        """
        Use Grounder to obtain target and cue objects.

        Returns:
            Tuple[List[str], List[str]]: Lists of target objects and cue objects.
        """
        # Example code; should be implemented based on Grounder's interface
        # For example:
        target_objects, cue_objects = self.grounder.inference_query_grounding(
            video_path=self.video_path,
            question=self.question
        )
        # Here, assuming fixed target and cue objects
        # target_objects = ["couch"]  # Target objects to find
        # cue_objects = ["TV", "chair"]  # Cue objects

        logger.info(f"Target objects: {target_objects}")
        logger.info(f"Cue objects: {cue_objects}")
        self.results["Searching_Objects"] = {"target_objects": target_objects, "cue_objects": cue_objects}
        return target_objects, cue_objects
    
    def set_searching_targets(self, target_objects, cue_objects):
        """
        Initialize and configure the TStarSearcher for video object search.

        Args:
            target_objects (List[str]): List of target objects to search for in the video.
            cue_objects (List[str]): List of cue objects to assist in locating target objects.

        Returns:
            TStarSearcher: Configured instance of the TStarSearcher class.

        Notes:
            - The `TStarSearcher` is responsible for performing the object search within the video
            using the specified targets and cues.
            - Key parameters such as the search frame limit (`search_nframes`), grid shape (`image_grid_shape`),
            confidence threshold (`confidence_threshold`), and search budget (`search_budget`) are passed
            to the `TStarSearcher`.
            - The `yolo_scorer` is used as the object detection model for evaluating the objects in the video.
        """
        video_searcher = TStarSearcher(
            video_path=self.video_path,
            target_objects=target_objects,
            cue_objects=cue_objects,
            search_nframes=self.search_nframes,
            image_grid_shape=(self.grid_rows, self.grid_cols),
            output_dir=self.output_dir,
            confidence_threshold=self.confidence_threshold,
            search_budget=self.search_budget,
            yolo_scorer=self.yolo_scorer
        )

        return video_searcher



    def perform_search(self, video_searcher: TStarSearcher) -> Tuple[List[np.ndarray], List[float]]:
        """
        Execute the frame search process and retrieve relevant frames and timestamps.

        Args:
            video_searcher (TStarSearcher): Instance of TStarSearcher.

        Returns:
            Tuple[List[np.ndarray], List[float]]: List of frames and their corresponding timestamps.
        """
        all_frames, time_stamps = video_searcher.search_with_visualization()
        logger.info(f"Found {len(all_frames)} frames, timestamps: {time_stamps}")
        
        self.results['timestamps'] = time_stamps
        return all_frames, time_stamps

    def perform_qa(self, frames: List[np.ndarray]) -> str:
        """
        Perform question-answering on the retrieved frames.

        Args:
            frames (List[np.ndarray]): List of frames to analyze.

        Returns:
            str: Answer generated by VLM.
        """
        answer = self.grounder.inference_qa(
            frames=frames,
            question=self.question,
            options=self.options
        )
        self.results['answer'] = answer
        return answer

    def plot_and_save_scores(self, video_searcher: TStarSearcher):
        """
        Plot the score distribution and save the plot.

        Args:
            video_searcher (TStarSearcher): Instance of TStarSearcher.
        """
        plot_path = os.path.join(self.output_dir, "score_distribution.png")
        video_searcher.plot_score_distribution(save_path=plot_path)
        logger.info(f"Score distribution plot saved to {plot_path}")

        # save_P_history as .git

    def save_frames(self, frames: List[np.ndarray], timestamps: List[float]):
        """
        Save the retrieved frames as image files.

        Args:
            frames (List[np.ndarray]): List of frames to save.
            timestamps (List[float]): Corresponding timestamps of the frames.
        """
        # Ensure the output directory exists
        
        
        output_dir = os.path.join(self.output_dir, "frame_sampling")
        os.makedirs(output_dir, exist_ok=True)
        for idx, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            frame_path = os.path.join(
                output_dir,
                f"frame_{idx}_at_{timestamp:.2f}s.jpg"
            )
            
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved frame to {frame_path}")

    def save_searching_iters(self, video_searcher):

        image_grid_iters = video_searcher.image_grid_iters # iters, b, image # b = 1 for v1
        detect_annotot_iters = video_searcher.detect_annotot_iters # iters, b, image
        detect_bbox_iters = video_searcher.detect_bbox_iters #iters, b, n_objects, xxyy, 
            
        fps = 1  # 设置帧率为 1
        for b in range(len(image_grid_iters[0])):
            images =  [image_grid_iter[b] for image_grid_iter in image_grid_iters]
            anno_images = [detect_annotot_iter[b] for detect_annotot_iter in detect_annotot_iters] 

            frame_size = (anno_images[0].shape[1], anno_images[0].shape[0])  # 获取图像大小 (宽度, 高度)

            # 设置视频的参数
            video_id=self.video_path.split("/")[-1].split(".")[0]
            question = self.question
            output_video_path = os.path.join(self.output_dir, f"{question}.gif")  # 视频保存路径
            save_as_gif(images=anno_images, output_gif_path=output_video_path)
            self.save_score_history_as_gif(video_searcher)
            self.save_Score_history_as_heatmap_gif(video_searcher)

    def save_p_history_as_gif(self, video_searcher, output_gif_path=None, fps=1):
        """
        Save P_history as a GIF, with each iteration represented as a frame.

        Args:
            video_searcher: Object containing P_history data.
            output_gif_path: File path to save the GIF.
            fps: Frames per second for the GIF.
        """
        frames = []
        duration_per_frame = 1000 // fps  # Convert fps to milliseconds
        question = self.question
        output_gif_path = os.path.join(self.output_dir, f"{question}_distribution.gif")  # 视频保存路径

        # Create a temporary directory to save individual plots
        temp_dir = os.path.join(os.path.dirname(output_gif_path), "temp_p_history_frames")
        os.makedirs(temp_dir, exist_ok=True)

        # Generate a plot for each iteration in P_history
        for i, iteration in enumerate(video_searcher.P_history):
            plt.figure(figsize=(10, 6))
            plt.plot(iteration, label=f"Iteration {i + 1}")
            plt.xlabel("Frame Index")
            plt.ylabel("Value")
            plt.title(f"P_history Iteration {i + 1}")
            plt.grid(True)
            plt.legend()

            # Save the plot as a temporary PNG file
            temp_file_path = os.path.join(temp_dir, f"frame_{i + 1}.png")
            plt.savefig(temp_file_path, format='png', dpi=300)
            plt.close()  # Free memory

            # Add the frame to the list of images for the GIF
            frames.append(Image.open(temp_file_path))

        # Save all frames as a GIF
        frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_per_frame,
            loop=0
        )
        print(f"GIF saved to {output_gif_path}")
        # Clean up temporary directory
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)

    def save_score_history_as_gif(self, video_searcher, output_gif_path=None, fps=1):
        """
        Save Score_history as a GIF, with each iteration represented as a frame.

        Args:
            video_searcher: Object containing Score_history data.
            output_gif_path: File path to save the GIF.
            fps: Frames per second for the GIF.
        """
        frames = []
        duration_per_frame = 1000 // fps  # Convert fps to milliseconds
        question = self.question
        output_gif_path = os.path.join(self.output_dir, f"{question[:-1]}_score_distribution.gif")  # 视频保存路径

        # Create a temporary directory to save individual plots
        temp_dir = os.path.join(os.path.dirname(output_gif_path), "temp_score_history_frames")
        os.makedirs(temp_dir, exist_ok=True)

        # Generate a plot for each iteration in Score_history
        for i, iteration in enumerate(video_searcher.Score_history):
            spline_scores = self.spline_scores(score_distribution=iteration, non_visiting_frames=video_searcher.non_visiting_history[i],
                                               video_length=video_searcher.total_frame_num)
            

            plt.figure(figsize=(10, 6))
            # Highlight sampling points with red dots
            sampled_indices = [idx for idx, visited in enumerate(video_searcher.non_visiting_history[i]) if visited == 0]
            sampled_values = [0 for idx in sampled_indices]
            plt.scatter(sampled_indices, sampled_values, color='red',s=10, label="Sampled Frames")

            plt.plot(spline_scores, label=f"Target Frame Belief")


            plt.xlabel("Timeslot (sec)")
            plt.ylabel("Score Value")
            plt.title(f"Score_history Iteration {i + 1}")
            plt.ylim(0, 1)  # Fix y-axis range to 0-1
            plt.xlim(0, len(spline_scores)+5)  # Fix y-axis range to 0-1
            plt.grid(True)
            plt.legend()

            # Save the plot as a temporary PNG file
            temp_file_path = os.path.join(temp_dir, f"frame_{i + 1}.png")
            plt.savefig(temp_file_path, format='png', dpi=300)
            plt.close()  # Free memory

            # Add the frame to the list of images for the GIF
            frames.append(Image.open(temp_file_path))

        # Save all frames as a GIF
        frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_per_frame,
            loop=0
        )
        print(f"GIF saved to {output_gif_path}")

        # Clean up temporary directory
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)

    def spline_scores(self, score_distribution, non_visiting_frames, video_length):
        # Extract indices and scores of visited frames
        frame_indices = np.array([idx for idx, visited in enumerate(non_visiting_frames) if visited == 0])
        observed_scores = np.array([score_distribution[idx] for idx in frame_indices])

        # If no frames have been visited, return uniform distribution
        if len(frame_indices) == 0:
            return np.ones(video_length) / video_length

        # Spline interpolation
        spline = UnivariateSpline(frame_indices, observed_scores, s=0.8)
        all_frames = np.arange(video_length)
        spline_scores = spline(all_frames)
        # spline_scores = spline_scores / spline_scores.sum()
        return spline_scores

    def save_Score_history_as_heatmap_gif(self, video_searcher, output_gif_path=None, fps=1):
        """
        Save Score_history as a heatmap GIF, with each iteration represented as a frame.

        Args:
            video_searcher: Object containing Score_history data.
            output_gif_path: File path to save the GIF.
            fps: Frames per second for the GIF.
        """
        frames = []
        duration_per_frame = 1000 // fps  # Convert fps to milliseconds
        question = self.question
        output_gif_path = os.path.join(self.output_dir, f"{question}_score_heatmap.gif")  # 视频保存路径

        # Create a temporary directory to save individual plots
        temp_dir = os.path.join(os.path.dirname(output_gif_path), "temp_score_heatmap_frames")
        os.makedirs(temp_dir, exist_ok=True)

        # Generate a heatmap for each iteration in Score_history
        for i, iteration in enumerate(video_searcher.Score_history):
            spline_scores = self.spline_scores(score_distribution=iteration, non_visiting_frames=video_searcher.non_visiting_history[i],
                                               video_length=video_searcher.total_frame_num)
            

            plt.figure(figsize=(10, 2))
            
            # Convert Score_history to a 2D array (e.g., 1 x len(iteration)) for heatmap visualization
            heatmap_data = np.array([spline_scores])
            sns.heatmap(heatmap_data, cmap="viridis", cbar=True, xticklabels=False, yticklabels=False, vmin=0, vmax=1)

            plt.title(f"Score History Heatmap - Iteration {i + 1}")
            plt.xlabel("Frame Index")
            plt.ylabel("Heatmap")

            # Save the heatmap as a temporary PNG file
            temp_file_path = os.path.join(temp_dir, f"frame_{i + 1}.png")
            plt.savefig(temp_file_path, format='png', dpi=300)
            plt.close()  # Free memory

            # Add the frame to the list of images for the GIF
            frames.append(Image.open(temp_file_path))

        # Save all frames as a GIF
        frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_per_frame,
            loop=0
        )
        print(f"Heatmap GIF saved to {output_gif_path}")

        # Clean up temporary directory
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)

    def set_to_3D(self):

        pass
import seaborn as sns
def initialize_yolo(
    config_path: str,
    checkpoint_path: str,
    device: str
) -> YoloInterface:
    """
    Initialize the YOLO object detection model.

    Args:
        config_path (str): Path to the YOLO configuration file.
        checkpoint_path (str): Path to the YOLO model checkpoint.
        device (str): Device for model inference (e.g., "cuda:0").

    Returns:
        YoloWorldInterface: Initialized YOLO interface instance.

    Raises:
        FileNotFoundError: If the configuration file or checkpoint file is not found.
    """

    yolo = YoloWorldInterface(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device
    )
    logger.info("YoloWorldInterface initialized successfully.")
    return yolo


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="TStarSearcher: Video Frame Search and QA Tool")
    parser.add_argument('--video_path', type=str, default="./38737402-19bd-4689-9e74-3af391b15feb.mp4", help='Path to the input video file.')
    parser.add_argument('--question', type=str, default="What is the color of my couch?", help='Question for video content QA.')
    parser.add_argument('--options', type=str, default="A) Red\nB) Black\nC) Green\nD) White\n", help='Multiple-choice options for the question, e.g., "A) Option1\nB) Option2\nC) Option3\nD) Option4"')
    parser.add_argument('--config_path', type=str, default="./YOLOWorld/configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py", help='Path to the YOLO configuration file.')
    parser.add_argument('--checkpoint_path', type=str, default="./pretrained/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth", help='Path to the YOLO model checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device for model inference (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--search_nframes', type=int, default=8, help='Number of top frames to return.')
    parser.add_argument('--grid_rows', type=int, default=4, help='Number of rows in the image grid.')
    parser.add_argument('--grid_cols', type=int, default=4, help='Number of columns in the image grid.')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='YOLO detection confidence threshold.')
    parser.add_argument('--search_budget', type=float, default=0.5, help='Maximum ratio of frames to process during search.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save outputs.')
    parser.add_argument('--prefix', type=str, default='stitched_image', help='Prefix for output filenames.')
    return parser.parse_args()


def main():
    """
    Main function to execute TStarSearcher.
    """
    args = parse_arguments()

    # Initialize Grounder
    grounder = TStarUniversalGrounder(
        backend="gpt4",
        gpt4_model_name="gpt-4o"
    )
    logger.info("TStarUniversalGrounder initialized successfully.")

    # Initialize YOLO interface
    yolo_interface = initialize_yolo(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device
    )

    # Initialize VideoSearcher
    searcher = TStarFramework(
        grounder=grounder,
        yolo_scorer=yolo_interface,
        video_path=args.video_path,
        question=args.question,
        options=args.options,
        search_nframes=args.search_nframes,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        search_budget=args.search_budget,
        prefix=args.prefix,
        device=args.device
    )

    # Run the search and QA process
    searcher.run()

    # Output the results
    print("Final Results:")
    print(f"Grounding Objects: {searcher.results['Searching_Objects']}")
    print(f"Frame Timestamps: {searcher.results['timestamps']}")
    print(f"Answer: {searcher.results['answer']}")


    


if __name__ == "__main__":
    main()
