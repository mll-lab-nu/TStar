import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from decord import VideoReader, cpu
from scipy.interpolate import UnivariateSpline
import copy
from tqdm import tqdm
import os
import sys
import cv2
import copy
import logging

# Assuming YoloWorldInterface is defined elsewhere and imported correctly
# from your_project.yolo_interface import YoloWorldInterface
# 导入自定义的 TStar 接口
from TStar.interface_yolo import YoloWorldInterface, YoloV5Interface, YoloInterface




class TStarSearcher:
    """
    A class to perform keyframe search in a video using object detection and dynamic sampling.

    Attributes:
        video_path (str): Path to the video file.
        target_objects (List[str]): List of target objects to find.
        cue_objects (List[str]): List of cue objects for context.
        confidence_threshold (float): Minimum confidence threshold for object detection.
        search_nframes (int): Number of keyframes to search for.
        image_grid_shape (Tuple[int, int]): Shape of the image grid for detection.
        output_dir (Optional[str]): Directory to save outputs.
        profix (str): Prefix for output files.
        object2weight (dict): Weights assigned to specific objects.
        raw_fps (float): Original frames per second of the video.
        total_frame_num (int): Total number of frames adjusted for sampling rate.
        duration (float): Duration of the video in seconds.
        remaining_targets (List[str]): Targets yet to be found.
        search_budget (int): Budget for the number of frames to process.
        score_distribution (np.ndarray): Scores assigned to each frame.
        P_history (List[List[float]]): History of probability distributions.
        non_visiting_frames (np.ndarray): Indicator for frames not yet visited.
        yolo (YoloWorldInterface): YOLO interface for object detection.
    """

    def __init__(
        self,
        video_path: str,
        target_objects: List[str],
        cue_objects: List[str],
        search_nframes: int = 8,
        image_grid_shape: Tuple[int, int] = (8, 8),
        search_budget: float = 0.1,
        output_dir: Optional[str] = None,
        prefix: str = None,
        confidence_threshold: float = 0.5,
        object2weight: Optional[dict] = None,
        yolo_scorer: Optional[YoloInterface] = None,

    ):
        """
        Initializes the TStarSearcher object with video properties and configurations.
        
        Args:
            video_path (str): Path to the input video file.
            target_objects (List[str]): List of objects to detect as primary targets.
            cue_objects (List[str]): List of contextual objects to aid detection.
            cue_object (Optional[str]): A single cue object for additional focus.
            search_nframes (int): Number of keyframes to identify.
            image_grid_shape (Tuple[int, int]): Grid dimensions for image tiling.
            output_dir (Optional[str]): Directory to store results.
            profix (str): Prefix for saved output files.
            confidence_threshold (float): Threshold for object detection confidence.
            object2weight (Optional[dict]): Mapping of objects to their respective detection weights.
            config_path (str): Path to the YOLO configuration file.
            checkpoint_path (str): Path to the YOLO model checkpoint.
            device (str): Device for model inference (e.g., "cuda:0").
        """
        self.video_path = video_path
        self.target_objects = target_objects
        self.cue_objects = cue_objects
        self.search_nframes = search_nframes
        self.image_grid_shape = image_grid_shape
        self.output_dir = output_dir
        self.profix = prefix
        self.confidence_threshold = confidence_threshold
        self.object2weight = object2weight if object2weight else {}
        self.fps = 1  # Sampling at 1 fps

        # Video properties
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        self.raw_fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frame_num / self.raw_fps

        # Adjust total frame number based on sampling rate
        self.total_frame_num = int(self.duration * self.fps)
        self.remaining_targets = target_objects.copy()
        self.search_budget = min(1000, self.total_frame_num*search_budget)

        # Initialize distributions
        self.score_distribution = np.zeros(self.total_frame_num) #+ 0.1
        self.non_visiting_frames = np.ones(self.total_frame_num)
        self.P = np.ones(self.total_frame_num) * self.confidence_threshold * 0.3

        self.P_history = []
        self.Score_history = []
        self.non_visiting_history  = []
        # Initialize YOLO interface
        self.yolo = yolo_scorer
        self.reset_yolo_vocabulary(target_objects=target_objects, cue_objects=cue_objects)
        for object in target_objects:
            self.object2weight[object] = 1.0
        for object in cue_objects:
            self.object2weight[object] = 0.5

    def initialize_yolo(self, config_path: str, checkpoint_path: str, device: str = "cuda:0"):
        """
        Initializes the YOLO object detection model with the given configurations.

        Args:
            config_path (str): Path to the YOLO configuration file.
            checkpoint_path (str): Path to the YOLO model checkpoint.
            device (str): Device for model inference (e.g., "cuda:0").
        """
        self.yolo = YoloWorldInterface(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device
        )

    def reset_yolo_vocabulary(self, target_objects: List[str], cue_objects: List[str]):
        """
        Dynamically resets the YOLO vocabulary with the specified target and cue objects.

        Args:
            target_objects (List[str]): New list of target objects for detection.
            cue_objects (List[str]): New list of cue objects for detection context.
        """
        self.target_objects = target_objects
        self.cue_objects = cue_objects
        self.yolo.reparameterize_object_list(target_objects, cue_objects)

    ### --- Detection Methods --- ###

    def imageGridScoreFunction(
        self,
        images: List[np.ndarray],
        output_dir: Optional[str],
        image_grids: Tuple[int, int]
    ) -> Tuple[np.ndarray, List[List[List[str]]]]:
        """
        Perform object detection on a batch of images using the YOLO interface.

        Args:
            images (List[np.ndarray]): List of images to process.
            output_dir (Optional[str]): Directory to save detection results.
            image_grids (Tuple[int, int]): Dimensions of the image grid (rows, cols).

        Returns:
            Tuple[np.ndarray, List[List[List[str]]]]: Confidence maps and detected object lists.
                - confidence_maps: numpy array of shape (num_images, grid_rows, grid_cols)
                - detected_objects_maps: list of lists, each sublist corresponds to a grid_image and contains detected objects per cell
        """
        if len(images) == 0:
            return np.array([]), []

        grid_rows, grid_cols = image_grids
        grid_height = images[0].shape[0] / grid_rows
        grid_width = images[0].shape[1] / grid_cols

        confidence_maps = []
        detected_objects_maps = []

        # Perform detection on all images
        for image in images:
            # Run the YOLO inference
            detections = self.yolo.inference_detector(
                images=[image],  # Single image as a batch
                max_dets=50,
                use_amp=False
            )

            # Initialize confidence map and detected objects map
            confidence_map = np.zeros((grid_rows, grid_cols))
            detected_objects_map = [[] for _ in range(grid_rows * grid_cols)]

            # Process detections
            for detection in detections:
                for bbox, label, confidence in zip(detection.xyxy, detection.class_id, detection.confidence):
                    # Convert class ID to object name
                    object_name = self.yolo.texts[label][0] #@Jinhui TBD for YOLOWorld

                    # Apply object weight if available
                    weight = self.object2weight.get(object_name, 0.5)
                    adjusted_confidence = confidence * weight

                    # Calculate bounding box center
                    x_min, y_min, x_max, y_max = bbox
                    box_center_x = (x_min + x_max) / 2
                    box_center_y = (y_min + y_max) / 2

                    # Map center to grid cell
                    grid_x = int(box_center_x // grid_width)
                    grid_y = int(box_center_y // grid_height)

                    # Ensure grid indices are valid
                    grid_x = min(grid_x, grid_cols - 1)
                    grid_y = min(grid_y, grid_rows - 1)

                    # Update confidence map and detected objects
                    cell_index = grid_y * grid_cols + grid_x
                    confidence_map[grid_y, grid_x] = max(confidence_map[grid_y, grid_x], adjusted_confidence)
                    detected_objects_map[cell_index].append(object_name)

            confidence_maps.append(confidence_map)
            detected_objects_maps.append(detected_objects_map)

        return np.stack(confidence_maps), detected_objects_maps

    def read_frame_batch(self, video_path: str, frame_indices: List[int]) -> Tuple[List[int], np.ndarray]:
        """
        Reads a batch of frames from the video at specified indices.

        Args:
            video_path (str): Path to the video file.
            frame_indices (List[int]): Indices of frames to read.

        Returns:
            Tuple[List[int], np.ndarray]: List of indices and corresponding frame array.
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        return frame_indices, vr.get_batch(frame_indices).asnumpy()

    def create_image_grid(self, frames: List[np.ndarray], rows: int, cols: int) -> np.ndarray:
        """
        Combine frames into a single image grid.

        Args:
            frames (List[np.ndarray]): List of frame images.
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.

        Returns:
            np.ndarray: Combined image grid.
        """
        if len(frames) != rows * cols:
            raise ValueError("Frame count does not match grid dimensions")

        # Resize frames to fit the grid
        resized_frames = [cv2.resize(frame, (200, 95)) for frame in frames]  # Resize to 160x120
        grid_rows = [np.hstack(resized_frames[i * cols:(i + 1) * cols]) for i in range(rows)]
        return np.vstack(grid_rows)

    ### --- Scoring Methods --- ###

    def score_image_grids(
        self,
        images: List[np.ndarray],
        image_grids: Tuple[int, int]
    ) -> Tuple[np.ndarray, List[List[List[str]]]]:
        """
        Generate confidence maps and detected objects for each image grid.

        Args:
            images (List[np.ndarray]): List of image grids to detect objects.
            image_grids (Tuple[int, int]): Grid dimensions (rows, cols).

        Returns:
            Tuple[np.ndarray, List[List[List[str]]]]: Confidence maps and detected objects maps.
        """
        return self.imageGridScoreFunction(
            images=images,
            output_dir=self.output_dir,
            image_grids=image_grids
        )


    def store_score_distribution(self):
        """
        Stores a copy of the current probability distribution to the history.
        """
        self.P_history.append(copy.deepcopy(self.P).tolist())
        self.Score_history.append(copy.deepcopy(self.score_distribution).tolist())
        self.non_visiting_history.append(copy.deepcopy(self.non_visiting_frames).tolist())
    
    def update_top_25_with_window(
        self,
        frame_confidences: List[float],
        sampled_frame_indices: List[int],
        window_size: int = 5
    ):
        """
        Update score distribution for top 25% frames and their neighbors.

        Args:
            frame_confidences (List[float]): Confidence scores for sampled frames.
            sampled_frame_indices (List[int]): Corresponding frame indices.
            window_size (int): Number of neighboring frames to update.
        """
        # Calculate the threshold for top 25%
        top_25_threshold = np.percentile(frame_confidences, 75)

        # Identify top 25% frames
        top_25_indices = [
            frame_idx for frame_idx, confidence in zip(sampled_frame_indices, frame_confidences)
            if confidence >= top_25_threshold
        ]

        # Update neighboring frames
        for frame_idx in top_25_indices:
            for offset in range(-window_size, window_size + 1):
                neighbor_idx = frame_idx + offset
                if 0 <= neighbor_idx < len(self.score_distribution):
                    self.score_distribution[neighbor_idx] = max(
                        self.score_distribution[neighbor_idx],
                        self.score_distribution[frame_idx]/(abs(offset) + 1) 
                    )

    def spline_keyframe_distribution(
        self,
        non_visiting_frames: np.ndarray,
        score_distribution: np.ndarray,
        video_length: int
    ) -> np.ndarray:
        """
        Generate a probability distribution over frames using spline interpolation.

        Args:
            non_visiting_frames (np.ndarray): Indicator array for frames not yet visited.
            score_distribution (np.ndarray): Current score distribution over frames.
            video_length (int): Total number of frames.

        Returns:
            np.ndarray: Normalized probability distribution over frames.
        """
        # Extract indices and scores of visited frames
        frame_indices = np.array([idx for idx, visited in enumerate(non_visiting_frames) if visited == 0])
        observed_scores = np.array([score_distribution[idx] for idx in frame_indices])

        # If no frames have been visited, return uniform distribution
        if len(frame_indices) == 0:
            return np.ones(video_length) / video_length

        # Spline interpolation
        spline = UnivariateSpline(frame_indices, observed_scores, s=0.5)
        all_frames = np.arange(video_length)
        spline_scores = spline(all_frames)

        # Apply sigmoid function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        adjusted_scores = np.maximum(1 / video_length, spline_scores)
        p_distribution = sigmoid(adjusted_scores)

        # Normalize the distribution
        p_distribution /= p_distribution.sum()

        return p_distribution

    def update_frame_distribution(
        self,
        sampled_frame_indices: List[int],
        confidence_maps: np.ndarray,
        detected_objects_maps: List[List[List[str]]]
    ) -> Tuple[List[float], List[List[str]]]:
        """
        Update the frame distribution based on detection results.

        Args:
            sampled_frame_indices (List[int]): Indices of sampled frames.
            confidence_maps (np.ndarray): Confidence maps from detection.
            detected_objects_maps (List[List[List[str]]]): Detected objects from detection.

        Returns:
            Tuple[List[float], List[List[str]]]: Frame confidences and detected objects.
        """
        confidence_map = confidence_maps[0]  # Only one image grid @TBD
        detected_objects_map = detected_objects_maps[0]

        grid_rows, grid_cols = self.image_grid_shape

        frame_confidences = []
        frame_detected_objects = []
        for idx, frame_idx in enumerate(sampled_frame_indices):
            # Calculate grid cell position
            row = idx // grid_cols
            col = idx % grid_cols
            confidence = confidence_map[row, col]
            detected_objects = detected_objects_map[idx]
            frame_confidences.append(confidence)
            frame_detected_objects.append(detected_objects)

        # Update non-visiting frames and score distribution
        for frame_idx, confidence in zip(sampled_frame_indices, frame_confidences):
            self.non_visiting_frames[frame_idx] = 0  # Mark as visited
            self.score_distribution[frame_idx] = confidence

        # Update top 25% frames
        self.update_top_25_with_window(frame_confidences, sampled_frame_indices)

        # Update probability distribution
        self.P = self.spline_keyframe_distribution(
            self.non_visiting_frames,
            self.score_distribution,
            len(self.score_distribution)
        )

        # Store the updated distribution
        self.store_score_distribution()

        return frame_confidences, frame_detected_objects

    ### --- Sampling Methods --- ###

    def sample_frames(self, num_samples: int) -> Tuple[List[int], np.ndarray]:
        """
        Sample frames based on the current score distribution.

        Args:
            num_samples (int): Number of frames to sample.

        Returns:
            Tuple[List[int], np.ndarray]: Sampled frame indices and frame data.
        """
        if num_samples > self.total_frame_num:
            num_samples = self.total_frame_num

        if len(self.Score_history) == 0:  # If Score_history is empty, use uniform sampling with equal intervals
            # Ensure the frames are sampled with equal intervals
            interval = self.total_frame_num // num_samples  # Calculate the interval between frames
            sampled_frame_secs = np.arange(0, self.total_frame_num, interval)[:num_samples]  # Generate indices with equal intervals
            # If we have less samples than requested, we need to select the remaining samples in a way that doesn't exceed total_frame_num
            if len(sampled_frame_secs) < num_samples:
                # Add the last frame if not enough frames were selected
                sampled_frame_secs = np.append(sampled_frame_secs, self.total_frame_num - 1)

        else:
            # Adjust probabilities for visited frames
            _P = (self.P + num_samples / self.total_frame_num) * self.non_visiting_frames
            # Calculate the threshold for the top 25% frames
            threshold = np.percentile(_P, 75)  # Get the value at the 75th percentile (top 25%)

            # Filter out frames with scores below the threshold (keep only the top 25%)
            top_25_mask = _P >= threshold
            _P = _P * top_25_mask
            _P /= _P.sum()

            # Check if we have enough non-zero entries in the probability distribution
            non_zero_entries = np.count_nonzero(_P)
            if non_zero_entries < num_samples:
                # If not enough non-zero entries, adjust threshold or sample from all frames
                print(f"Warning: Not enough non-zero entries, adjusting threshold to sample {num_samples} frames.")
                _P =  (self.P + num_samples / self.total_frame_num) 
                _P /= _P.sum()

            # Sample frames based on the adjusted probabilities
            sampled_frame_secs = np.random.choice(
                self.total_frame_num,
                size=num_samples,
                replace=False,
                p=_P
            )
        # Convert sampled frame seconds to frame indices
        sampled_frame_indices = [int(sec * self.raw_fps / self.fps) for sec in sampled_frame_secs]

        # Read frames
        frame_indices, frames = self.read_frame_batch(
            video_path=self.video_path,
            frame_indices=sampled_frame_indices
        )

        resized_frames = [cv2.resize(frame, (200*4, 95*4)) for frame in frames]  # Resize to 160x120

        return sampled_frame_secs.tolist(), resized_frames

    ### --- Verification Methods --- ###

    def verify_and_remove_target(
        self,
        frame_sec: int,
        detected_objects: List[str],
        confidence_threshold: float,
    ) -> bool:
        """
        Verify target object detection in an individual frame and remove it from the target list if confirmed.

        Args:
            frame_sec (int): The timestamp of the frame in seconds.
            detected_objects (List[str]): Objects detected in the grid image for this frame.
            confidence_threshold (float): Threshold to confirm target detection.

        Returns:
            bool: True if a target was found and removed, False otherwise.
        """
        for target in list(self.remaining_targets):
            if target in detected_objects:
                frame_idx = int(frame_sec * self.raw_fps / self.fps)
                # Read the individual frame
                _, frames = self.read_frame_batch(self.video_path, [frame_idx])
                
                
                resized_frames = [cv2.resize(frame, (200*3, 95*3)) for frame in frames]  # Resize to 160x120
                frame = resized_frames[0]  # Extract the frame from the list
                # Perform detection on the individual frame
                single_confidence_maps, single_detected_objects_maps = self.score_image_grids(
                    [frame], (1, 1)
                )
                single_confidence = single_confidence_maps[0, 0, 0]
                single_detected_objects = single_detected_objects_maps[0][0]
                self.score_distribution[frame_sec] = single_confidence

                self.image_grid_iters.append([frame])
                self.detect_annotot_iters.append(self.yolo.bbox_visualization(images=[frame], detections_inbatch=self.yolo.detections_inbatch))
                self.detect_bbox_iters.append(self.yolo.detections_inbatch)
                
                # Check if target object confidence exceeds the threshold
                if target in single_detected_objects and single_confidence > confidence_threshold:
                    self.remaining_targets.remove(target)
                    print(f"Found target '{target}' in frame {frame_idx}, score {single_confidence:.2f}")
                    return True

        return False

    ### --- Visualization Methods --- ###

    def plot_score_distribution(self, save_path: Optional[str] = None):
        """
        Plot the score distribution over time.

        Args:
            save_path (Optional[str]): File path to save the plot.
        """
        time_axis = np.linspace(0, self.duration, len(self.score_distribution))

        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, self.score_distribution, label="Score Distribution")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Score")
        plt.title("Score Distribution Over Time")
        plt.grid(True)
        plt.legend()

        if save_path:
            plt.savefig(save_path, format='png', dpi=300)
            print(f"Plot saved to {save_path}")

        plt.show()

    ### --- Main Search Logic --- ###

    def search(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Perform the keyframe search based on object detection and dynamic sampling.

        Returns:
            Tuple[List[np.ndarray], List[float]]: Extracted keyframes and their timestamps.
        """
        
        K = self.search_nframes  # Number of keyframes to find
            # Estimate the total number of iterations based on search_budget and frames per iteration
        video_length = int(self.total_frame_num)
        
        # Initialize tqdm progress bar
        progress_bar = tqdm(total=video_length, desc="Searching Iterations / video_length", unit="iter", dynamic_ncols=True)
        
        while self.remaining_targets and self.search_budget > 0:
            grid_rows, grid_cols = self.image_grid_shape
            num_frames_in_grid = grid_rows * grid_cols

            # Sample frames based on the current distribution
            sampled_frame_secs, frames = self.sample_frames(num_frames_in_grid)
            self.search_budget -= num_frames_in_grid

            # Create an image grid from the sampled frames
            grid_image = self.create_image_grid(frames, grid_rows, grid_cols)

            # Perform object detection on the image grid
            confidence_maps, detected_objects_maps = self.score_image_grids(
                images=[grid_image],
                image_grids=self.image_grid_shape
            )

            # Update frame distributions based on detection results
            frame_confidences, frame_detected_objects = self.update_frame_distribution(
                sampled_frame_indices=sampled_frame_secs,
                confidence_maps=confidence_maps,
                detected_objects_maps=detected_objects_maps
            )

            # Verify and remove detected targets
            for frame_sec, detected_objects in zip(sampled_frame_secs, frame_detected_objects):
                self.verify_and_remove_target(
                    frame_sec=frame_sec,
                    detected_objects=detected_objects,
                    confidence_threshold=self.confidence_threshold,
                )
            # Update the progress bar
            progress_bar.update(1)
        
        # Close the progress bar once the loop is done
        progress_bar.close()
        # Select top K frames based on the score distribution
        top_k_indices = np.argsort(self.score_distribution)[-K:][::-1]
        top_k_frames = []
        time_stamps = []

        # Read and store the top K frames
        for idx in top_k_indices:
            frame_idx = int(idx * self.raw_fps / self.fps)
            _, frame = self.read_frame_batch(self.video_path, [frame_idx])
            top_k_frames.append(frame[0])
            time_stamps.append(idx / self.fps)

        return top_k_frames, time_stamps



    def search_with_visualization(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Perform the keyframe search based on object detection and dynamic sampling.

        Returns:
            Tuple[List[np.ndarray], List[float]]: Extracted keyframes and their timestamps.
        """


        # Initialize history 
        self.image_grid_iters = [] # iters, b, image
        self.detect_annotot_iters = [] # iters, b, image
        self.detect_bbox_iters = [] #iters, b, n_objects, xxyy
            
        K = self.search_nframes  # Number of keyframes to find
            # Estimate the total number of iterations based on search_budget and frames per iteration
        video_length = int(self.total_frame_num)
        
        # Initialize tqdm progress bar
        progress_bar = tqdm(total=video_length, desc="Searching Iterations / video_length", unit="iter", dynamic_ncols=True)
        
        while self.remaining_targets and self.search_budget > 0:
            grid_rows, grid_cols = self.image_grid_shape
            num_frames_in_grid = grid_rows * grid_cols

            # Sample frames based on the current distribution
            sampled_frame_secs, frames = self.sample_frames(num_frames_in_grid)
            self.search_budget -= num_frames_in_grid

            # Create an image grid from the sampled frames
            grid_image = self.create_image_grid(frames, grid_rows, grid_cols)
            

            # Perform object detection on the image grid
            confidence_maps, detected_objects_maps = self.score_image_grids(
                images=[grid_image],
                image_grids=self.image_grid_shape
            )

            self.image_grid_iters.append([grid_image])
            self.detect_annotot_iters.append(self.yolo.bbox_visualization(images=[grid_image], detections_inbatch=self.yolo.detections_inbatch))
            self.detect_bbox_iters.append(self.yolo.detections_inbatch)
            
            # Update frame distributions based on detection results
            frame_confidences, frame_detected_objects = self.update_frame_distribution(
                sampled_frame_indices=sampled_frame_secs,
                confidence_maps=confidence_maps,
                detected_objects_maps=detected_objects_maps
            )

            # Verify and remove detected targets
            for frame_sec, detected_objects in zip(sampled_frame_secs, frame_detected_objects):
                self.verify_and_remove_target(
                    frame_sec=frame_sec,
                    detected_objects=detected_objects,
                    confidence_threshold=self.confidence_threshold,
                )
            # Update the progress bar
            progress_bar.update(1)
        
        # Close the progress bar once the loop is done
        progress_bar.close()
        # Select top K frames based on the score distribution
        top_k_indices = np.argsort(self.score_distribution)[-K:][::-1]
        top_k_frames = []
        time_stamps = []

        # Read and store the top K frames
        for idx in top_k_indices:
            frame_idx = int(idx * self.raw_fps / self.fps)
            _, frame = self.read_frame_batch(self.video_path, [frame_idx])
            top_k_frames.append(frame[0])
            time_stamps.append(idx / self.fps)

        return top_k_frames, time_stamps





# Example usage
if __name__ == "__main__":
    # Define video path and target objects
    video_path = "./38737402-19bd-4689-9e74-3af391b15feb.mp4"
    query = "what is the color of the couch?"
    target_objects = ["couch"]  # Target objects to find
    cue_objects = ["TV", "chair"]

    # Create VideoSearcher instance
    searcher = TStarSearcher(
        video_path=video_path,
        target_objects=target_objects,
        cue_objects=cue_objects,
        search_nframes=8,
        image_grid_shape=(4, 4),
        confidence_threshold=0.6
    )

    # Perform the search
    all_frames, time_stamps = searcher.search()

    # Process results
    print(f"Found {len(all_frames)} frames, timestamps: {time_stamps}")

    # Plot the score distribution
    searcher.plot_score_distribution(save_path='/path/to/save/score_distribution.png')
