import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from decord import VideoReader, cpu
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm

# Import the YOLO interface and HeuristicInterface from TStar package
from TStar.interface_heuristic import YoloWorldInterface, HeuristicInterface


class TStarSearcher:
    """
    A class to perform keyframe search in a video using object detection and dynamic sampling.

    External attributes and interfaces remain unchanged.
    """

    def __init__(
        self,
        video_path: str,
        heuristic: HeuristicInterface,
        target_objects: List[str],
        cue_objects: List[str],
        search_nframes: int = 8,
        image_grid_shape: Tuple[int, int] = (8, 8),
        search_budget: float = 0.1,
        output_dir: Optional[str] = None,
        confidence_threshold: float = 0.5,
        object2weight: Optional[dict] = None,
    ):
        """
        Initialize TStarSearcher with video properties and configuration.

        Args:
            video_path (str): Path to the input video file.
            heuristic (HeuristicInterface): YOLO interface for detection.
            target_objects (List[str]): List of primary objects to detect.
            cue_objects (List[str]): List of contextual objects to aid detection.
            search_nframes (int): Number of keyframes to search for.
            image_grid_shape (Tuple[int, int]): Dimensions for tiling images.
            search_budget (float): Fraction (or capped value) for the number of frames to process.
            output_dir (Optional[str]): Directory for saving outputs.
            confidence_threshold (float): Detection confidence threshold.
            object2weight (Optional[dict]): Mapping of object names to their detection weights.
        """
        self.video_path = video_path
        self.target_objects = target_objects
        self.cue_objects = cue_objects
        self.search_nframes = search_nframes
        self.image_grid_shape = image_grid_shape
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.object2weight = object2weight if object2weight else {}
        self.fps = 1  # Sampling rate: 1 frame per second

        # Video properties
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        self.raw_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = total_frames / self.raw_fps

        # Adjust total frame number based on sampling rate
        self.total_frame_num = int(self.duration * self.fps)
        self.remaining_targets = target_objects.copy()
        self.search_budget = min(1000, self.total_frame_num * search_budget)

        # Initialize distributions and histories
        self.score_distribution = np.zeros(self.total_frame_num)
        self.non_visiting_frames = np.ones(self.total_frame_num)
        self.P = np.ones(self.total_frame_num) * self.confidence_threshold * 0.3

        self.P_history = []
        self.Score_history = []
        self.non_visiting_history = []

        # Set YOLO interface (heuristic)
        self.heuristic = heuristic
        self.heuristic.reparameterize_object_list(target_objects, cue_objects)
        for obj in target_objects:
            self.object2weight[obj] = 1.0
        for obj in cue_objects:
            self.object2weight[obj] = 0.5

    # --- Detection Methods ---
    def imageGridScoreFunction(
        self,
        images: List[np.ndarray],
        output_dir: Optional[str],
        image_grids: Tuple[int, int]
    ) -> Tuple[np.ndarray, List[List[List[str]]]]:
        """
        Run object detection on a batch of images using the YOLO interface and map detections to a grid.

        Args:
            images (List[np.ndarray]): List of images.
            output_dir (Optional[str]): Directory to save results (unused in refactoring).
            image_grids (Tuple[int, int]): (rows, cols) grid dimensions.

        Returns:
            Tuple containing:
                - confidence_maps: np.ndarray with shape (num_images, grid_rows, grid_cols)
                - detected_objects_maps: List of detected objects per grid cell for each image.
        """
        if not images:
            return np.array([]), []

        grid_rows, grid_cols = image_grids
        grid_height = images[0].shape[0] / grid_rows
        grid_width = images[0].shape[1] / grid_cols

        confidence_maps = []
        detected_objects_maps = []

        for image in images:
            detections = self.heuristic.inference_detector(
                images=[image],
                use_amp=False
            )

            # Initialize map for each grid cell
            confidence_map = np.zeros((grid_rows, grid_cols))
            detected_objects_map = [[] for _ in range(grid_rows * grid_cols)]

            for detection in detections:
                for bbox, label, confidence in zip(detection.xyxy, detection.class_id, detection.confidence):
                    object_name = self.heuristic.texts[label][0]  # Map class id to name
                    weight = self.object2weight.get(object_name, 0.5)
                    adjusted_confidence = confidence * weight

                    x_min, y_min, x_max, y_max = bbox
                    box_center_x = (x_min + x_max) / 2
                    box_center_y = (y_min + y_max) / 2

                    grid_x = int(box_center_x // grid_width)
                    grid_y = int(box_center_y // grid_height)
                    grid_x = min(grid_x, grid_cols - 1)
                    grid_y = min(grid_y, grid_rows - 1)

                    cell_index = grid_y * grid_cols + grid_x
                    confidence_map[grid_y, grid_x] = max(confidence_map[grid_y, grid_x], adjusted_confidence)
                    detected_objects_map[cell_index].append(object_name)

            confidence_maps.append(confidence_map)
            detected_objects_maps.append(detected_objects_map)

        return np.stack(confidence_maps), detected_objects_maps

    def read_frame_batch(self, video_path: str, frame_indices: List[int]) -> Tuple[List[int], np.ndarray]:
        """
        Read a batch of frames from the video at specified indices.

        Args:
            video_path (str): Video file path.
            frame_indices (List[int]): Indices of frames to read.

        Returns:
            Tuple of frame indices and corresponding frame data (numpy array).
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        return frame_indices, vr.get_batch(frame_indices).asnumpy()

    def create_image_grid(self, frames: List[np.ndarray], rows: int, cols: int) -> np.ndarray:
        """
        Combine a list of frames into a single image grid.

        Args:
            frames (List[np.ndarray]): List of frames.
            rows (int): Number of grid rows.
            cols (int): Number of grid columns.

        Returns:
            np.ndarray: Combined image grid.
        """
        if len(frames) != rows * cols:
            raise ValueError("Frame count does not match grid dimensions")
        # Resize frames (hardcoded to 200x95 here)
        resized_frames = [cv2.resize(frame, (200, 95)) for frame in frames]
        grid_rows = [np.hstack(resized_frames[i * cols:(i + 1) * cols]) for i in range(rows)]
        return np.vstack(grid_rows)

    def score_image_grids(
        self,
        images: List[np.ndarray],
        image_grids: Tuple[int, int]
    ) -> Tuple[np.ndarray, List[List[List[str]]]]:
        """
        Generate confidence maps and detected objects for image grids.

        Args:
            images (List[np.ndarray]): List of grid images.
            image_grids (Tuple[int, int]): Grid dimensions (rows, cols).

        Returns:
            Tuple containing confidence maps and detected objects maps.
        """
        return self.imageGridScoreFunction(images, self.output_dir, image_grids)

    def store_score_distribution(self):
        """
        Save a copy of the current probability distribution and histories.
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
        Update the score distribution for the top 25% frames and their neighbors.

        Args:
            frame_confidences (List[float]): Confidence scores for frames.
            sampled_frame_indices (List[int]): Corresponding frame indices.
            window_size (int): Number of neighboring frames to update.
        """
        top_25_threshold = np.percentile(frame_confidences, 75)
        top_25_indices = [
            frame_idx for frame_idx, confidence in zip(sampled_frame_indices, frame_confidences)
            if confidence >= top_25_threshold
        ]
        for frame_idx in top_25_indices:
            for offset in range(-window_size, window_size + 1):
                neighbor_idx = frame_idx + offset
                if 0 <= neighbor_idx < len(self.score_distribution):
                    self.score_distribution[neighbor_idx] = max(
                        self.score_distribution[neighbor_idx],
                        self.score_distribution[frame_idx] / (abs(offset) + 1)
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
            non_visiting_frames (np.ndarray): Array indicating unvisited frames.
            score_distribution (np.ndarray): Current score distribution.
            video_length (int): Total number of frames.

        Returns:
            np.ndarray: Normalized probability distribution.
        """
        visited_indices = np.array([idx for idx, visited in enumerate(non_visiting_frames) if visited == 0])
        observed_scores = np.array([score_distribution[idx] for idx in visited_indices])
        if len(visited_indices) == 0:
            return np.ones(video_length) / video_length

        spline = UnivariateSpline(visited_indices, observed_scores, s=0.5)
        all_frames = np.arange(video_length)
        spline_scores = spline(all_frames)

        # Apply a sigmoid function to smooth the scores
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        adjusted_scores = np.maximum(1 / video_length, spline_scores)
        p_distribution = sigmoid(adjusted_scores)
        p_distribution /= p_distribution.sum()
        return p_distribution

    def update_frame_distribution(
        self,
        sampled_frame_indices: List[int],
        confidence_maps: np.ndarray,
        detected_objects_maps: List[List[List[str]]]
    ) -> Tuple[List[float], List[List[str]]]:
        """
        Update frame distribution based on detection results.

        Args:
            sampled_frame_indices (List[int]): Indices of sampled frames.
            confidence_maps (np.ndarray): Confidence maps from detection.
            detected_objects_maps (List[List[List[str]]]): Detected objects per grid cell.

        Returns:
            Tuple containing:
                - List of frame confidences.
                - List of detected objects for each frame.
        """
        # Assuming one grid image is used
        confidence_map = confidence_maps[0]
        detected_objects_map = detected_objects_maps[0]
        grid_rows, grid_cols = self.image_grid_shape

        frame_confidences = []
        frame_detected_objects = []
        for idx, _ in enumerate(sampled_frame_indices):
            row = idx // grid_cols
            col = idx % grid_cols
            frame_confidences.append(confidence_map[row, col])
            frame_detected_objects.append(detected_objects_map[idx])

        # Mark frames as visited and update scores
        for frame_idx, confidence in zip(sampled_frame_indices, frame_confidences):
            self.non_visiting_frames[frame_idx] = 0
            self.score_distribution[frame_idx] = confidence

        self.update_top_25_with_window(frame_confidences, sampled_frame_indices)
        self.P = self.spline_keyframe_distribution(
            self.non_visiting_frames,
            self.score_distribution,
            len(self.score_distribution)
        )
        self.store_score_distribution()

        return frame_confidences, frame_detected_objects

    # --- Sampling Methods ---
    def sample_frames(self, num_samples: int) -> Tuple[List[int], List[np.ndarray]]:
        """
        Sample frames based on the current probability distribution.

        Args:
            num_samples (int): Number of frames to sample.

        Returns:
            Tuple containing:
                - List of sampled frame "seconds" (indices in the sampling rate).
                - List of resized frame images.
        """
        if num_samples > self.total_frame_num:
            num_samples = self.total_frame_num

        if not self.Score_history:
            interval = self.total_frame_num // num_samples
            sampled_frame_secs = np.arange(0, self.total_frame_num, interval)[:num_samples]
            if len(sampled_frame_secs) < num_samples:
                sampled_frame_secs = np.append(sampled_frame_secs, self.total_frame_num - 1)
        else:
            _P = (self.P + num_samples / self.total_frame_num) * self.non_visiting_frames
            threshold = np.percentile(_P, 75)
            top_25_mask = _P >= threshold
            _P = _P * top_25_mask
            if _P.sum() == 0 or np.count_nonzero(_P) < num_samples:
                print(f"Warning: Not enough non-zero entries, adjusting probability distribution.")
                _P = (self.P + num_samples / self.total_frame_num)
            _P /= _P.sum()
            sampled_frame_secs = np.random.choice(
                self.total_frame_num,
                size=num_samples,
                replace=False,
                p=_P
            )

        sampled_frame_indices = [int(sec * self.raw_fps / self.fps) for sec in sampled_frame_secs]
        indices, frames = self.read_frame_batch(self.video_path, sampled_frame_indices)
        resized_frames = [cv2.resize(frame, (200 * 4, 95 * 4)) for frame in frames]
        return sampled_frame_secs.tolist(), resized_frames

    # --- Verification Methods ---
    def verify_and_remove_target(
        self,
        frame_sec: int,
        detected_objects: List[str],
        confidence_threshold: float,
    ) -> bool:
        """
        Verify detection of a target in a frame and remove it from remaining targets if confirmed.

        Args:
            frame_sec (int): Frame timestamp (in sampled seconds).
            detected_objects (List[str]): Detected objects in the frame.
            confidence_threshold (float): Threshold for confirmation.

        Returns:
            bool: True if a target is found and removed, else False.
        """
        for target in list(self.remaining_targets):
            if target in detected_objects:
                frame_idx = int(frame_sec * self.raw_fps / self.fps)
                _, frame = self.read_frame_batch(self.video_path, [frame_idx])
                resized_frame = cv2.resize(frame[0], (200 * 3, 95 * 3))
                conf_map, det_obj_map = self.score_image_grids([resized_frame], (1, 1))
                single_confidence = conf_map[0, 0, 0]
                single_detected_objects = det_obj_map[0][0]
                self.score_distribution[frame_sec] = single_confidence

                self.image_grid_iters.append([resized_frame])
                self.detect_annotot_iters.append(self.heuristic.bbox_visualization(
                    images=[resized_frame],
                    detections_inbatch=self.heuristic.detections_inbatch
                ))
                self.detect_bbox_iters.append(self.heuristic.detections_inbatch)

                if target in single_detected_objects and single_confidence > confidence_threshold:
                    self.remaining_targets.remove(target)
                    print(f"Found target '{target}' in frame {frame_idx}, score {single_confidence:.2f}")
                    return True
        return False

    # --- Visualization Methods ---
    def plot_score_distribution(self, save_path: Optional[str] = None):
        """
        Plot the score distribution over time.

        Args:
            save_path (Optional[str]): Path to save the plot image.
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

    # --- Main Search Logic ---
    def search(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Perform keyframe search using object detection and dynamic sampling.

        Returns:
            Tuple containing:
                - List of keyframe images.
                - List of corresponding timestamps.
        """
        K = self.search_nframes
        video_length = int(self.total_frame_num)
        progress_bar = tqdm(total=video_length, desc="Searching Iterations", unit="iter", dynamic_ncols=True)

        while self.remaining_targets and self.search_budget > 0:
            grid_rows, grid_cols = self.image_grid_shape
            num_frames_in_grid = grid_rows * grid_cols
            sampled_frame_secs, frames = self.sample_frames(num_frames_in_grid)
            self.search_budget -= num_frames_in_grid

            grid_image = self.create_image_grid(frames, grid_rows, grid_cols)
            confidence_maps, detected_objects_maps = self.score_image_grids(
                images=[grid_image],
                image_grids=self.image_grid_shape
            )
            # Append grid and detection visualization for history
            self.image_grid_iters.append([grid_image])
            self.detect_annotot_iters.append(self.heuristic.bbox_visualization(
                images=[grid_image],
                detections_inbatch=self.heuristic.detections_inbatch
            ))
            self.detect_bbox_iters.append(self.heuristic.detections_inbatch)

            frame_confidences, frame_detected_objects = self.update_frame_distribution(
                sampled_frame_indices=sampled_frame_secs,
                confidence_maps=confidence_maps,
                detected_objects_maps=detected_objects_maps
            )
            for frame_sec, detected_objects in zip(sampled_frame_secs, frame_detected_objects):
                self.verify_and_remove_target(
                    frame_sec=frame_sec,
                    detected_objects=detected_objects,
                    confidence_threshold=self.confidence_threshold,
                )
            progress_bar.update(1)
        progress_bar.close()

        top_k_indices = np.argsort(self.score_distribution)[-K:][::-1]
        top_k_frames = []
        time_stamps = []
        for idx in top_k_indices:
            frame_idx = int(idx * self.raw_fps / self.fps)
            _, frame = self.read_frame_batch(self.video_path, [frame_idx])
            top_k_frames.append(frame[0])
            time_stamps.append(idx / self.fps)

        return top_k_frames, time_stamps

    def search_with_visualization(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Perform keyframe search and maintain visual history.

        Returns:
            Tuple containing keyframe images and their timestamps.
        """
        self.image_grid_iters = []      # List of image grid iterations
        self.detect_annotot_iters = []  # List of annotated image iterations
        self.detect_bbox_iters = []     # List of bbox detections per iteration

        K = self.search_nframes
        video_length = int(self.total_frame_num)
        progress_bar = tqdm(total=video_length, desc="Searching Iterations", unit="iter", dynamic_ncols=True)

        while self.remaining_targets and self.search_budget > 0:
            grid_rows, grid_cols = self.image_grid_shape
            num_frames_in_grid = grid_rows * grid_cols
            sampled_frame_secs, frames = self.sample_frames(num_frames_in_grid)
            self.search_budget -= num_frames_in_grid

            grid_image = self.create_image_grid(frames, grid_rows, grid_cols)
            confidence_maps, detected_objects_maps = self.score_image_grids(
                images=[grid_image],
                image_grids=self.image_grid_shape
            )
            self.image_grid_iters.append([grid_image])
            self.detect_annotot_iters.append(self.heuristic.bbox_visualization(
                images=[grid_image],
                detections_inbatch=self.heuristic.detections_inbatch
            ))
            self.detect_bbox_iters.append(self.heuristic.detections_inbatch)

            frame_confidences, frame_detected_objects = self.update_frame_distribution(
                sampled_frame_indices=sampled_frame_secs,
                confidence_maps=confidence_maps,
                detected_objects_maps=detected_objects_maps
            )
            for frame_sec, detected_objects in zip(sampled_frame_secs, frame_detected_objects):
                self.verify_and_remove_target(
                    frame_sec=frame_sec,
                    detected_objects=detected_objects,
                    confidence_threshold=self.confidence_threshold,
                )
            progress_bar.update(1)
        progress_bar.close()

        top_k_indices = np.argsort(self.score_distribution)[-K:][::-1]
        top_k_frames = []
        time_stamps = []
        for idx in top_k_indices:
            frame_idx = int(idx * self.raw_fps / self.fps)
            _, frame = self.read_frame_batch(self.video_path, [frame_idx])
            top_k_frames.append(frame[0])
            time_stamps.append(idx / self.fps)

        return top_k_frames, time_stamps


# Example usage
if __name__ == "__main__":
    video_path = "./38737402-19bd-4689-9e74-3af391b15feb.mp4"
    query = "what is the color of the couch?"
    target_objects = ["couch"]
    cue_objects = ["TV", "chair"]

