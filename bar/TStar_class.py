import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
from decord import VideoReader, cpu
from scipy.interpolate import UnivariateSpline
import copy


@dataclass(order=True)
class CellBlockItem:
    """
    Represents a block item in a grid cell with a score and associated metadata.
    """
    frame_index: int
    start_sec: float
    end_sec: float
    detect_object_list: List[str] = field(default_factory=list)
    score: float = field(init=False, compare=True)

    def __post_init__(self):
        self.score = 0.0  # Initialize score to 0

    def update_score(self, additional_score: float):
        """
        Update the score of the cell block item.

        Args:
            additional_score (float): Score to add.
        """
        self.score += additional_score


class TStarSampler:
    """
    TStarSampler handles frame sampling and distribution updates using specified strategies.
    """

    def __init__(self, video_path, total_frame_num: int, raw_fps: float, init_distribution: str = 'uniform', update_strategy: str = 'spline-prop'):
        """
        Initialize the TStarSampler.

        Args:
            total_frame_num (int): Total number of frames in the video.
            raw_fps (float): Original frames per second of the video.
            init_distribution (str, optional): Initial distribution type. Defaults to 'uniform'.
            update_strategy (str, optional): Strategy for updating the distribution. Defaults to 'spline-prop'.
        """
        self.total_frame_num = total_frame_num
        self.raw_fps = raw_fps
        self.init_distribution = init_distribution
        self.update_strategy = update_strategy
        self.video_path = video_path

        # Initialize distributions
        self.P = np.ones(self.total_frame_num) / self.total_frame_num  # Uniform initial distribution
        self.non_visiting_frames = np.ones(self.total_frame_num)
        self.score_distribution = np.zeros(self.total_frame_num)
        self.P_history = []

    def sample_frames(self, num_samples: int) -> List[int]:
        """
        Sample frames based on the current probability distribution.

        Args:
            num_samples (int): Number of frames to sample.

        Returns:
            List[int]: List of sampled frame indices.
        """
        if num_samples > self.total_frame_num:
            num_samples = self.total_frame_num

        # Adjust the probability distribution
        _P = self.non_visiting_frames * self.P + num_samples / self.total_frame_num
        _P = _P / _P.sum()  # Normalize

        sampled_frame_indices = np.random.choice(
            self.total_frame_num,
            size=num_samples,
            replace=False,
            p=_P
        )
        return sampled_frame_indices.tolist()

    def update_frame_distribution(self, sampled_frame_indices, confidence_maps, detected_objects_maps,image_grid_shape=(8,8)):

        """
        Update the frame distribution based on detection results.

        Args:
            sampled_frame_indices (List[int]): List of sampled frame indices.
            frame_confidences (List[float]): List of confidences corresponding to sampled frames.
        """


        confidence_map = confidence_maps[0]  # 只有一张图像
        detected_objects_map = detected_objects_maps[0]  # 只有一张图像

        # 将检测结果映射回对应的帧
        grid_rows, grid_cols = image_grid_shape

        frame_confidences = []
        frame_detected_objects = []
        for idx, frame_idx in enumerate(sampled_frame_indices):
            # 计算对应的网格单元
            row = idx // grid_cols
            col = idx % grid_cols
            confidence = confidence_map[row, col]
            detected_objects = detected_objects_map[idx]
            frame_confidences.append(confidence)
            frame_detected_objects.append(detected_objects)

        # 更新分布 P 和分数分布
        for frame_idx, confidence in zip(sampled_frame_indices, frame_confidences):
            self.non_visiting_frames[frame_idx] = 0  # 更新 non_visiting_frames
            self.score_distribution[frame_idx] = confidence

        for frame_idx, confidence in zip(sampled_frame_indices, frame_confidences):
            self.non_visiting_frames[frame_idx] = 0  # Mark frame as visited
            self.score_distribution[frame_idx] = confidence

        # Update top 25% frames with window
        self.update_top_25_with_window(frame_confidences, sampled_frame_indices)

        # Update probability distribution P
        if self.update_strategy == 'spline-prop':
            self.P = self.spline_keyframe_distribution()
        else:
            # Implement other update strategies if needed
            pass

        return frame_confidences, frame_detected_objects

    def update_top_25_with_window(self, frame_confidences: List[float], sampled_frame_indices: List[int], window_size: int = 5):
        """
        Update score distribution by extending high-confidence frames to their neighbors.

        Args:
            frame_confidences (List[float]): List of confidences.
            sampled_frame_indices (List[int]): List of sampled frame indices.
            window_size (int, optional): Window size for extending scores to neighbor frames. Defaults to 5.
        """
        # Compute the 75th percentile confidence as threshold
        top_25_threshold = np.percentile(frame_confidences, 75)

        # Get indices of frames with confidences above threshold
        top_25_indices = [
            frame_idx for frame_idx, confidence in zip(sampled_frame_indices, frame_confidences)
            if confidence >= top_25_threshold
        ]

        # Update neighboring frames' scores
        for frame_idx in top_25_indices:
            for offset in range(-window_size, window_size + 1):
                neighbor_idx = frame_idx + offset
                if 0 <= neighbor_idx < self.total_frame_num:
                    self.score_distribution[neighbor_idx] = max(
                        self.score_distribution[neighbor_idx],
                        self.score_distribution[frame_idx]
                    )

    def spline_keyframe_distribution(self) -> np.ndarray:
        """
        Update the probability distribution using spline-based propagation.

        Returns:
            np.ndarray: Updated probability distribution P.
        """
        # Get indices of visited frames
        frame_indices = np.where(self.non_visiting_frames == 0)[0]
        observed_scores = self.score_distribution[frame_indices]

        # If no frames have been observed, return uniform distribution
        if len(frame_indices) == 0:
            return np.ones(self.total_frame_num) / self.total_frame_num

        # Spline fitting
        spline = UnivariateSpline(frame_indices, observed_scores, s=0.5)

        # Predict scores for all frames
        all_frames = np.arange(self.total_frame_num)
        spline_scores = spline(all_frames)

        # Apply sigmoid function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Ensure scores are not less than 1 / total_frame_num
        adjusted_scores = np.maximum(1 / self.total_frame_num, spline_scores)

        # Compute probability distribution
        p_distribution = sigmoid(adjusted_scores)

        # Normalize
        p_distribution /= p_distribution.sum()

        return p_distribution

    def store_score_distribution(self):
        """
        Store the current score distribution to history.
        """
        self.P_history.append(self.score_distribution.copy())

    def read_frame_batch(self, frame_indices: List[int]) -> Tuple[List[int], List[np.ndarray]]:
        """
        Read a batch of frames from the video.

        Args:
            frame_indices (List[int]): List of frame indices to read.

        Returns:
            Tuple[List[int], List[np.ndarray]]: (frame_indices, frames)
        """
        vr = VideoReader(self.video_path, ctx=cpu(0))
        frames = vr.get_batch(frame_indices).asnumpy()
        return frame_indices, frames

    def create_image_grid(self, frames: List[np.ndarray], rows: int, cols: int) -> np.ndarray:
        """
        Stitch frames into an image grid.

        Args:
            frames (List[np.ndarray]): List of frames to stitch.
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.

        Returns:
            np.ndarray: The stitched image grid.
        """
        if len(frames) != rows * cols:
            raise ValueError("Number of frames does not match grid size")

        # Resize frames to fit the grid
        resized_frames = [cv2.resize(frame, (160, 120)) for frame in frames]
        grid_rows = []
        for i in range(rows):
            row = np.hstack(resized_frames[i * cols:(i + 1) * cols])
            grid_rows.append(row)
        grid_image = np.vstack(grid_rows)
        return grid_image

class ImageGridScorer:
    """
    ImageGridScorer handles object detection using YOLO models.
    """

    def __init__(self, confidence_threshold: float = 0.5, object2weight: Optional[dict] = None, model_name: str = 'yolov5s'):
        """
        Initialize the object detector.

        Args:
            confidence_threshold (float, optional): Confidence threshold for detections. Defaults to 0.5.
            object2weight (Optional[dict], optional): Weights for specific objects. Defaults to None.
            model_name (str, optional): YOLO model name to use. Defaults to 'yolov5s'.
        """
        self.confidence_threshold = confidence_threshold
        self.object2weight = object2weight if object2weight else {}
        self.model_name = model_name
        self.detector = None  # Will be initialized in initialize_model()

    def initialize_model(self):
        """
        Initialize the YOLO model for object detection.
        """
        # Initialize YOLO model
        self.detector = torch.hub.load('ultralytics/yolov5', self.model_name, pretrained=True)
        self.detector.conf = self.confidence_threshold

    def score_image_grids(self, images: List[np.ndarray], image_grids: Tuple[int, int]) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Perform object detection on image grids.

        Args:
            images (List[np.ndarray]): List of image grids.
            image_grids (Tuple[int, int]): The grid shape used for the images.

        Returns:
            Tuple[np.ndarray, List[List[str]]]: (confidence_maps, detected_objects_maps)
        """
        num_images = len(images)
        if num_images == 0:
            return np.array([]), []

        grid_rows, grid_cols = image_grids
        grid_height = images[0].shape[0] / grid_rows
        grid_width = images[0].shape[1] / grid_cols

        confidence_maps = []
        detected_objects_maps = []

        results = self.detector(images)
        for img_idx in range(num_images):
            result = results.xyxy[img_idx]
            confidence_map = np.zeros((grid_rows, grid_cols))
            detected_objects_map = [[] for _ in range(grid_rows * grid_cols)]

            for *bbox, conf, cls in result.cpu().numpy():
                class_name = self.detector.names[int(cls)]
                weight = self.object2weight.get(class_name, 1.0)
                adjusted_conf = conf * weight

                x_min, y_min, x_max, y_max = bbox
                box_center_x = (x_min + x_max) / 2
                box_center_y = (y_min + y_max) / 2

                grid_x = int(box_center_x // grid_width)
                grid_y = int(box_center_y // grid_height)

                grid_x = min(grid_x, grid_cols - 1)
                grid_y = min(grid_y, grid_rows - 1)

                confidence_map[grid_y, grid_x] += adjusted_conf

                cell_index = grid_y * grid_cols + grid_x
                detected_objects_map[cell_index].append(class_name)

            confidence_maps.append(confidence_map)
            detected_objects_maps.append(detected_objects_map)

        return np.stack(confidence_maps), detected_objects_maps


class VideoSearcher:
    """
    VideoSearcher performs keyframe search in a video for target objects.
    """

    def __init__(
        self,
        video_path: str,
        target_objects: List[str],
        cue_objects: List[str],
        cue_object: Optional[str] = None,
        search_nframes: int = 8,
        image_grid_shape: Tuple[int, int] = (8, 8),
        output_dir: Optional[str] = None,
        prefix: str = "stitched_image",
        confidence_threshold: float = 0.5,
        object2weight: Optional[dict] = None,
        yolo_model_name: str = 'yolov5s'
    ):
        """
        Initialize the VideoSearcher.

        Args:
            video_path (str): Path to the video file.
            target_objects (List[str]): List of target objects to search for.
            cue_objects (List[str]): List of cue objects.
            cue_object (Optional[str], optional): Specific cue object. Defaults to None.
            search_nframes (int, optional): Number of frames to search. Defaults to 8.
            image_grid_shape (Tuple[int, int], optional): Grid shape for image stitching. Defaults to (8, 8).
            output_dir (Optional[str], optional): Output directory for results. Defaults to None.
            prefix (str, optional): Prefix for output files. Defaults to "stitched_image".
            confidence_threshold (float, optional): Confidence threshold for detections. Defaults to 0.5.
            object2weight (Optional[dict], optional): Weights for specific objects. Defaults to None.
            yolo_model_name (str, optional): YOLO model name to use. Defaults to 'yolov5s'.
        """
        self.video_path = video_path
        self.target_objects = target_objects
        self.cue_objects = cue_objects
        self.search_nframes = search_nframes
        self.image_grid_shape = image_grid_shape
        self.output_dir = output_dir
        self.prefix = prefix
        self.confidence_threshold = confidence_threshold
        self.object2weight = object2weight if object2weight else {}

        # Initialize video reader
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        self.raw_fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frame_num / self.raw_fps
        self.total_frame_num = int(self.duration)  # Adjusted to 1 fps

        # Initialize remaining targets
        self.remaining_targets = target_objects.copy()
        # Initialize search budget
        self.search_budget = min(1000, self.total_frame_num)

        # Initialize TStarSampler
        self.video_sampler = TStarSampler(
            video_path=video_path,
            total_frame_num=self.total_frame_num,
            raw_fps=self.raw_fps,
            init_distribution='uniform',
            update_strategy='spline-prop'
        )

        # Initialize ImageGridScorer
        self.object_detector = ImageGridScorer(
            confidence_threshold=self.confidence_threshold,
            object2weight=self.object2weight,
            model_name=yolo_model_name
        )
        self.object_detector.initialize_model()

    def verify_and_remove_target(
        self,
        frame_idx: int,
        detected_objects: List[str],
        target_objects: List[str],
        video_sampler: TStarSampler,
        object_detector: ImageGridScorer,
        confidence_threshold: float,
    ) -> bool:
        """
        Verify target object detection in an individual frame and remove it from the target list if confirmed.

        Args:
            frame_idx (int): The index of the frame to verify.
            detected_objects (List[str]): Objects detected in the grid image for this frame.
            target_objects (List[str]): List of remaining target objects.
            video_sampler (TStarSampler): Sampler to read the specific frame.
            object_detector (ImageGridScorer): Object detector to perform individual frame detection.
            confidence_threshold (float): Threshold to confirm target detection.

        Returns:
            bool: True if a target was found and removed, False otherwise.
        """
        for target in list(target_objects):
            if target in detected_objects:
                # Read the individual frame
                _, frame = video_sampler.read_frame_batch([frame_idx])
                frame = frame[0]  # Extract the frame from the list

                # Perform detection on the individual frame
                single_confidence_maps, single_detected_objects_maps = object_detector.score_image_grids(
                    [frame], (1, 1)
                )
                single_confidence = single_confidence_maps[0, 0]
                single_detected_objects = single_detected_objects_maps[0]

                # Check if target object confidence exceeds the threshold
                if target in single_detected_objects and single_confidence > confidence_threshold:
                    target_objects.remove(target)
                    print(f"Found target '{target}' in frame {frame_idx}, score {single_confidence:.2f}")
                    return True
        return False

    def search(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Execute the keyframe search process.

        Returns:
            Tuple[List[np.ndarray], List[float]]: (all_frames, time_stamps)
        """
        K = self.search_nframes
        self.video_sampler.P_history = []
        self.video_sampler.non_visiting_frames = np.ones(self.total_frame_num)

        while self.remaining_targets and self.search_budget > 0:
            grid_rows, grid_cols = self.image_grid_shape
            num_frames_in_grid = grid_rows * grid_cols
            sampled_frame_indices = self.video_sampler.sample_frames(num_frames_in_grid)
            frame_indices, frames = self.video_sampler.read_frame_batch(sampled_frame_indices)
            grid_image = self.video_sampler.create_image_grid(frames, grid_rows, grid_cols)
            
            self.search_budget -= num_frames_in_grid
            print(f"Sampled frame indices: {sampled_frame_indices}")

            # Perform object detection using ImageGridScorer
            confidence_maps, detected_objects_maps = self.object_detector.score_image_grids([grid_image], self.image_grid_shape)
            # Update the sampler with detection results
            cell_detected_confidences, cell_detected_objects = self.video_sampler.update_frame_distribution(sampled_frame_indices, confidence_maps, detected_objects_maps, image_grid_shape=self.image_grid_shape)

            # 检查是否检测到目标对象
            for frame_idx, detected_objects in zip(sampled_frame_indices, cell_detected_objects):
                for target in list(self.remaining_targets):
                    if target in detected_objects and self.video_sampler.score_distribution[frame_idx] > self.confidence_threshold:
                        self.verify_and_remove_target(
                            frame_idx=frame_idx,
                            detected_objects=detected_objects,
                            target_objects=self.remaining_targets,
                            video_sampler=self.video_sampler,
                            object_detector=self.object_detector,
                            confidence_threshold=self.confidence_threshold,
                        )


        # Retrieve top K frames
        top_k_indices = np.argsort(self.video_sampler.score_distribution)[-K:][::-1]
        top_k_frames = []
        time_stamps = []

        # Read top K frames
        for frame_idx in top_k_indices:
            _, frames = self.video_sampler.read_frame_batch([frame_idx])
            top_k_frames.append(frames[0])
            time_stamps.append(frame_idx / self.raw_fps)

        return top_k_frames, time_stamps

    def plot_score_distribution(self, save_path: Optional[str] = None):
        """
        Plot the score distribution history.

        Args:
            save_path (Optional[str], optional): Path to save the plot. Defaults to None.
        """
        time_axis = np.arange(0, self.duration, 1 / self.raw_fps)
        plt.figure(figsize=(12, 6))

        for i, history in enumerate(self.video_sampler.P_history):
            if (i + 1) % 3 != 0:
                continue
            plt.plot(
                time_axis[:len(history)],
                history,
                label=f'History {i+1}'
            )

        plt.xlabel('Time (seconds)')
        plt.ylabel('Score')
        plt.title('Score Distribution History')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, format='png', dpi=300)
            print(f"Plot saved to {save_path}")

        plt.show()


if __name__ == "__main__":
    # Define video path and target objects
    video_path = "/home/yejinhui/Projects/VisualSearch/38737402-19bd-4689-9e74-3af391b15feb.mp4"
    query = "What is the color of the couch?"
    target_objects = ["couch"]  # Example target object
    cue_objects = ["TV", "chair"]

    # Create VideoSearcher instance
    searcher = VideoSearcher(
        video_path=video_path,
        target_objects=target_objects,
        cue_objects=cue_objects,
        search_nframes=8,
        image_grid_shape=(4, 4),
        confidence_threshold=0.6,
        yolo_model_name='yolov5s'
    )

    # Execute search
    all_frames, time_stamps = searcher.search()

    # Handle results
    print(f"Found {len(all_frames)} frames, time stamps: {time_stamps}")

    # Plot score distribution
    searcher.plot_score_distribution(save_path='/home/yejinhui/Projects/VisualSearch/38737402-19bd.png')

    # Optional: Save or display results
    # for idx, frame in enumerate(all_frames):
    #     cv2.imwrite(f"frame_{idx}.jpg", frame)
