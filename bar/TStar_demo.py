import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from decord import VideoReader, cpu

import numpy as np
from scipy.interpolate import UnivariateSpline


@dataclass(order=True)
class CellBlockItem:
    score: float = field(init=False, compare=True)
    frame_index: int
    start_sec: float
    end_sec: float
    detect_object_list: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.score = 0.0  # 初始化评分为0
    
    def update_score(self, additional_score: float):
        self.score += additional_score


class TStarSampler:
    def __init__(self, init_distribution, update_strategy):
    

        pass
    
    def reset(self):
        pass

    def sample(self, video, current_grid):
        pass

    def update(self, scores):
        pass




class VideoSearcher:
    def __init__(
        self,
        video_path: str,
        target_objects: List[str],
        cue_objects: List[str],
        cue_object: Optional[str] = None,
        search_nframes: int = 8,
        image_grid_shape: Tuple[int, int] = (8, 8),
        output_dir: Optional[str] = None,
        profix: str = "stitched_image",
        confidence_threshold: float = 0.5,
        object2weight: Optional[dict] = None
    ):
        self.video_path = video_path
        self.target_objects = target_objects
        self.cue_objects = cue_objects
        self.search_nframes = search_nframes
        self.image_grid_shape = image_grid_shape
        self.output_dir = output_dir
        self.profix = profix
        self.confidence_threshold = confidence_threshold
        self.object2weight = object2weight if object2weight else {}
        self.mrr = 10
        self.fps = 1

        # 初始化视频读取器
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")
        self.raw_fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frame_num / self.raw_fps
        self.total_frame_num = int(self.total_frame_num / self.raw_fps * self.fps) # 1 fps
        # 初始化目标对象
        self.remaining_targets = target_objects
        # 初始化搜索预算
        self.search_budget = min(1000, self.total_frame_num)
        self.score_distribution = np.zeros(self.total_frame_num)


    def initialize_yolo(self, detective_object_list=None, YOLO="yolov5"):
        # 这个模型用于评价图片不同部分对
        # TODO: 初始化YOLO模型
        # self.detect_model = YourYOLOModel()
        # self.detect_objects = ["person", "car", "bicycle"]  # 示例
        # self.test_pipeline = ...  # 定义您的测试管道
        if "yolov5" in YOLO:
            # 初始化YOLO模型
            self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) 
            #
            # self.detector = torch.hub.load('wondervictor/YOLO-World', 'yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395', pretrained=True)
            # # we can not load yolo world for hub
            self.detector.conf = self.confidence_threshold  # 设置置信度阈值
            return
        
        # self.object2weight = heuristicer.object2weight
        # self.heuristicer = heuristicer

    def read_frame_batch(self, video_path, frame_indices):
        
        vr = VideoReader(video_path, ctx=cpu(0))  # 每个线程独立创建 VideoReader
        return frame_indices, vr.get_batch(frame_indices).asnumpy()


    def create_image_grid(self, frames: List[np.ndarray], rows: int, cols: int) -> np.ndarray:
        """将帧列表拼接成一个图像网格"""
        if len(frames) != rows * cols:
            raise ValueError("帧数量与网格尺寸不匹配")

        # 调整每帧的大小以适应网格
        resized_frames = [cv2.resize(frame, (160, 120)) for frame in frames]  # 调整到160x120
        grid_rows = []
        for i in range(rows):
            row = np.hstack(resized_frames[i * cols:(i + 1) * cols])
            grid_rows.append(row)
        grid_image = np.vstack(grid_rows)
        return grid_image

    def score_patches_yolov5(self, images: List[np.ndarray], output_dir: Optional[str], profix: str, image_grids: Tuple[int, int]) -> Tuple[np.ndarray, List[List[str]]]:
        """
        使用YOLO模型对一组图像网格进行对象检测，并生成每个网格单元的置信度累加值和检测到的对象列表。
        
        Args:
            images (List[np.ndarray]): 要检测的图像网格列表。
            output_dir (str): 检测结果的输出目录。
            profix (str): 图像前缀。
            image_grids (Tuple[int, int]): 图像网格的行数和列数。
        
        Returns:
            Tuple[np.ndarray, List[List[str]]]: (confidence_maps, detected_objects_maps)
                - confidence_maps: numpy array of shape (num_images, grid_rows, grid_cols)
                - detected_objects_maps: list of lists, each sublist corresponds to a grid_image and contains detected objects per cell
        """
        num_images = len(images)
        
        if num_images == 0:
            return np.array([]), []
        
        grid_rows, grid_cols = image_grids
        grid_height = images[0].shape[0] / grid_rows
        grid_width = images[0].shape[1] / grid_cols
        
        confidence_maps = []
        detected_objects_maps = []
        
        # 使用YOLO模型进行批量检测
        results = self.detector(images)  # Batch size = num_images
        
        for img_idx in range(num_images):
            result = results.xyxy[img_idx]  # 结果是一个[N, 6]的Tensor，包含[x1, y1, x2, y2, conf, cls]
            confidence_map = np.zeros((grid_rows, grid_cols))
            detected_objects_map = [[] for _ in range(grid_rows * grid_cols)]
            
            for *bbox, conf, cls in result.cpu().numpy():
                class_name = self.detector.names[int(cls)]

                if class_name not in self.target_objects + self.cue_objects:
                    continue
                # 根据 object2weight 调整置信度
                weight = self.object2weight.get(class_name, 1.0)
                adjusted_conf = conf * weight
                
                # 计算边界框中心点
                x_min, y_min, x_max, y_max = bbox
                box_center_x = (x_min + x_max) / 2
                box_center_y = (y_min + y_max) / 2
                
                # 计算中心点所在的网格坐标
                grid_x = int(box_center_x // grid_width)
                grid_y = int(box_center_y // grid_height)
                
                # 确保索引在有效范围内
                grid_x = min(grid_x, grid_cols - 1)
                grid_y = min(grid_y, grid_rows - 1)
                
                # 累加置信度
                confidence_map[grid_y, grid_x] += adjusted_conf
                
                # 添加检测到的对象
                cell_index = grid_y * grid_cols + grid_x
                detected_objects_map[cell_index].append(class_name)
            
            confidence_maps.append(confidence_map)
            detected_objects_maps.append(detected_objects_map)
        
        return np.stack(confidence_maps), detected_objects_maps  # Shape: (num_images, grid_rows, grid_cols), list of list
    def score_image_grids(self, images: List[np.ndarray], image_grids: Tuple[int, int]) -> Tuple[np.ndarray, List[List[str]]]:
        """
        生成每个网格单元的置信度累加值和检测到的对象列表。
        
        Args:
            images (List[np.ndarray]): 要检测的图像网格列表。
            output_dir (str): 检测结果的输出目录。
            profix (str): 图像前缀。
            image_grids (Tuple[int, int]): 图像网格的行数和列数。
        
        Returns:
            Tuple[np.ndarray, List[List[str]]]: (confidence_maps, detected_objects_maps)
                - confidence_maps: numpy array of shape (num_images, grid_rows, grid_cols)
                - detected_objects_maps: list of lists, each sublist corresponds to a grid_image and contains detected objects per cell
        """
        return self.score_patches_yolov5(
                images=images,
                output_dir=self.output_dir,
                profix=images,
                image_grids=image_grids
            )
    def score_patches_yolov10(self, images: List[np.ndarray], output_dir: Optional[str], profix: str, image_grids: Tuple[int, int]) -> Tuple[np.ndarray, List[List[str]]]:
        """
        使用YOLO模型对一组图像网格进行对象检测，并生成每个网格单元的置信度累加值和检测到的对象列表。
        
        Args:
            images (List[np.ndarray]): 要检测的图像网格列表。
            output_dir (str): 检测结果的输出目录。
            profix (str): 图像前缀。
            image_grids (Tuple[int, int]): 图像网格的行数和列数。
        
        Returns:
            Tuple[np.ndarray, List[List[str]]]: (confidence_maps, detected_objects_maps)
                - confidence_maps: numpy array of shape (num_images, grid_rows, grid_cols)
                - detected_objects_maps: list of lists, each sublist corresponds to a grid_image and contains detected objects per cell
        """
        # 计算图像的数量
        num_images = len(images)
        
        # 如果没有图像，返回空
        if num_images == 0:
            return np.array([]), []
        
        # 分割网格的大小
        grid_rows, grid_cols = image_grids
        grid_height = images[0].shape[0] / grid_rows
        grid_width = images[0].shape[1] / grid_cols
        
        confidence_maps = []
        detected_objects_maps = []
        
        # 检测物体 #TBD
        detect_dict_list = self.heuristicer.inference_detector(
            self.heuristicer.detect_model,
            images,
            self.heuristicer.detect_objects,
            self.heuristicer.test_pipeline,
            35,
            0.1,
            output_dir=output_dir,
            use_amp=False,
            show=False,
            annotation=False
        )
        
        for detect_dict in detect_dict_list:
            # 创建一个 2D 数组存储每个网格的置信度累加值
            confidence_map = np.zeros((grid_rows, grid_cols))
            # 创建一个列表存储每个网格的检测到的对象列表
            detected_objects_map = [[] for _ in range(grid_rows * grid_cols)]
            
            # 遍历所有检测到的 bounding boxes
            for detected_box_i in range(len(detect_dict.xyxy)):
                box_area = detect_dict.xyxy[detected_box_i]
                box_class_id = detect_dict.class_id[detected_box_i]
                box_confidence = detect_dict.confidence[detected_box_i]
                box_class_name = self.heuristicer.detect_objects[box_class_id][0]

                # 根据 self.object2weight 调整置信度
                weight = self.object2weight.get(box_class_name, 1.0)  # 如果没有找到，默认为1
                box_confidence *= weight

                # 计算边界框的中心点
                x_min, y_min, x_max, y_max = box_area
                box_center_x = (x_min + x_max) / 2
                box_center_y = (y_min + y_max) / 2

                # 计算中心点所在的网格坐标
                grid_x = int(box_center_x // grid_width)
                grid_y = int(box_center_y // grid_height)

                # 确保索引在有效范围内
                grid_x = min(grid_x, grid_cols - 1)
                grid_y = min(grid_y, grid_rows - 1)

                # 将置信度累加到相应的网格区域 # do add
                best_det = max(confidence_map[grid_y, grid_x], box_confidence)
                confidence_map[grid_y, grid_x] = best_det

                # 添加检测到的对象
                cell_index = grid_y * grid_cols + grid_x
                detected_objects_map[cell_index].append(box_class_name)
            
            confidence_maps.append(confidence_map)
            detected_objects_maps.append(detected_objects_map)
        
        return np.stack(confidence_maps), detected_objects_maps  # Shape: (num_images, grid_rows, grid_cols), list of list

    def update_score_distribution(self, frame_indices: List[int], confidences: List[float]):
        """根据帧索引和对应的置信度更新分数分布"""
        for frame_idx, confidence in zip(frame_indices, confidences):
            self.score_distribution[frame_idx] = confidence
        
        self.smooth_score_distribution()
    
    def store_score_distribution(self):
        """
        对 score_distribution 进行深度拷贝并返回拷贝后的副本。
        
        返回:
        - dict: 深度拷贝后的 score_distribution 副本。

        """
        self.P_hository.append(copy.deepcopy(self.score_distribution).tolist())
        return None

    def smooth_score_distribution(self, window_size=3):
        """
        对 score_distribution 进行时间维度上的平滑。
        
        参数:
        - window_size: int，平滑窗口的半径，例如 1 表示取左右各 1 帧的分数来平滑。
        """
        smoothed_scores = {}

        # 遍历所有帧，计算平滑后的得分
        for frame_idx in self.score_distribution.keys():
            # 获取当前帧左右相邻的帧索引
            start_idx = max(0, frame_idx - window_size)
            end_idx = min(frame_idx + window_size + 1, self.total_frame_num)
            
            # 计算窗口内的平均分数
            window_scores = [
                self.score_distribution.get(i, 0) for i in range(start_idx, end_idx)
            ]
            smoothed_scores[frame_idx] = sum(window_scores) / len(window_scores)

        # 更新 score_distribution 为平滑后的分数
        self.score_distribution = smoothed_scores
    def plot_score_distribution(self, save_path=None):
        

        """
        绘制所有历史分数分布图，X轴为时间（秒），Y轴为分数。
        
        参数:
        - save_path: str (可选)，如果提供，将图像保存到指定路径。
        """
        time_axis = np.arange(0, self.duration, 1 / self.raw_fps)

        plt.figure(figsize=(12, 6))

        # 遍历 P_hository 中的每个历史分布，并绘制曲线
        for i, history in enumerate(self.P_hository):
            if (i + 1) % 5 != 0:
                continue
            plt.plot(
                time_axis[:len(history)], 
                list(history), 
                label=f'History {i+1}'
            )
        
        plt.xlabel('时间 (秒)')
        plt.ylabel('分数')
        plt.title('所有历史分数分布')
        plt.legend()
        plt.grid(True)
        
        # 检查是否提供了保存路径
        if save_path:
            plt.savefig(save_path, format='png', dpi=300)  # 将图片保存为高分辨率 PNG 格式
            print(f"图片已保存至 {save_path}")
        
        plt.show()
    
    def sample_frames(self, num_samples: int, fps=1):
        """根据当前分布 P 采样帧索引"""
        if num_samples > self.total_frame_num:
            num_samples = self.total_frame_num
        
        _P = (self.P + num_samples/self.total_frame_num ) * (self.non_visiting_frames)
        _P = _P / _P.sum()  # 重新归一化 --> 不能提前
        sampled_frames_insec = np.random.choice(
            self.total_frame_num,
            size=num_samples,
            replace=False,
            p=_P
        )

        sampled_frame_indexes = sampled_frames_insec
        frame_indices = [int(i * self.raw_fps / fps) for i in sampled_frame_indexes]
        frame_indices, frames = self.read_frame_batch(video_path=self.video_path, frame_indices=frame_indices)

        return sampled_frames_insec.tolist(), frames
    def update_frame_distribution(self, sampled_frame_indices, confidence_maps, detected_objects_maps):

        confidence_map = confidence_maps[0]  # 只有一张图像
        detected_objects_map = detected_objects_maps[0]  # 只有一张图像

        # 将检测结果映射回对应的帧
        grid_rows, grid_cols = self.image_grid_shape

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

        # Priority searching top 0.25

        # 获取 top 25% 的分数和对应的 frame_idx
        self.update_top_25_with_window(frame_confidences, sampled_frame_indices)

        # map score_distribution to possibility
        self.P = self.spline_keyframe_distribution(self.non_visiting_frames, self.score_distribution, len(self.score_distribution))

        self.store_score_distribution()
        return frame_confidences, frame_detected_objects
    
    def store_score_distribution(self):
        """
        对 score_distribution 进行深度拷贝并返回拷贝后的副本。
        
        返回:
        - dict: 深度拷贝后的 score_distribution 副本。

        """
        self.P_hository.append(copy.deepcopy(self.P).tolist())
        return None
    def spline_keyframe_distribution(self, non_visiting_frames, score_distribution, video_length):
        """
        根据样条拟合和概率归一化计算帧分布。

        Args:
            non_visiting_frames (dict): 一个字典，键是帧索引，值表示是否已经访问 (0 表示已访问)。
            score_distribution (dict): 一个字典，键是帧索引，值是对应的帧分数。
            video_length (int): 视频的总帧数。

        Returns:
            np.ndarray: 归一化的概率分布 p_distribution。
        """
        # 提取已访问的帧索引和对应的分数
        frame_indices = np.array([idx for idx, visited in enumerate(non_visiting_frames) if visited == 0])
        observed_scores = np.array([score_distribution[idx] for idx in frame_indices])

        # 如果没有观测帧，则返回均匀分布
        if len(frame_indices) == 0:
            return np.ones(video_length) / video_length
        # Step 1: 使用样条拟合已观测的分数分布
        spline = UnivariateSpline(frame_indices, observed_scores, s=0.5)

        # Step 2: 预测全帧范围的分布
        all_frames = np.arange(1, video_length + 1)
        spline_scores = spline(all_frames)

        # Step 3: 应用 sigmoid 函数并归一化
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # 确保分数不低于 1/L
        adjusted_scores = np.maximum(1 / video_length, spline_scores)

        # 计算概率分布
        p_distribution = sigmoid(adjusted_scores)

        # 归一化概率分布
        p_distribution /= p_distribution.sum()

        return p_distribution

    def update_top_25_with_window(self, frame_confidences, sampled_frame_indices, window_size=5):
        """
        更新分数分布，使得前 25% 高分帧的前后 window_size 帧的分数设置为当前帧分数。
        
        Args:
            frame_confidences (list): 帧的置信度分数列表。
            sampled_frame_indices (list): 对应的帧索引列表。
            score_distribution (dict): 全局的帧分数分布字典。
            window_size (int): 前后扩展的帧窗口大小。
        """
        # 计算前 25% 的分数阈值
        top_25_threshold = np.percentile(frame_confidences, 75)

        # 找出前 25% 的帧索引
        top_25_indices = [
            frame_idx for frame_idx, confidence in zip(sampled_frame_indices, frame_confidences)
            if confidence >= top_25_threshold
        ]

        # 设置 window_size 前后帧的分数
        for frame_idx in top_25_indices:
            for offset in range(-window_size, window_size + 1):
                neighbor_idx = frame_idx + offset
                if 0 <= neighbor_idx < len(self.score_distribution):  # 确保索引合法
                    self.score_distribution[neighbor_idx] = max(self.score_distribution[neighbor_idx], self.score_distribution[frame_idx])
    def verify_and_remove_target(
        self,
        frame_sec: int,
        detected_objects: List[str],
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
        for target in list(self.remaining_targets):
            if target in detected_objects:
                frame_idx = int(frame_sec / self.fps * self.raw_fps)
                # Read the individual frame
                _, frame = self.read_frame_batch(self.video_path, [frame_idx])
                frame = frame[0]  # Extract the frame from the list

                # Perform detection on the individual frame
                single_confidence_maps, single_detected_objects_maps = self.score_image_grids(
                    [frame], (1, 1)
                )
                single_confidence = single_confidence_maps.tolist()[0][0][0]
                single_detected_objects = single_detected_objects_maps[0][0]
                self.score_distribution[frame_sec] = single_confidence
                # Check if target object confidence exceeds the threshold
                if target in single_detected_objects and single_confidence > confidence_threshold:
                    self.remaining_targets.remove(target)
                    
                    print(f"Found target '{target}' in frame {frame_idx}, score {single_confidence:.2f}")
                    return True
        
        return False

    def search(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        执行KeyFrameSearch过程，返回所有确认的帧和对应的时间戳。
        """
        K = self.search_nframes  # 最终需要的关键帧数量
        # 初始化分布分数（均匀分布）
        self.P = np.ones(self.total_frame_num) * self.confidence_threshold*0.5
        self.P_hository = []
        self.non_visiting_frames = np.ones(self.total_frame_num)
        


        while self.remaining_targets and self.search_budget > 0:
            grid_cols, grid_rows = self.image_grid_shape  # grid size为 n x n
            num_frames_in_grid = grid_cols * grid_rows
            sampled_frame_indices, frames = self.sample_frames(num_frames_in_grid)
            self.search_budget -= num_frames_in_grid
            # 提取帧并生成图像网格 
            
            grid_image = self.create_image_grid(frames, *self.image_grid_shape)
            print(f"采样帧索引: {sampled_frame_indices}")

            # 对图像网格执行对象检测并评分
            confidence_maps, detected_objects_maps = self.score_image_grids(
                images=[grid_image],
                image_grids=self.image_grid_shape
            )

            cell_detected_confidences, cell_detected_objects = self.update_frame_distribution(sampled_frame_indices, confidence_maps, detected_objects_maps)

            # 检查是否检测到目标对象
            for frame_sec, detected_objects in zip(sampled_frame_indices, cell_detected_objects):
                for target in list(self.remaining_targets):
                    if target in detected_objects:
                        self.verify_and_remove_target(
                            frame_sec=frame_sec,
                            detected_objects=detected_objects,
                            confidence_threshold=self.confidence_threshold,
                        )

 
        top_k_indices = np.argsort(self.score_distribution)[-K:][::-1]  # 取最高K分数的帧
        top_k_frames = []
        time_stamps = []

        return top_k_frames, time_stamps


if __name__ == "__main__":
    # 定义视频路径和目标对象
    video_path = "/home/yejinhui/Projects/VisualSearch/38737402-19bd-4689-9e74-3af391b15feb.mp4"
    query = "what is the color of the couch?"
    target_objects = ["couch"]  # 例如，目标对象
    cue_objects = ["TV", "chair"] 
    # 创建 VideoSearcher 实例
    searcher = VideoSearcher(
        video_path=video_path,
        target_objects=target_objects,
        cue_objects=cue_objects,
        search_nframes=8,
        image_grid_shape=(4, 4),
        confidence_threshold=0.6
    )

    searcher.initialize_yolo()

    # 执行搜索
    all_frames, time_stamps = searcher.search()

    # 处理结果
    print(f"共找到 {len(all_frames)} 帧，时间戳列表: {time_stamps}")

    # 绘制分数分布图
    searcher.plot_score_distribution(save_path='/home/yejinhui/Projects/VisualSearch/llava/VideoFrameSearch/score_distribution.png')

sourece=/hpc2hdd/home/jye624/Dataset/ego4d_data/ego4d_data/v1/256p

rsync -av --progress --inplace --rsh=rsh /hpc2hdd/home/jye624/Dataset/ego4d_data/ego4d_rule_trajs/Search_Trajectory rsync://paperspace@184.105.3.82:/home/paperspace


rsync -av --progress --partial /hpc2hdd/home/jye624/Dataset/ego4d_data/ego4d_data/v1/256p paperspace@184.105.3.82:/home/paperspace
