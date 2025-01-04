import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from decord import VideoReader, cpu

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


class VideoSearcher:
    def __init__(
        self,
        video_path: str,
        target_objects: List[str],
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
        self.cue_object = cue_object
        self.search_nframes = search_nframes
        self.image_grid_shape = image_grid_shape
        self.output_dir = output_dir
        self.profix = profix
        self.confidence_threshold = confidence_threshold
        self.object2weight = object2weight if object2weight else {}
        self.mrr = 10

        # 初始化视频读取器
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frame_num / self.fps
        self.total_frame_num = int(self.total_frame_num / self.fps) # 1 fps
        # 初始化目标对象
        self.remaining_targets = set(target_objects)
        # 初始化搜索预算
        self.search_budget = min(1000, self.total_frame_num)
        # 分数分布记录（以帧为单位）
        self.score_distribution = np.zeros(self.total_frame_num)


    def initialize_yolo(self, heuristicer=None):
        # 这个模型用于评价图片不同部分对
        # TODO: 初始化YOLO模型
        # self.detect_model = YourYOLOModel()
        # self.detect_objects = ["person", "car", "bicycle"]  # 示例
        # self.test_pipeline = ...  # 定义您的测试管道
        if heuristicer == None:
            # 初始化YOLO模型
            self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.detector.conf = self.confidence_threshold  # 设置置信度阈值
            return
        
        self.object2weight = heuristicer.object2weight
        self.heuristicer = heuristicer


    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """根据帧索引获取帧图像"""
        if frame_index < 0 or frame_index >= self.total_frame_num:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
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
    def score_image_grids(self, images: List[np.ndarray], output_dir: Optional[str], profix: str, image_grids: Tuple[int, int]) -> Tuple[np.ndarray, List[List[str]]]:
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
            0.05,
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
        time_axis = np.arange(0, self.duration, 1 / self.fps)

        plt.figure(figsize=(12, 6))

        # 遍历 P_hository 中的每个历史分布，并绘制曲线
        for i, history in enumerate(self.P_hository):
            if (i + 1) % 3 != 0:
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
    
    def sample_frames(self, num_samples: int) -> List[int]:
        """根据当前分布 P 采样帧索引"""
        if num_samples > self.total_frame_num:
            num_samples = self.total_frame_num

        _P = self.P / self.P.sum()  # 重新归一化
        sampled_frames_insec = np.random.choice(
            self.total_frame_num,
            size=num_samples,
            replace=False,
            p=_P
        )
        sampled_frame_indexes = sampled_frames_insec * self.fps
        return sampled_frame_indexes.tolist()

    def search(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        执行KeyFrameSearch过程，返回所有确认的帧和对应的时间戳。
        """
        K = self.search_nframes  # 最终需要的关键帧数量
        # 初始化分布分数（均匀分布）
        self.P = np.ones(self.total_frame_num) * self.confidence_threshold*0.5
        self.P_hository = []

        while self.remaining_targets and self.search_budget > 0:
            n, _ = self.image_grid_shape  # grid size为 n x n
            num_samples = n * n
            # if self.search_budget < num_samples:
            #     num_samples = self.search_budget
            sampled_frame_indices = self.sample_frames(num_samples)
            self.search_budget -= num_samples

            # 提取帧并生成图像网格 
            frames = self.read_frame_batch(video_path=self.video_path, frame_indices=sampled_frame_indices)
            grid_image = self.create_image_grid(frames, *self.image_grid_shape)
            print(f"采样帧索引: {sampled_frame_indices}")

            # 对图像网格执行对象检测并评分
            confidence_maps, detected_objects_maps = self.score_image_grids(
                images=[grid_image],
                output_dir=self.output_dir,
                profix=self.profix,
                image_grids=self.image_grid_shape
            )
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
                self.P[frame_idx] = confidence  # 更新 P
                self.score_distribution[frame_idx] = confidence

            # 检查是否检测到目标对象
            for frame_idx, detected_objects in zip(sampled_frame_indices, frame_detected_objects):
                for target in list(self.remaining_targets):
                    if target in detected_objects and self.score_distribution[frame_idx] > self.confidence_threshold:
                        self.remaining_targets.remove(target)
                        print(f"确认目标对象 '{target}' 在帧 {frame_idx}，分数 {self.score_distribution[frame_idx]:.2f}")

            # 内部循环：如果未检测到所有目标对象且有剩余预算，进行进一步搜索
            while self.remaining_targets and self.search_budget > 0:
                # 识别低置信度的帧，保留高置信度的25%
                num_high_conf = max(1, len(frame_confidences) // 4)
                high_conf_indices = np.argsort(frame_confidences)[-num_high_conf:]
                retained_frame_indices = [sampled_frame_indices[i] for i in high_conf_indices][:int(0.25*num_samples)]
                retained_confidences = [frame_confidences[i] for i in high_conf_indices]

                # 重新采样更多帧，集中在保留的区域
                additional_samples = int(num_samples - len(retained_frame_indices))

                additional_frame_indices = self.sample_frames(additional_samples)
                self.search_budget -= additional_samples

                # 提取帧并生成图像网格
                additional_frame_indices = additional_frame_indices+retained_frame_indices
                additional_frame_indices.sort()
                additional_frames = [self.get_frame(idx) for idx in additional_frame_indices]
                # 如果某些帧为空，使用空白图像代替
                additional_frames = [frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8) for frame in additional_frames]
                additional_grid_image = self.create_image_grid(additional_frames, *self.image_grid_shape)
                print(f"进一步采样帧索引: {additional_frame_indices}")

                # 对图像网格执行对象检测并评分
                additional_confidence_maps, additional_detected_objects_maps = self.score_image_grids(
                    images=[additional_grid_image],
                    output_dir=self.output_dir,
                    profix=self.profix,
                    image_grids=self.image_grid_shape
                )
                additional_confidence_map = additional_confidence_maps[0]  # 只有一张图像
                additional_detected_objects_map = additional_detected_objects_maps[0]  # 只有一张图像

                # 将检测结果映射回对应的帧
                additional_frame_confidences = []
                additional_frame_detected_objects = []
                for idx, frame_idx in enumerate(additional_frame_indices):
                    # 计算对应的网格单元
                    row = idx // grid_cols
                    col = idx % grid_cols
                    confidence = additional_confidence_map[row, col]
                    detected_objects = additional_detected_objects_map[idx]
                    additional_frame_confidences.append(confidence)
                    additional_frame_detected_objects.append(detected_objects)

                # 更新分布 P 和分数分布
                for frame_idx, confidence in zip(additional_frame_indices, additional_frame_confidences):
                    self.P[frame_idx] = confidence  # 更新 P
                    self.score_distribution[frame_idx] = confidence

                # 检查是否检测到目标对象
                for frame_idx, detected_objects in zip(additional_frame_indices, additional_frame_detected_objects):
                    for target in list(self.remaining_targets):
                        if target in detected_objects and self.score_distribution[frame_idx] > self.confidence_threshold:
                            self.remaining_targets.remove(target)
                            print(f"确认目标对象 '{target}' 在帧 {frame_idx}，分数 {self.score_distribution[frame_idx]:.2f}")

                # 更新原始帧列表以包含新采样的帧
                sampled_frame_indices += additional_frame_indices
                frame_confidences += additional_frame_confidences
                frame_detected_objects += additional_frame_detected_objects
                self.store_score_distribution()
            # 搜索结束，采样前K关键帧
            top_k_indices = np.argsort(self.score_distribution)[-K:][::-1]  # 取最高K分数的帧
            top_k_frames = []
            time_stamps = []
            for frame_idx in top_k_indices:
                frame = self.get_frame(frame_idx)
                if frame is not None:
                    top_k_frames.append(frame)
                    time_stamps.append(frame_idx / self.fps)
                    print(f"选定关键帧: {frame_idx}，时间戳: {frame_idx / self.fps:.2f} 秒")
    
            return top_k_frames, time_stamps


if __name__ == "__main__":
    # 定义视频路径和目标对象
    video_path = "/home/yejinhui/Projects/VisualSearch/38737402-19bd-4689-9e74-3af391b15feb.mp4"
    target_objects = ["person", "handbag", "bicycle"]  # 例如，目标对象

    # 创建 VideoSearcher 实例
    searcher = VideoSearcher(
        video_path=video_path,
        target_objects=target_objects,
        search_nframes=8,
        image_grid_shape=(8, 8),
        confidence_threshold=0.9
    )

    # 执行搜索
    all_frames, time_stamps = searcher.search()

    # 处理结果
    print(f"共找到 {len(all_frames)} 帧，时间戳列表: {time_stamps}")

    # 绘制分数分布图
    searcher.plot_score_distribution(save_path='/home/yejinhui/Projects/VisualSearch/llava/VideoFrameSearch/score_distribution.png')

    # 可选：保存或显示结果
    # for idx, frame in enumerate(all_frames):
    #     cv2.imwrite(f"frame_{idx}.jpg", frame)
