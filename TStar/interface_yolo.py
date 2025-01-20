
import os
import cv2
import os.path as osp
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
# from mmengine.runner.amp import autocast
from torch.amp import autocast
import torch
import supervision as sv
from typing import Dict, Optional, Sequence, List

import supervision as sv
from supervision.draw.color import Color, ColorPalette

class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


class YoloInterface:
    def __init__(self):
        """
        Initialize the YOLO-World model with the given configuration and checkpoint.

        Args:
        """
        
     
        pass
    def set_BBoxAnnotator(self):
        self.BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
        # MASK_ANNOTATOR = sv.MaskAnnotator()
        self.LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                        text_scale=0.5,
                                        text_thickness=1,
                                        smart_position=True,
                                        color=ColorPalette.LEGACY)

class YoloWorldInterface(YoloInterface):
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda:0"):
        """
        Initialize the YOLO-World model with the given configuration and checkpoint.

        Args:
            config_path (str): Path to the model configuration file.
            checkpoint_path (str): Path to the model checkpoint.
            device (str): Device to run the model on (e.g., 'cuda:0', 'cpu').
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device

        # Load configuration
        cfg = Config.fromfile(config_path)
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config_path))[0])
        cfg.load_from = checkpoint_path

        # Initialize the model
        self.model = init_detector(cfg, checkpoint=checkpoint_path, device=device)
        self.set_BBoxAnnotator()

        # Initialize the test pipeline
        # build test pipeline
        self.model.cfg.test_dataloader.dataset.pipeline[
            0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(self.model.cfg.test_dataloader.dataset.pipeline)

        

    def reparameterize_object_list(self, target_objects: List[str], cue_objects: List[str]):
        """
        Reparameterize the detect object list to be used by the YOLO model.

        Args:
            target_objects (List[str]): List of target object names.
            cue_objects (List[str]): List of cue object names.
        """
        # Combine target objects and cue objects into the final text format
        combined_texts = target_objects + cue_objects

        # Format the text prompts for the YOLO model
        self.texts = [[obj.strip()] for obj in combined_texts] + [[' ']]

        # Reparameterize the YOLO model with the provided text prompts
        self.model.reparameterize(self.texts)


    def inference(self, image: str, max_dets: int = 100, score_threshold: float = 0.3, use_amp: bool = False):
        """
        Run inference on a single image.

        Args:
            image (str): Path to the image.
            max_dets (int): Maximum number of detections to keep.
            score_threshold (float): Score threshold for filtering detections.
            use_amp (bool): Whether to use mixed precision for inference.

        Returns:
            sv.Detections: Detection results.
        """
        # Prepare data for inference
        data_info = dict(img_id=0, img_path=image, texts=self.texts)
        data_info = self.test_pipeline(data_info)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                          data_samples=[data_info['data_samples']])

        # Run inference
        with autocast(enabled=use_amp), torch.no_grad():
            output = self.model.test_step(data_batch)[0]
            pred_instances = output.pred_instances
            pred_instances = pred_instances[pred_instances.scores.float() > score_threshold]

        if len(pred_instances.scores) > max_dets:
            indices = pred_instances.scores.float().topk(max_dets)[1]
            pred_instances = pred_instances[indices]

        pred_instances = pred_instances.cpu().numpy()

        # Process detections
        detections = sv.Detections(
            xyxy=pred_instances['bboxes'],
            class_id=pred_instances['labels'],
            confidence=pred_instances['scores'],
            mask=pred_instances.get('masks', None)
        )
        return detections
    
    def inference_detector(self, images, max_dets=50, score_threshold=0.2, use_amp: bool = False):
        data_info = dict(img_id=0, img=images[0], texts=self.texts) #TBD for batch searching
        data_info = self.test_pipeline(data_info)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                        data_samples=[data_info['data_samples']])
        detections_inbatch = []
        with torch.no_grad():
            outputs = self.model.test_step(data_batch)
            # cover to searcher interface format
            
            for output in outputs:
                pred_instances = output.pred_instances
                pred_instances = pred_instances[pred_instances.scores.float() >
                                                score_threshold]
                if len(pred_instances.scores) > max_dets:
                    indices = pred_instances.scores.float().topk(max_dets)[1]
                    pred_instances = pred_instances[indices]

                output.pred_instances = pred_instances

                if 'masks' in pred_instances:
                    masks = pred_instances['masks']
                else:
                    masks = None
                pred_instances = pred_instances.cpu().numpy()
                detections = sv.Detections(xyxy=pred_instances['bboxes'],
                    class_id=pred_instances['labels'],
                    confidence=pred_instances['scores'],
                    mask=masks)
                detections_inbatch.append(detections)
        self.detect_outputs_raw = outputs
        self.detections_inbatch = detections_inbatch
        return detections_inbatch

    def bbox_visualization(self, images, detections_inbatch):
        anno_images = []
        # detections_inbatch = self.detections_inbatch
        for b, detections in enumerate(detections_inbatch):
            texts = self.texts
            labels = [
                f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
                zip(detections.class_id, detections.confidence)
            ]

        
            index = len(detections_inbatch) -1 
            image = images[index]
            anno_image = image.copy()
  
    
            anno_image = self.BOUNDING_BOX_ANNOTATOR.annotate(anno_image, detections)
            anno_image = self.LABEL_ANNOTATOR.annotate(anno_image, detections, labels=labels)
            anno_images.append(anno_image)
        
        return anno_images



import torch
from typing import List
import supervision as sv  # 确保已安装 Supervision 库
import os.path as osp

class YoloV5Interface(YoloInterface):
    def __init__(self,config_path="ultralytics/yolov5", checkpoint_path: str = 'yolov5s', device: str = 'cuda:0'):
        """
        初始化 YOLOv5 模型。

        Args:
            model_name (str): YOLOv5 模型变体名称（如 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'）。
            device (str): 运行模型的设备（如 'cuda:0', 'cpu'）。
        """
        self.device = device
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        
        self.model.to(self.device)
        self.model.eval()
        self.target_classes = None  # 用于存储目标类别列表

        self.texts = None
        self.test_pipeline = None

    def reparameterize_object_list(self, target_objects: List[str], cue_objects: List[str]):
        """
        重新参数化检测对象列表，以便在推理时使用。

        Args:
            target_objects (List[str]): 目标对象名称列表。
            cue_objects (List[str]): 线索对象名称列表。
        """
        # 合并目标对象和线索对象
        combined_objects = target_objects + cue_objects
        self.target_classes = combined_objects

    def inference(self, images: str, max_dets: int = 100, score_threshold: float = 0.3, use_amp: bool = False):
        """
        对单张图像运行推理。

        Args:
            image (str): 图像路径。
            max_dets (int): 保留的最大检测数量。
            score_threshold (float): 过滤检测的分数阈值。
            use_amp (bool): 是否使用混合精度进行推理。

        Returns:
            sv.Detections: 检测结果。
        """
        results = self.model(images, size=640)  # 可以根据需要调整输入尺寸

        # 提取检测结果（假设批量大小为 1）
        detections_batch = results.pred  # B tensors of shape (N, 6) [x1, y1, x2, y2, confidence, class]

        # 应用分数阈值
        # 用于存储每个批次过滤后的检测结果
        filtered_detections = []

        for detections in detections_batch:
            # 应用分数阈值，过滤掉 confidence <= score_threshold 的检测
            detections = detections[detections[:, 4] > score_threshold]
            # 如果设置了 topk，截取前 topk 个检测
            if len(detections) > max_dets:
                detections = detections[:max_dets]
            # 如果设置了目标类别，过滤检测结果
            if self.target_classes is not None:
                # 获取所有类别名称
                class_names = self.model.names
                # 获取目标类别的类别ID
                target_class_ids = [i for i, name in class_names.items() if name in self.target_classes]

                
                # 过滤检测结果
                detections = detections[[cls in target_class_ids for cls in detections[:, 5]]]
                # 转换为 Supervision 库的 Detections 对象
                detections = sv.Detections(
                    xyxy=detections[:, :4].cpu().numpy(),
                    confidence=detections[:, 4].cpu().numpy(),
                    class_id=detections[:, 5].cpu().numpy().astype(int)
                )
                

                filtered_detections.append(detections)

        return filtered_detections


