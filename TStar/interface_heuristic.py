
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

import torch
from typing import List
import supervision as sv  # 确保已安装 Supervision 库
import os.path as osp



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


class TStarUniversalHeuristic:
    def __init__(self, heuristic: str = None, device: str = "cuda:0"):
        """
        Initialize the YOLO-World model with the given configuration and checkpoint.

        Args:
            config_path (str): Path to the model configuration file.
            checkpoint_path (str): Path to the model checkpoint.
            device (str): Device to run the model on (e.g., 'cuda:0', 'cpu').
        """


class YoloInterface(TStarUniversalHeuristic):
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




from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import numpy as np

class OWLInterface(TStarUniversalHeuristic):
    
    def __init__(self, config_path: str, checkpoint_path: None, device: str = "cuda:0"):
        
        self.processor, self.model = self.load_model_and_tokenizer(config_path)
        self.device = device
        self.model = self.model.to(self.device)
        self.texts = ["couch", "table", "woman"]

    def load_model_and_tokenizer(self, model_name):
        processor = OwlViTProcessor.from_pretrained(model_name)
        model = OwlViTForObjectDetection.from_pretrained(model_name)
        return processor, model

    def forward_model(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    def inference(self, image_path, use_amp: bool = False):
        with Image.open(image_path).convert("RGB") as image:
            width, height = image.size
        inputs = self.processor(text=self.texts, images=image, return_tensors="pt").to(self.device)

        # Run model inference
        outputs = self.forward_model(inputs)

        # Post-process outputs
        target_size = torch.tensor([[height, width]])
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_size)[0]
        detections = sv.Detections.from_transformers(transformers_results=results)
        return detections
    
    def inference_detector(self, image, use_amp: bool = False):
        batch_images = []
        for i in range(4):
            for j in range(4):
                # Extract the smaller image from the grid
                small_image = image[i*120:(i+1)*120, j*160:(j+1)*160]
                batch_images.append(small_image)
        batch_images = np.array(batch_images)
        images = [Image.fromarray(np.uint8(img)).convert("RGB") for img in batch_images]
        inputs = self.processor(text= self.texts, images=batch_images[0], return_tensors="pt").to(self.device)
        height, width = batch_images[0].shape[:2]
        detections_inbatch = []
        with torch.no_grad():
            # Run model inference
            outputs = self.forward_model(inputs)
            for output in outputs:
                # Post-process outputs
                target_size = torch.tensor([[height, width]])
                results = self.processor.post_process_grounded_object_detection(
                    outputs=outputs, target_sizes=target_size)[0]
                detections = sv.Detections.from_transformers(transformers_results=results)
                detections_inbatch.append(detections)
        # save image 
        bounding_box_annotator = sv.BoxAnnotator()
        annotated_image = bounding_box_annotator.annotate(batch_images[0] , detections_inbatch[0])

        output_image = Image.fromarray(annotated_image[:, :, ::-1])
        output_image.save("output/annotated_image.png")
        return detections_inbatch

    def bbox_visualization(self, image_path, output_path):
        image =  Image.open(image_path).convert("RGB")
        # Annotate image
        detections = self.inference(image_path)
        bounding_box_annotator = sv.BoxAnnotator()
        annotated_image = np.array(image)
        annotated_image = bounding_box_annotator.annotate(annotated_image, detections)

        output_image = Image.fromarray(annotated_image[:, :, ::-1])
        output_image.save(output_path)

    def reparameterize_object_list(self, target_objects: List[str], cue_objects: List[str]):
        """
        Reparameterize the detect object list to be used by the OWL model.

        Args:
            target_objects (List[str]): List of target object names.
            cue_objects (List[str]): List of cue object names.
        """
        # Combine target objects and cue objects into the final text format
        combined_texts = target_objects + cue_objects

        # Format the text prompts for the YOLO model
        self.texts = [[obj.strip()] for obj in combined_texts] + [[' ']]

        # Reparameterize the YOLO model with the provided text prompts
        # self.model.reparameterize(self.texts)
        


  
        #

