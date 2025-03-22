
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

import torch.nn as nn

class HeuristicInterface:
    def __init__(self, heuristic_type: str = "owl-vit", **kwargs):
        """
        Initialize the YOLO-World model with the given configuration and checkpoint.

        Args:
            config_path (str): Path to the model configuration file.
            checkpoint_path (str): Path to the model checkpoint.
            device (str): Device to run the model on (e.g., 'cuda:0', 'cpu').
        """

class YoloWorldInterface(HeuristicInterface):
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
    
    def set_BBoxAnnotator(self):
        self.BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
        # MASK_ANNOTATOR = sv.MaskAnnotator()
        self.LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                        text_scale=0.5,
                                        text_thickness=1,
                                        smart_position=True,
                                        color=ColorPalette.LEGACY)


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


class OWLInterface(HeuristicInterface):
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        self.processor, self.model = self.load_model_and_tokenizer(model_name_or_path)
        self.texts = ["couch", "table", "woman"] #TODO MV
        self.device = device
        self.model = self.model.to(self.device)

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
    
    def inference_detector(self, images, **kwargs):
        batch_images = np.array(images)
        inputs = self.processor(text= self.texts, images=batch_images[0], return_tensors="pt").to(self.device)
        height, width = batch_images[0].shape[:2]
        detections_inbatch = []
        with torch.no_grad():
            # Run model inference 
            outputs = self.forward_model(inputs)

            target_sizes = torch.tensor([[height, width] for i in batch_images]) 
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=0.05)
        for result in results:
            detections = sv.Detections.from_transformers(transformers_results=result)
            detections_inbatch.append(detections)

        check = True
        if check:
            # save first image for checking
            bounding_box_annotator = sv.BoxAnnotator()
            annotated_image = bounding_box_annotator.annotate(batch_images[0] , detections_inbatch[0])

            output_image = Image.fromarray(annotated_image[:, :, ::-1])
            output_image.save("./annotated_image.png")
            self.detections_inbatch = detections_inbatch
        return detections_inbatch

    def bbox_visualization(self, images, detections_inbatch):
        bounding_box_annotator = sv.BoxAnnotator()
        annotated_images = []
        for image, detections in zip(images,detections_inbatch):
            annotated_image = bounding_box_annotator.annotate(image, detections)
            # output_image = Image.fromarray(annotated_image[:, :, ::-1])
            annotated_images.append(annotated_image)
            
        return annotated_images
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
        