import os
import sys
import cv2
import logging
import numpy as np
from typing import List, Tuple
from TStar.interface_grounding import TStarUniversalGrounder
from TStar.interface_heuristic import YoloWorldInterface, OWLInterface, HeuristicInterface
from TStar.interface_searcher import TStarSearcher
from TStar.utilites import save_as_gif


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class TStarFramework:
    """
    Main class for performing object-based frame search and question-answering in a video.
    """

    def __init__(
        self,
        video_path: str,
        heuristic: HeuristicInterface,
        grounder: TStarUniversalGrounder,
        question: str,
        options: str,
        search_nframes: int = 8,
        grid_rows: int = 4,
        grid_cols: int = 4,
        output_dir: str = './output',
        confidence_threshold: float = 0.6,
        search_budget: int = 1000
    ):
        self.video_path = video_path
        self.grounder = grounder
        self.heuristic = heuristic
        self.question = question
        self.options = options
        self.search_nframes = search_nframes
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.output_dir = os.path.join(output_dir, os.path.basename(video_path).split('.')[0], question[:-1])
        self.confidence_threshold = confidence_threshold
        self.search_budget = search_budget
        self._create_output_dir()

        self.results = {} # to store search results, e.g., grounding, frames

    def _create_output_dir(self):
        """
        Ensure that the output directory exists.
        """
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        """
        Run the TStar framework to search for objects and answer questions.
        """
        target_objects, cue_objects = self.get_grounded_objects()
        video_searcher = self.initialize_videoSearcher(target_objects, cue_objects)
        all_frames, time_stamps = self.perform_search(video_searcher, visualization=True)
        answer = self.perform_qa(all_frames)
        logger.info(f"Answer: {answer}")
        
        return {
            "Grounding Objects": {'target_objects': target_objects, 'cue_objects': cue_objects},
            "Frame Timestamps": time_stamps,
            "Answer": answer
        }

    def get_grounded_objects(self) -> Tuple[List[str], List[str]]:
        """
        Use Grounder to obtain target and cue objects.
        """
        target_objects, cue_objects = self.grounder.inference_query_grounding(
            video_path=self.video_path,
            question=self.question,
            options=self.options
        )
        self.results["Grounding Objects"] = {"target_objects": target_objects, "cue_objects":cue_objects}
        logger.info(f"Target objects: {target_objects}")
        logger.info(f"Cue objects: {cue_objects}")
        return target_objects, cue_objects



    def initialize_videoSearcher(self, target_objects: List[str], cue_objects: List[str]) -> TStarSearcher:
        """
        Initialize and configure the TStarSearcher with the given objects.
        """
        videoSearcher =  TStarSearcher(
            video_path=self.video_path,
            target_objects=target_objects,
            cue_objects=cue_objects,
            search_nframes=self.search_nframes,
            image_grid_shape=(self.grid_rows, self.grid_cols),
            output_dir=self.output_dir,
            confidence_threshold=self.confidence_threshold,
            search_budget=self.search_budget,
            heuristic=self.heuristic
        )

        return videoSearcher

    def perform_search(self, video_searcher: TStarSearcher, visualization: bool = False) -> Tuple[List[np.ndarray], List[float]]:
        """
        Perform the search for relevant frames and their timestamps.
        """
        if visualization:
            all_frames, time_stamps = video_searcher.search()
            self._save_frames(all_frames, time_stamps)
            self._save_searching_iterations(video_searcher)
            self._plot_and_save_scores(video_searcher)
        else:
            all_frames, time_stamps = video_searcher.search()
        
        logger.info(f"Found {len(all_frames)} frames, timestamps: {time_stamps}")
        return all_frames, time_stamps

    def perform_qa(self, frames: List[np.ndarray]) -> str:
        """
        Perform question answering on the retrieved frames.
        """
        return self.grounder.inference_qa(
            frames=frames,
            question=self.question,
            options=self.options
        )

    def _save_frames(self, frames: List[np.ndarray], timestamps: List[float]):
        """
        Save the relevant frames as image files.
        """
        frame_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)

        for idx, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            frame_path = os.path.join(frame_dir, f"frame_{idx}_at_{timestamp:.2f}s.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved frame to {frame_path}")

    def _save_searching_iterations(self, video_searcher: TStarSearcher):
        """
        Save the frames and their annotations from search iterations.
        """
        image_grid_iters = video_searcher.image_grid_iters
        detect_annotot_iters = video_searcher.detect_annotot_iters
        
        for b in range(len(image_grid_iters[0])):
            images = [image_grid_iter[b] for image_grid_iter in image_grid_iters]
            anno_images = [detect_annotot_iter[b] for detect_annotot_iter in detect_annotot_iters]
            output_video_path = os.path.join(self.output_dir, f"search_iterations.gif")
            save_as_gif(images=anno_images, output_gif_path=output_video_path)
            logger.info(f"Saved search iterations GIF to {output_video_path}")

    def _plot_and_save_scores(self, video_searcher: TStarSearcher):
        """
        Plot and save the score distribution from the search process.
        """
        plot_path = os.path.join(self.output_dir, "score_distribution.png")
        video_searcher.plot_score_distribution(save_path=plot_path)
        logger.info(f"Score distribution plot saved to {plot_path}")


def initialize_heuristic(heuristic_type: str = "owl-vit") -> HeuristicInterface:
    """
    Initialize the object detection model based on the selected heuristic type.
    """
    if heuristic_type == 'owl-vit':
        model_name = "google/owlvit-base-patch32"
        owl_interface = OWLInterface(model_name_or_path=model_name)
        logger.info("OWLInterface initialized successfully.")
        return owl_interface
    elif heuristic_type == 'yolo-World':
        config_path = "./YOLO-World/configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
        checkpoint_path = "./pretrained/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
        yolo_interface = YoloWorldInterface(config_path=config_path, checkpoint_path=checkpoint_path)
        logger.info("YoloWorldInterface initialized successfully.")
        return yolo_interface
    else:
        raise NotImplementedError(f"Heuristic type '{heuristic_type}' is not implemented.")


def run_tstar(
    video_path: str,
    question: str,
    options: str,
    grounder: str = "gpt-4o",
    heuristic: str = "owl-vit",
    search_nframes: int = 8,
    grid_rows: int = 4,
    grid_cols: int = 4,
    confidence_threshold: float = 0.6,
    search_budget: float = 0.5,
    output_dir: str = './output'
):
    """
    Execute the TStar video frame search and question-answering process.
    """
    grounder = TStarUniversalGrounder(model_name=grounder)
    heuristic = initialize_heuristic(heuristic)

    TStarQA = TStarFramework(
        video_path=video_path,
        grounder=grounder,
        heuristic=heuristic,
        question=question,
        options=options,
        search_nframes=search_nframes,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        output_dir=output_dir,
        confidence_threshold=confidence_threshold,
        search_budget=search_budget
    )

    return TStarQA.run()

if __name__ == "__main__":
    # Example call to run_tstar with the appropriate arguments.
    video_path = "/data/guoweiyu/LV-Haystack/Datasets/ego4d_data/ego4d_data/v1/256p/38737402-19bd-4689-9e74-3af391b15feb.mp4"
    question =  "What is the color of the couch?"
    options = "A) Red, B) Blue, C) Green, D) Yellow"

    run_tstar(video_path, question, options)
