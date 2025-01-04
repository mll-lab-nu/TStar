# VL-Haystack

## Getting Started

### Installation

```bash
## Follow docs/installation to implemet Grounding (e.g., LLaVA) and Searching (e.g., YOLO) Function
###  Install Query Grounder Interface(LLaVA or GPT-API)
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
### Install Image Grid Scorer Interface e.g., YOLO-WORLD
git clone --recursive https://github.com/AILab-CVC/YOLO-World.git
```

Structure:
LV-Haystack.github.io
    | LLaVA-NeXT/ % Grounding and QA
    | YOLO-World/ % Heuristic
    | TStar/      %  Searching
      - interface_llm.py
      - interface_yolo.py
      - interface_searcher.py % main searching for T* processing
      - TStarFramework.py     % main function for T*
      - val_LV_Haystack.py    % inference on LV-Haystack
    | readme.md  


## Run VideoSearching Dome

```bash
#  Install Query Grounder (LLaVA or GPT-API)
## e.g., LLaVA 
python TStar/TStarFramework.py \
    --llava_model_path lmms-lab/llava-onevision-qwen2-7b-ov \
    --video_path /path/to/video.mp4 \
    --question "What is the color of the couch?" \
    --options "A) Red, B) Blue, C) Green, D) Yellow" \
    --yolo_config_path ./YOLOWorld/configs/yolo_config.py \
    --yolo_checkpoint_path ./pretrained/yolo_checkpoint.pth \
    --search_nframes 10 \
    --image_grid_shape 4 4 \
    --confidence_threshold 0.6 \
    --device cuda:0
```


## Test VL-Bench

```bash
#  Install Query Grounder (LLaVA or GPT-API)
## e.g., LLaVA 
python TStar.py \
    --model mll-lab/tstar-v1 \
    --video example.mp4 \
    --query "What is the man look like?" \
    --output (default output.json. Contains model, video, query, keyframe-numbers, keyframe-jpg itself.)
```

## Searching KeyFrame for Your Dataset

 def run_pipeline(self, frames: List[Image.Image], question: str, options: str, grounding_options: Optional[str] = None) -> Tuple[str, List[Image.Image], List[float]]:
        """
        Run the full pipeline: grounding, searching for keyframes, and performing QA.

        Args:
            frames (List[Image.Image]): Initial frames for grounding.
            question (str): Question to be answered.
            options (str): Multiple-choice options for the question.
            grounding_options (Optional[str]): Additional options for grounding.

        Returns:
            Tuple[str, List[Image.Image], List[float]]:
                - Answer from the QA inference.
                - List of keyframes used for QA.
                - List of timestamps corresponding to the keyframes.
        """
        # Step 1: Perform grounding to identify target and cue objects
        self.inference_query_grounding(frames, question, grounding_options)

        # Step 2: Search for keyframes
        keyframes, timestamps = self.search()

        print(f"Found Keyframes: {len(keyframes)} frames at timestamps: {timestamps}")

        # Step 3: Perform inference QA using the identified keyframes
        answer = self.inference_qa(keyframes, question, options)

        print(f"Inferred Answer: {answer}")

        return answer, keyframes, timestamps
