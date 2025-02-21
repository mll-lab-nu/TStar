# TStar: A Unified KeyFrame Searching Framework for Video Question Answering

**TStar** is an advanced framework that integrates Keyframe Searching into Vision-Language Models (VLMs), enhancing their performance for extremely long video understandiong tasks. By efficiently identifying relevant frames within videos, TStar improves the ability of state-of-the-art models like LLaVA-oneVision and GPT-4o to understand and reason over video data.

## Features
- **Iteratively Searching**: Iteratively identifies and focuses on the most relevant visual information in videos based on the question being asked.
- **Plug-in**: Easily integrates various grounding and searching backends.
- **Efficient Video QA**: Combines T* keyframe search with advanced video question answering capabilities.

---

## Getting Started
### Installation

```bash
## Follow docs/installation to implemet Grounding (e.g., LLaVA) and Searching (e.g., YOLO) Function
###  Install Query Grounder Interface(LLaVA or GPT-API) 
### Optional if you test with GPT4o
git clone https://github.com/LLaVA-VL/LLaVA-NeXT  

### Install Image Grid Scorer Interface e.g., YOLO-WORLD
git clone --recursive https://github.com/AILab-CVC/YOLO-World.git
```

### Structure:
```bash
LV-Haystack/
├── LLaVA-NeXT/                # Query grounding and QA interface (e.g., LLaVA or GPT-4 API)
├── YOLO-World/                # Object detection model with open vocabulary
├── TStar/                     # Core Python module for T* keyframe search 
│   ├── interface_llm.py       # Interface for grounding questions with VLMs
│   ├── interface_yolo.py      # Function for scoring images using YOLO
│   ├── interface_searcher.py  # Logic for searching keyframes in T*
│   ├── TStarFramework.py      # Example class integrating T* searching with QA
├── HaystackBench              # Scripts for inference on the LV-Haystack dataset
│   ├── run_TStar_onDataset.py # Run keyframe search on a given dataset (e.g., LongVideoBench)
│   ├── val_kfs_results.py     # Evaluate keyframe search results on LV-Haystack
│   ├── val_qa_results.py      # Evaluate video question answering with searched keyframes
├── README.md                  # Documentation for the repository


```

## Run TStar Demo

The example below demonstrates how to perform video question answering with keyframe searching framework. This example uses GPT-4o as the VLM and YOLO-World as scoring function.

```python
export OPENAI_API_KEY=your_openai_api_key

python run_TStar_Demo_onVideo.py \
    --video_path /path/to/LV-Haystack/38737402-19bd-4689-9e74-3af391b15feb.mp4 \
    --question "What is the color of the couch?" \
    --options "A) Red, B) Blue, C) Green, D) Yellow" \
    --yolo_config_path ./YOLOWorld/configs/yolo_config.py \
    --yolo_checkpoint_path ./pretrained/yolo_checkpoint.pth \
    --search_nframes 8 \
    --image_grid_shape 8 8 \
    --confidence_threshold 0.6 \
    --device cuda:0
```


## Test on LV-HayStack
To evaluate T* on a dataset (e.g., LV-Haystack), use the following command:

```bash
bash ./eval_LV_Haystack.sh
```
</details>

### Running T* on Your Dataset

To process your own dataset with T*, you need to prepare a JSON file describing the dataset. The JSON file should follow the format below:
<details>
  <summary>Click to expand JSON examples!</summary>
  
```bash
[
    {
        "file_name": "example_video.mp4",
        "question": "What is the color of the couch?",
        "choices": {
            "A": "Red",
            "B": "Blue",
            "C": "Green",
            "D": "Yellow"
        },
        "frame_indexes": [10, 50, 100]  // Optional: Use this for specific frame sampling
    },
    {
        "file_name": "another_video.mp4",
        "question": "What object is next to the chair?",
        "choices": {
            "A": "Table",
            "B": "Lamp",
            "C": "Sofa",
            "D": "Bookshelf"
        }
    }
]
```
</details>

Once your dataset is prepared, you can run TStar to perform keyframe searching. Use the following command:

<details>
  <summary>Click to expand python script!</summary>
  
```python
python HaystackBench/run_TStar_onDataset.py \
    --input_json path_to_your_annotations.json \
    --output_json path_to_your_annotations_Tstar_frames.json \
    --video_dir ./Data/Haystack-Bench/videos \
    --yolo_config_path ./YOLOWorld/configs/pretrain/yolo_world_v2_xl_vlpan_bn.py \
    --yolo_checkpoint_path ./pretrained/yolo_checkpoint.pth \
    --search_nframes 8 \
    --image_grid_shape 8 8 \
    --confidence_threshold 0.5 \
    --device cuda:0

# new you have add predict frame index in your annotations json
# and sampine frame with the T* prediction for your works!

```
</details>
