# TStar: A Unified KeyFrame Searching Framework for Video Question Answering

**TStar** is a comprehensive framework designed to integrate **KeyFrame Searching** into Vision-Language Models (VLMs) to enhance Video Question Answering (VQA). By leveraging efficient keyframe searching, TStar dynamically identifies relevant frames in videos, enabling state-of-the-art VLMs like **LLaVA** to achieve improved performance in understanding and reasoning over video data.

## Features
- **Keyframe Searching**: Dynamically detects and extracts relevant keyframes.
- **Modular Design**: Easily integrates various grounding and searching backends.
- **Efficient Video QA**: Combines T* searching with advanced VQA.

---

## Getting Started
### Installation

```bash
## Follow docs/installation to implemet Grounding (e.g., LLaVA) and Searching (e.g., YOLO) Function
###  Install Query Grounder Interface(LLaVA or GPT-API)
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
### Install Image Grid Scorer Interface e.g., YOLO-WORLD
git clone --recursive https://github.com/AILab-CVC/YOLO-World.git
```

### Structure:
```bash
VL-Haystack/
├── LLaVA-NeXT/                # Query grounding and QA interface (LLaVA or skip by using GPT-4o-API)
├── YOLO-World/                # Heuristic-based image scoring and searching (YOLO)
├── TStar/                     # Core T* searching and framework integration
│   ├── interface_llm.py       # LLM-based interface for question grounding and answering
│   ├── interface_yolo.py      # YOLO-based object detection interface
│   ├── interface_searcher.py  # Searching logic for T* heuristic processing
│   ├── TStarFramewor.py  # Demonstration class for integrating T* searching with QA
├── HaystackBench              # Script for inference on LV-Haystack dataset
│   ├── val_LV_Haystack.py  
├── README.md                  # Project readme

```

## Run VideoSearching Demo

The example below demonstrates how to perform video question answering with keyframe searching framework. This example uses LLaVA-OneVision as the VLM and YOLO-World for keyframe searching.

```python
export OPENAI_API_KEY=your_openai_api_key

python TStar/TStarFramework.py \
    --video_path /path/to/LV-Haystack/38737402-19bd-4689-9e74-3af391b15feb.mp4 \
    --question "What is the color of the couch?" \
    --options "A) Red, B) Blue, C) Green, D) Yellow" \
    --llava_model_path lmms-lab/llava-onevision-qwen2-7b-ov \
    --yolo_config_path ./YOLOWorld/configs/yolo_config.py \
    --yolo_checkpoint_path ./pretrained/yolo_checkpoint.pth \
    --search_nframes 8 \
    --image_grid_shape 8 8 \
    --confidence_threshold 0.6 \
    --device cuda:0
```


## Test VL-HayStack
To evaluate T* on a dataset (e.g., LV-Haystack), use the following command:

```python
python TStar/val_LV_Haystack.py \
    --input_json ./Datasets/Haystack-Bench/annotations.json \
    --output_json ./Resoults/Haystack_Bench_Seaching.json \
    --video_dir ./Data/Haystack-Bench/videos \
    --llava_model_path lmms-lab/llava-onevision-qwen2-7b-ov \
    --yolo_config_path ./YOLOWorld/configs/pretrain/yolo_world_v2_xl_vlpan_bn.py \
    --yolo_checkpoint_path ./pretrained/yolo_checkpoint.pth \
    --search_nframes 8 \
    --image_grid_shape 8 8 \
    --confidence_threshold 0.5 \
    --output_dir ./outputs \
    --device cuda:0
```

## Searching KeyFrame for Your Dataset

### Preparing Your Dataset
To process your own dataset with VL-Haystack, you need to prepare a JSON file describing the dataset. The JSON file should follow the format below:
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


### Running the Framework on Your Dataset

```python
python TStar/val_LV_Haystack.py \
```


### Output Format

After running the script, the '''output.json''' file will be updated with the following keys:

<details>
  <summary>Click to expand output examples!</summary>
[
    {
        "file_name": "example_video.mp4",
        "question": "What is the color of the couch?",
        "choices": { },
        "predicted_answer": "B",
        "frame_indexes": [10, 50, 100],
        "predicted_timestamps": [0.33, 1.67, 3.33],
        "estimated_frame_distribution": [0.01, 0.0, ..., 0.1]
    },
]
</details>

 Use the predicted_timestamps or sampling from estimated_frame_distribution as inputs for tasks like Video QA.



 \begin{table}[ht]
    \centering
    \setlength\tabcolsep{3pt} % Reduce horizontal padding between columns
    \setlength\extrarowheight{1pt} % Reduce vertical padding between rows
    \arrayrulecolor[gray]{0.7} % Set a light gray color for the borders
    \begin{adjustbox}{width=0.98\linewidth}
    \begin{tabular}{l|c|c|c|c|c}
        \toprule
        \multicolumn{6}{c}{\bf LongVideoBench} \\
        \cmidrule{1-6}
        
        \multirow{3}{*}{\bf Model and Size} & \multirow{3}{*}{\bf \#Frame} & \multicolumn{4}{c}{\textbf{Video Length}} \\
        & & \textbf{XLong} & \textbf{Long} & \textbf{Medium} & \textbf{Short} \\
        & & 15-60min & 2-10min & 15-60s & 8-15s \\

        \midrule
        
        GPT4o & 8 & 47.1 & 49.4 & 67.3 & 69.7 \\
        \rowcolor{gray!10}
        GPT4o + {\fancy} & 8 & \textbf{51.9 / 51.2} & \textbf{52.4 / 51.0} & \textbf{72.7 / 73.0} & \textbf{70.0/ 70.3} \\
        \arrayrulecolor{gray!40}\hline \arrayrulecolor{black}

        LLaVA-OneVision-72B & 8 & 53.7 & 57.4 & 74.1 & 73.0 \\
        \rowcolor{gray!10}
        LLaVA-OneVision-72B + {\fancy} & 8 & \textbf{55.5 / 55.0} & \textbf{63.7/ 63.3} & \textbf{76.3 / 76.9 } & \textbf{73.5 / 73.9} \\

        \midrule
        GPT4o & 32 & 50.5 & 57.3 & 73.5 & 71.4 \\
        \rowcolor{gray!10}
        GPT4o + {\fancy} & 32 & \textbf{53.1 / 53.4} & \textbf{59.4 / 59} & \textbf{74.3/ 74.3} & \textbf{71.4, 71.4} \\
        \arrayrulecolor{gray!40}\hline \arrayrulecolor{black}

        LLaVA-OneVision-72B & 32 & 56.5 & 61.6 & 77.4 & 74.3 \\
        \rowcolor{gray!10}
        LLaVA-OneVision-72B + {\fancy} & 32 & \textbf{62.4 / 62.7} & \textbf{64.1/ 63.7} & \textbf{79.3 / 79.3} & \textbf{74.6 / 74.6} \\

        \bottomrule
    \end{tabular}
    \end{adjustbox}
    \caption{
    Extrinsic evaluation results for $T^*$ as an additional frame selection module for VLMs on LongVideoBench. The metric is QA accuracy (\%). We run {\fancy} three times and report average accuracy and Standard Deviation.
    }
    \label{tab:longvideobench_results}    
\end{table}



# Jinhui

export https_proxy="http://10.120.16.212:20000"; export http_proxy="http://10.120.16.212:20000"; export all_proxy="socks5://10.120.16.212:20000"

