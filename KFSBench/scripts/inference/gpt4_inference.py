import argparse
import json
import numpy as np
from PIL import Image
import base64
import openai
import os
import time
from tqdm import tqdm
from kfs.utils import get_video_fps, VideoTime, extract_frames
from kfs.evaluation.datasets import insert_subtitles_into_frames
from concurrent.futures import ThreadPoolExecutor


def form_question_input(di, data_path, frame_paths, frame_timestamps):        
    with open(os.path.join(data_path, "subtitles", di["subtitle_path"])) as f:
        subtitles = json.load(f)
    inputs = insert_subtitles_into_frames(frame_paths, frame_timestamps, subtitles, di["starting_timestamp_for_subtitles"], di["duration"])

    ##### YOU MAY MODIFY THE FOLLOWING PART TO ADAPT TO YOUR MODEL #####
    inputs += ["Question: " + di["question"]]
    inputs += [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(di["candidates"])]
    inputs += ["Answer with the option's letter from the given choices directly. Remember this is a multiple-choice question, so please just output one letter, and do not include any other information."]
    ##### YOU MAY MODIFY THE PREVIOUS PART TO ADAPT TO YOUR MODEL #####
    input_content = []
    for input_item in inputs:
        if ".jpg" not in input_item:
            input_content.append({"type": "text", "text": input_item})
        else:
            base64_image = encode_image(input_item)
            input_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}})
        

    ##### CORRECT CHOICE WILL BE "@" FOR TEST SET SAMPLES #####
    return input_content

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate question-answering on video frames using GPT.")
    parser.add_argument("--videos_path", type=str, default="~/data/LongVideoBench", help="Path to video files.")
    parser.add_argument("--json_path", type=str, default="~/data/data-annotation-kfs/data/lvbench/json_results", help="Path to JSON QA files.")
    parser.add_argument("--json_file", type=str, required=True, help="JSON file with QA data.")
    parser.add_argument("--output_file", type=str, default="log/gpt4_result_lvbench.jsonl", help="Output JSONL file.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model.")
    parser.add_argument("--use_oracle", action="store_true", help="Use oracle answers for evaluation.")
    parser.add_argument("--use_uniform", action="store_true", help="Use uniform frames for evaluation.")

    return parser.parse_args()

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def gpt_evaluate(model, content):
    start = time.time()
    messages = [
        {"role": "user", "content": content},
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=300
    )
    latency = time.time() - start
    return_content = response.choices[0].message.content
    total_tokens = response.usage.total_tokens
    return return_content, latency, total_tokens

def process_json(data, videos_path, model, output_file, args):
    # 用多线程+pbar

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for item in data:
            futures.append(executor.submit(process_item, item, videos_path, model, output_file, args))
        for future in tqdm(futures):
            future.result()

    # for item in tqdm(data):
    #     process_item(item, videos_path, model, output_file)

def process_item(item, videos_path, model, output_file, args):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    candidate = item['candidates']
    frame_indexes = sorted(item['frame_indexes'])
    frame_timestamps = item['frame_timestamps']
    video_name = item['video_path']  # assuming this is the video filename
    ground_truth_index = item['position']
    

    video_path = os.path.join(videos_path, "videos", video_name)
    fps = get_video_fps(video_path)
    if fps is None:
        print(f"Error: Could not retrieve FPS for {video_path}")
        return

    if args.use_oracle:
        frame_indexes = item['position']
        frame_timestamps = [i / fps for i in frame_indexes]
    elif args.use_uniform:
        video_length = item['duration']
        frame_indexes = np.linspace(0, video_length * fps, num=len(frame_indexes), endpoint=False).astype(int)
        frame_timestamps = [i / fps for i in frame_indexes]
        
    temp_frames_path = "temp_frames/" + video_name
    os.makedirs(temp_frames_path, exist_ok=True)

    extract_frames(video_path, frame_indexes, temp_frames_path, use_decord=True, fig_scale=0.3, quality=100)
    frame_paths = [os.path.join(temp_frames_path, f"frame_{frame_index}.jpg") for frame_index in frame_indexes]
    
    question_input = form_question_input(item, videos_path, frame_paths, frame_timestamps)

    try:
        gpt_response, latency, total_tokens = gpt_evaluate(model, question_input)

        with open(output_file, "a") as out_f:
            out_f.write(json.dumps({
                **item,
                "gpt_response": gpt_response,
                "latency": latency,
                "total_tokens": total_tokens,
                "answer_correctness": item['outputs'][0].lower() == gpt_response[0].lower()
            }) + "\n")
        print(f"Processed {video_name} frame {frame_indexes}. Answer correctness: {item['outputs'][0].lower() == gpt_response[0].lower()}")
                    
    except Exception as e:
        print(f"Error processing {video_name} frame {frame_indexes}: {e}")

def main():
    args = parse_arguments()
    videos_path = args.videos_path
    json_file = os.path.join(args.json_path, args.json_file)
    output_file = args.output_file
    # load existing jsonl and drop those processed
    with open(json_file) as f:
        data = json.load(f)

    print(f"Processing {len(data)} items")
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            processed_data = [json.loads(line) for line in f]
        processed_id_names = set([item["id"] for item in processed_data])
        data = [item for item in data if item["id"] not in processed_id_names]
        print(f"Skipping {len(processed_data)} processed items, remaining {len(data)} items")

    process_json(data, videos_path, args.model, output_file, args)

if __name__ == "__main__":
    main()
