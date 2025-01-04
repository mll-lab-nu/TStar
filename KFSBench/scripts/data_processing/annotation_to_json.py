import os
from kfs.utils import load_json

def get_nested_dict_with_answers(folder_path):
    """Generate a nested dictionary for all image files and answer in the folder structure."""
    nested_dict = {}
    for root, _, files in os.walk(folder_path):
        relative_path = os.path.relpath(root, folder_path)
        path_parts = relative_path.split(os.sep)
        current_level = nested_dict

        for part in path_parts:
            part = int(part) if part.isdigit() else part
            if part == '.':
                continue
            current_level = current_level.setdefault(part, {})
        
        if files:
            current_level['files'] = files
    return nested_dict


results = get_nested_dict_with_answers("static/annotations")
questions = load_json("data/questions-all-formatted.json")

questions_formatted = []

for idx_v, video in enumerate(questions):
    video_frames, video_answers = [], []
    for idx_q, question in enumerate(video['questions']):
        files = results[idx_v][idx_q]['snapshots']['files'] # like ['frame_3882.jpg', 'frame_129.jpg', 'answer.txt']
        frames = [f for f in files if f.endswith('.jpg')]
        # extract int from frames
        frames = [int(f.split('_')[1].split('.')[0]) for f in frames]
        answer = [f for f in files if f.endswith('.txt')][0]
        # extract answer from answers
        answer = os.path.join("static/annotations", str(idx_v), str(idx_q), "snapshots", answer)
        answer = open(answer).read().strip()
        video_frames.append(frames)
        video_answers.append(answer)
    questions_formatted.append({**video, 'frames': sorted(video_frames), 'answers': video_answers})

# save back
import json
with open("data/questions-annotated.json", "w") as f:
    json.dump(questions_formatted, f, ensure_ascii=False, indent=4)

