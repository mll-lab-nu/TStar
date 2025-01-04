import json
metadata = json.load(open("data/lvbench/datasets/lvb_val.json"))
from collections import defaultdict

# Dictionary to store video data
video_questions = defaultdict(dict)

# Populate the video_questions dictionary
for item in metadata:
    video_name = item['video_path']
    question = item['question']
    candidate = item['candidates']
    if 'questions' not in video_questions[video_name]:
        video_questions[video_name]['questions'] = []
    if 'candidates' not in video_questions[video_name]:
        video_questions[video_name]['candidates'] = []
    if 'positions' not in video_questions[video_name]:
        video_questions[video_name]['positions'] = []
    if 'correct_choices' not in video_questions[video_name]:
        video_questions[video_name]['correct_choices'] = []
    if 'durations' not in video_questions[video_name]:
        video_questions[video_name]['durations'] = []

    video_questions[video_name]['questions'].append(item['question'])
    video_questions[video_name]['candidates'].append(item['candidates'])
    video_questions[video_name]['positions'].append(item['position'])
    video_questions[video_name]['correct_choices'].append(item['correct_choice'])
    video_questions[video_name]['durations'].append(item['duration'])


# Converting to the desired list of dictionaries format
transferred_data = [{"video_name": video, **instance} for video, instance in video_questions.items()]
transferred_data_short = [i for i in transferred_data if i['durations'][0] <= 600]
transferred_data_long = [i for i in transferred_data if i['durations'][0] > 600]
# Display the result
json.dump(transferred_data_short, open("data/lvbench/datasets/questions-all-formatted-lvbench.json", "w"), indent=4)
json.dump(transferred_data_long, open("data/lvbench/datasets/questions-all-formatted-lvbench-long.json", "w"), indent=4)
