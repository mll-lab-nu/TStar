import json

# 读取原始 JSON 文件
with open('data/questions-all.json', 'r') as infile:
    data = json.load(infile)

# 转换数据结构
converted_data = {}

for idx, entry in enumerate(data):
    if 'query' not in entry:
        print("QUERY_NOT_IN_ENTRY")
        continue
    video_name = f"{entry['source_clip_uid']}.mp4"
    questions = entry['query']
    clip_time = entry['binary_search_trajectory'][-1]
    clip_relative = [i - entry['source_clip_video_start_sec'] for i in clip_time]
    if video_name not in converted_data:
        converted_data[video_name] = {"questions": [], "CLIP-result-relative":[]}
    converted_data[video_name]["questions"].append(questions)
    converted_data[video_name]["CLIP-result-relative"].append(clip_relative)

converted_data = [{"video_name":key} | value for key, value in converted_data.items()]

# 保存转换后的 JSON 文件
with open('data/questions-all-formatted.json', 'w') as outfile:
    json.dump(converted_data, outfile, indent=4)
