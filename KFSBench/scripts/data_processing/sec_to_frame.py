import json
import re
import argparse

def sec_to_frames(sec_list, fps):
    """Convert a list of seconds to frames using fps."""
    return [int(sec * fps) for sec in sec_list]

def process_json(input_file, output_file, fps_file):
    # Load FPS data
    with open(fps_file, 'r') as f:
        fps_data = json.load(f)
    
    # Load JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Process each object in the JSON
    data_new = []
    for obj in data:
        # Find keys with frame_index_xxx pattern
        video_id = obj.get("video_id") + ".mp4"
        # if video_id not in fps_data:
        #     print(f"Error: Video ID {video_id} not found in FPS data")
        #     continue
        fps_str = fps_data[video_id]
        fps = eval(fps_str)  # Convert to numerical FPS
        plausible_keys = ["frame_index_quaZooming", "frame_index_RLsearch", "frame_index_TZooming", "zoomingin_searching", "frame_index_linearsearch"]
        for key in plausible_keys:
            if key in list(obj.keys()):
                if type(obj[key][0]) == int:
                    obj["frame_indexes"] = obj[key]
                else:
                    obj["frame_indexes"] = [int(i * fps) for i in obj[key]]
                break
        # if "frame_index_quaZooming" in list(obj.keys()):
        #     if type(obj["frame_index_quaZooming"][0]) == int:
        #         obj["frame_indexes"] = obj["frame_index_quaZooming"]
        #     else:
        #         obj["frame_indexes"] = [int(i * fps) for i in obj["frame_index_quaZooming"]]

        # elif "frame_index_RLsearch" in list(obj.keys()):
        #     obj["frame_indexes"] = [int(i * fps) for i in obj["frame_index_RLsearch"]]
        # elif "frame_index_TZooming" in list(obj.keys()):
        #     obj["frame_indexes"] = [int(i * fps) for i in obj["frame_index_TZooming"]]
        # elif "zoomingin_searching" in list(obj.keys()):
        #     obj["frame_indexes"] = [i for i in obj["zoomingin_searching"]]
        # elif "frame_index_linearsearch" in list(obj.keys()):
        #     obj["frame_indexes"] = [i for i in obj["frame_index_linearsearch"]]
        else:
            print(f"Error: No frame index keys found in object {obj}")
            continue 

        obj["frame_timestamps"] = [f / fps for f in obj["frame_indexes"]]  # Convert to seconds
        data_new.append(obj)

    
    # Save modified data back to JSON
    with open(output_file, 'w') as f:
        json.dump(data_new, f, indent=4)

# File paths
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--input_file", type=str, default="data/lvbench/json_results/tzoom_search.json")
argument_parser.add_argument("--output_file", type=str, default="data/lvbench/json_results/tzoom_search_converted.json")
argument_parser.add_argument("--fps_file", type=str, default="data/lvbench/datasets/fps.json")

args = argument_parser.parse_args()
input_file = args.input_file
output_file = args.output_file
fps_file = args.fps_file

# Run the conversion
process_json(input_file, output_file, fps_file)
