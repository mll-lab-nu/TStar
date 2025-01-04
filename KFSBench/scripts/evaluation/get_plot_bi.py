import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from kfs.utils import load_json

def format_value(value):
    """Round the value to 4 decimal places for cleaner output."""
    return round(value, 4)

def load_and_filter_json(path, group=None):
    """Load JSON data and filter by group if specified."""
    try:
        data = load_json(path)
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return []
    
    return [item for item in data if item['duration_group'] == int(group)] if group else data

def is_correct_choice(item, search_type):
    """Determine if the predicted answer is correct based on the search type."""
    if search_type == "linear_search":
        return item['linear_predict_answer'].index(max(item['linear_predict_answer'])) == item['correct_choice']
    elif search_type == "zoom_in_search":
        return ord(item['search_predict_answer']) - ord("A") == item['correct_choice']
    else:
        raise ValueError("Unknown search type: must be 'linear_search' or 'zoom_in_search'")

def calculate_dots(pred_json, search_type, args):
    """Calculate relative positions (dots) based on frame index and position."""
    fps_dict_path = load_json(args.fps_dict_path)
    dots = []
    for item in pred_json:
        frame_dots = []
        frame_indices = item['frame_index_linearsearch'] if search_type == "linear_search" else item['frame_index_adaframe_sec']
        fps = eval(fps_dict_path[item['video_path']])
        for pos in item['position']:
            min_diff_index = frame_indices[np.argmin([abs(pos / fps - (k / fps if search_type == "linear_search" else k)) for k in frame_indices])]
            min_diff_index /= fps if search_type == "linear_search" else 1
            frame_dot = (min_diff_index / item['duration'], pos / fps / item['duration'])
            frame_dots.append(frame_dot)
        dots.append(frame_dots)
    return dots

def plot_dots(dots_linear, correct_linear, dots_zoom_in, correct_zoom_in):
    """Create a dual plot comparing linear search and zoom-in search results."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 6))
    fig.suptitle("Relevance between Frame Index Difference and Answer Correctness", fontsize=16, fontweight='semibold')

    def plot_individual_dots(dots, correct, search_type, ax):
        """Plot dots on an individual subplot for a specific search type."""
        diff_correct = [d for i, frame_dots in enumerate(dots) if correct[i] for d in frame_dots]
        diff_incorrect = [d for i, frame_dots in enumerate(dots) if not correct[i] for d in frame_dots]

        if diff_correct:
            x, y = zip(*diff_correct)
            ax.scatter(x, y, color='green', label=f"Correct (Avg. Diff: {format_value(np.mean([abs(i[0] - i[1]) for i in diff_correct]))})", 
                       marker='o', s=30, alpha=0.7)
        if diff_incorrect:
            x, y = zip(*diff_incorrect)
            ax.scatter(x, y, color='lightcoral', label=f"Incorrect (Avg. Diff: {format_value(np.mean([abs(i[0] - i[1]) for i in diff_incorrect]))})", 
                       marker='x', s=25, alpha=0.7)

        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)

        ax.set_xlabel("Relative Annotation Position", fontsize=12)
        ax.set_ylabel("Relative Prediction Position", fontsize=12)
        ax.set_title(f"{search_type.capitalize().replace('_', ' ')}", fontsize=14)
        ax.grid(visible=True, which='both', color='lightgray', linestyle='--', linewidth=0.5)
        ax.legend(loc="upper left", bbox_to_anchor=(0.03, 0.98), fontsize=10, borderaxespad=0.)
        ax.set_aspect('equal', 'box')

    # Plot each search type
    plot_individual_dots(dots_linear, correct_linear, "linear_search", axs[0])
    plot_individual_dots(dots_zoom_in, correct_zoom_in, "zoom_in_search", axs[1])

    # Adjust layout and spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(wspace=0.3)
    os.makedirs("log/plots", exist_ok=True)
    plt.savefig("log/plots/dots_classification_plot_bi.png", dpi=300)
    plt.show()

def main(args):
    # Load and filter data
    linear_json = load_and_filter_json(args.linear_json, args.group)
    zoom_in_json = load_and_filter_json(args.zoom_in_json, args.group)

    # Calculate correctness and dots for each search type
    correct_linear = [is_correct_choice(item, "linear_search") for item in linear_json]
    correct_zoom_in = [is_correct_choice(item, "zoom_in_search") for item in zoom_in_json]
    dots_linear = calculate_dots(linear_json, "linear_search", args)
    dots_zoom_in = calculate_dots(zoom_in_json, "zoom_in_search", args)

    # Plot results
    plot_dots(dots_linear, correct_linear, dots_zoom_in, correct_zoom_in)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and plot metrics for linear and zoom-in search comparisons.")
    parser.add_argument("--linear_json", type=str, required=True, help="Path to the linear search JSON file.")
    parser.add_argument("--zoom_in_json", type=str, required=True, help="Path to the zoom-in search JSON file.")
    parser.add_argument("--group", type=str, required=False, help="Optional: Group filter for video frame positions.")
    parser.add_argument("--fps_dict_path", type=str, required=True, help="Path to the frame rate dictionary.")
    args = parser.parse_args()

    main(args)
