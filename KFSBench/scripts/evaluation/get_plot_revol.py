import json
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import savgol_filter

# Load JSON files
jsonfile = json.load(open('data/searching_distribution_shift.json'))
all_ids = []
for i in jsonfile:
    if i['video_id'] not in all_ids:
        all_ids.append(i['video_id'])
all_ids.sort(reverse=True)

# Define the plot function
def plot_history_subplot(ax, history, truth, window_length=11, polyorder=2):
    """
    Plots each sub-list in 'history' as a separate line on the given axis (ax) 
    with colors gradually becoming darker for later lists, using Savitzky-Golay filter to smooth the lines.
    
    Parameters:
    ax (matplotlib.axes): The axis to plot on.
    history (list of list of floats): A 2D list where each sub-list 
                                      represents a line to be plotted.
    window_length (int): The length of the filter window (must be odd).
    polyorder (int): The order of the polynomial used to fit the samples.
    """
    # Number of lines to plot
    num_lines = len(history)
    
    # Generate colors from light to dark blue
    colors = plt.cm.Blues(np.linspace(0.3, 0.7, num_lines))
    
    # Plot each sublist in history with Savitzky-Golay smoothing
    for i, line in enumerate(history):
        # Apply Savitzky-Golay filter
        smoothed_line = savgol_filter(line, window_length=window_length, polyorder=polyorder)
        ax.plot(smoothed_line, color=colors[i], label=f'Iteration {10*i+1}', linestyle='-', linewidth=1)
    
    # Plot ground truth
    ax.axvline(x=truth[0], color='red', linestyle='--', label='Ground Truth', linewidth=1)
    for t in truth[1:]:
        ax.axvline(x=t, color='red', linestyle='--', linewidth=1)
    
    # Customize plot appearance
    ax.set_xlabel("sec (Frame)")
    ax.set_ylabel("sample weight")
    ax.legend(fontsize=8)

# Set up the subplot grid
num_videos = len(all_ids)
rows = 3  # Adjust the grid size based on the number of plots
fig, axes = plt.subplots(rows, 1, figsize=(14*0.9, rows * 4 / 2))

# Flatten axes for easier indexing (in case rows or columns are more than needed)
axes = axes.flatten()

# Load FPS file
fps_file = json.load(open('data/lvbench/datasets/fps.json'))

# Iterate over videos and plot each one on a subplot
plot_count = 0
for item_id in range(num_videos):
    video = [i for i in jsonfile if i['video_id'] == all_ids[item_id]]
    fps = fps_file.get(video[0]['video_id'] + ".mp4")
    fps = eval(fps)

    # Extract history and ground truth positions
    history, gt = video[0]['distributions_history'], video[0]['position']
    gt = [i / fps for i in gt]
    if len(gt) < 2:
        continue
    max_attn = max(history[-1])
    gt_attn = [history[-1][int(cur)] for cur in gt]
    if min(gt_attn) < 0.5 * max_attn:
        continue

    # Plot on a subplot
    plot_history_subplot(axes[plot_count], history[::2], gt)
    axes[plot_count].set_title(f"Video ID: {video[0]['video_id']}")
    plot_count += 1

# Hide any unused subplots
for i in range(plot_count, len(axes)):
    axes[i].axis('off')

# Adjust layout and save the final plot grid
plt.tight_layout()
os.makedirs("log/plots", exist_ok=True)
plt.savefig("log/plots/all_plots_in_one_file.png", dpi=300)
plt.show()