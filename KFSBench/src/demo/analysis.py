import os
import numpy as np
from PIL import Image
from moviepy import VideoFileClip
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def create_timeline(frame_times, duration):
    try:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.set_xlim(0, duration)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Time (s)")
        ax.set_title("Video Timeline with Sampled Frames")

        ax.hlines(0.5, 0, duration, colors="gray", linestyles="dotted")
        ax.plot(frame_times, [0.5] * len(frame_times), 'ro')

        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)
    except Exception as e:
        print(f"Error in create_timeline: {e}")
        return None


def analyze_and_sample_frames(video_file, num_frames=8, batch=1, total_batches=3):
    try:
        if not video_file:
            return {"Error": "No file uploaded."}, None, None, None

        clip = VideoFileClip(video_file.name)
        metadata = {
            "Filename": os.path.basename(video_file.name),
            "Duration (seconds)": round(clip.duration, 2),
            "Resolution": f"{clip.size[0]}x{clip.size[1]}",
            "FPS": clip.fps,
        }

        total_frames = int(clip.duration * clip.fps)
        frame_indices = np.linspace(0, total_frames - 1, num_frames * total_batches, dtype=int)
        batch_indices = frame_indices[(batch - 1) * num_frames:batch * num_frames]
        frame_times = [i / clip.fps for i in batch_indices]

        sampled_frames = [Image.fromarray(clip.get_frame(time)) for time in frame_times]
        timeline_image = create_timeline(frame_times, clip.duration)
        clip.close()
        return metadata, sampled_frames, frame_times, timeline_image

    except Exception as e:
        return {"Error": str(e)}, None, None, None
