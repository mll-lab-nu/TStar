import gradio as gr
from moviepy import VideoFileClip
import os
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import asyncio
from kfs.demo.analysis import analyze_and_sample_frames, create_timeline

def create_ui_components():
    with gr.Row():
        video_input = gr.File(label="Upload your video", type="filepath")  # 视频上传
        submit_button = gr.Button("Analyze and Sample Iteratively")

    state_batches = gr.State([])  # 存储所有生成的批次数据
    current_display_batch = gr.State(None)  # 当前显示的批次

    output_timeline = gr.Image(label="Video Timeline", type="pil", visible=False)
    output_frames = gr.Gallery(label="Sampled Frames", columns=8, visible=False, height=200)
    batch_status = gr.Text(label="Batch Status", value="No Batch Processed Yet", visible=True)
    batch_selector = gr.Dropdown(choices=[], label="Select Batch", visible=False)
    output_metadata = gr.JSON(label="Video Metadata", visible=False)

    return (
        video_input,
        submit_button,
        state_batches,
        current_display_batch,
        output_timeline,
        output_frames,
        batch_status,
        batch_selector,
        output_metadata,
    )


# 切换批次显示函数
def switch_batch(state_batches, selected_batch):
    if not selected_batch or selected_batch == "":
        return None, None, None, None
    batch_index = int(selected_batch.split()[-1]) - 1
    timeline_image, frames, metadata = state_batches[batch_index]
    return gr.update(value=timeline_image, visible=True), gr.update(value=frames, visible=True), gr.update(value=metadata, visible=True), selected_batch


# 异步处理函数：动态生成帧并更新UI
async def process_video_iteratively_with_state(video_file, state_batches, current_display_batch, total_batches=10, num_frames=8):
    if not video_file:
        yield None, None, None, "No video uploaded!", None, state_batches, current_display_batch
        return

    metadata = None
    for batch in range(1, total_batches + 1):
        metadata, frames, frame_times, timeline_image = analyze_and_sample_frames(
            video_file, num_frames=num_frames, batch=batch, total_batches=total_batches
        )
        if metadata is None:
            continue

        # 更新状态
        state_batches.append((timeline_image, frames, metadata))
        batch_choices = [f"Batch {i + 1}" for i in range(len(state_batches))]

        if current_display_batch is None or current_display_batch == f"Batch {batch - 1}":
            current_display_batch = f"Batch {batch}"

        # 动态更新UI
        yield (
            gr.update(value=timeline_image, visible=True),
            gr.update(value=frames, visible=True),
            gr.update(value=metadata, visible=True),
            f"Processing Batch: {batch} / Total Batches: {total_batches}",
            gr.update(choices=batch_choices, value=f"Batch {batch}", visible=True),
            state_batches,
            current_display_batch
        )

        # 模拟处理延迟（如果需要）
        await asyncio.sleep(0.5)