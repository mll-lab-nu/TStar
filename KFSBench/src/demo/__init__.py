import gradio as gr
from kfs.demo.ui_components import create_ui_components, switch_batch, process_video_iteratively_with_state


if __name__ == "__main__":
    # 创建 Gradio UI
    with gr.Blocks() as demo:
        video_input, submit_button, state_batches, current_display_batch, output_timeline, output_frames, batch_status, batch_selector, output_metadata = create_ui_components()

        submit_button.click(
            fn=process_video_iteratively_with_state,
            inputs=[video_input, state_batches, current_display_batch],
            outputs=[
                output_timeline, output_frames, output_metadata, batch_status,
                batch_selector, state_batches, current_display_batch
            ]
        )

        batch_selector.change(
            fn=switch_batch,
            inputs=[state_batches, batch_selector],
            outputs=[output_timeline, output_frames, output_metadata, current_display_batch]
        )

    # 启动 Gradio 应用
    demo.launch()