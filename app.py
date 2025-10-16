#!/usr/bin/env python3
"""
Qwen Image Edit Gradio Demo
Runs locally on RTX 5090 with Blackwell architecture
"""

import gradio as gr
import torch
from diffusers import QwenImageEditPipeline
from dfloat11 import DFloat11Model
from PIL import Image
import numpy as np
from datetime import datetime
import gc

# Global variables
pipe = None
edit_history = []


def get_gpu_memory():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"GPU Memory: {allocated:.2f}GB / {total:.2f}GB (Reserved: {reserved:.2f}GB)"
    return "GPU not available"


def load_pipeline():
    """Load the Qwen Image Edit pipeline with memory optimizations"""
    global pipe

    if pipe is None:
        print("Loading Qwen-Image-Edit pipeline...")

        # Load the original pipeline
        pipe = QwenImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit",
            torch_dtype=torch.bfloat16,
        )

        # Apply DFloat11 compression to transformer (32% size reduction, lossless)
        print("Applying DFloat11 compression...")
        DFloat11Model.from_pretrained(
            "DFloat11/Qwen-Image-Edit-DF11",
            device="cpu",
            bfloat16_model=pipe.transformer,
        )

        # Enable CPU offloading to reduce peak VRAM (moves models to CPU when not in use)
        print("Enabling CPU offloading...")
        pipe.enable_model_cpu_offload()

        # Enable attention slicing to reduce memory during inference
        print("Enabling attention slicing...")
        pipe.enable_attention_slicing()

        print("Pipeline loaded successfully!")
        print(get_gpu_memory())

    return pipe


def preprocess_image(image, max_size=1024):
    """Resize image if too large while maintaining aspect ratio"""
    if image is None:
        return None

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Resize if necessary
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        image = image.resize((new_width, new_height), Image.LANCZOS)
        print(f"Resized image from {width}x{height} to {new_width}x{new_height}")

    return image


def edit_image(
    input_image,
    prompt,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    seed=-1,
    progress=gr.Progress()
):
    """
    Main image editing function

    Args:
        input_image: Input PIL Image or numpy array
        prompt: Text prompt for editing
        num_inference_steps: Number of denoising steps
        true_cfg_scale: Guidance scale for editing
        seed: Random seed (-1 for random)
        progress: Gradio progress tracker
    """
    global edit_history

    try:
        # Validation
        if input_image is None:
            return None, "Please upload an image first!", get_gpu_memory()

        if not prompt or prompt.strip() == "":
            return None, "Please enter a prompt!", get_gpu_memory()

        # Load pipeline
        progress(0.1, desc="Loading pipeline...")
        pipeline = load_pipeline()

        # Preprocess image
        progress(0.2, desc="Preprocessing image...")
        processed_image = preprocess_image(input_image)

        # Set seed
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Run inference
        progress(0.3, desc=f"Editing image ({num_inference_steps} steps)...")

        output = pipeline(
            image=processed_image,
            prompt=prompt,
            num_inference_steps=int(num_inference_steps),
            true_cfg_scale=float(true_cfg_scale),
            generator=generator,
        )

        edited_image = output.images[0]

        # Add to history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        edit_history.append({
            "timestamp": timestamp,
            "prompt": prompt,
            "original": processed_image,
            "edited": edited_image,
            "seed": seed,
        })

        # Keep only last 10 edits
        if len(edit_history) > 10:
            edit_history.pop(0)

        progress(1.0, desc="Done!")

        # Clean up
        gc.collect()
        torch.cuda.empty_cache()

        info_text = f"✓ Edit completed!\nSeed: {seed}\nSteps: {num_inference_steps}\nCFG Scale: {true_cfg_scale}"

        return edited_image, info_text, get_gpu_memory()

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return None, error_msg, get_gpu_memory()


def get_history_gallery():
    """Return gallery data for history"""
    if not edit_history:
        return []

    # Return list of (image, caption) tuples
    gallery_data = []
    for item in reversed(edit_history):  # Show newest first
        gallery_data.append((
            item["edited"],
            f"{item['timestamp']}\n{item['prompt'][:50]}..."
        ))

    return gallery_data


def clear_history():
    """Clear edit history"""
    global edit_history
    edit_history = []
    gc.collect()
    torch.cuda.empty_cache()
    return [], "History cleared!", get_gpu_memory()


# Build Gradio interface
def create_interface():
    """Create the Gradio interface"""

    with gr.Blocks(
        title="Qwen Image Edit Demo",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown("""
        # 🎨 Qwen Image Edit Demo

        Upload an image and describe the changes you want to make. The model will edit your image based on your prompt.

        **Powered by:** Qwen-Image-Edit - DF11
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### Input")
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=400
                )

                prompt = gr.Textbox(
                    label="Edit Prompt",
                    placeholder="Describe the changes you want to make...",
                    lines=3
                )

                # Advanced settings
                with gr.Accordion("Advanced Settings", open=False):
                    num_steps = gr.Slider(
                        minimum=20,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Inference Steps",
                        info="More steps = better quality but slower"
                    )

                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=4.0,
                        step=0.5,
                        label="CFG Scale",
                        info="How strongly to follow the prompt"
                    )

                    seed = gr.Number(
                        label="Seed",
                        value=-1,
                        precision=0,
                        info="-1 for random seed"
                    )

                # Buttons
                with gr.Row():
                    edit_btn = gr.Button("✨ Edit Image", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)

            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### Output")
                output_image = gr.Image(
                    label="Edited Image",
                    type="pil",
                    height=400
                )

                info_box = gr.Textbox(
                    label="Status",
                    lines=3,
                    interactive=False
                )

                gpu_memory = gr.Textbox(
                    label="GPU Memory Usage",
                    interactive=False,
                    value=get_gpu_memory()
                )

        # History section
        gr.Markdown("### 📜 Edit History (Last 10)")

        with gr.Row():
            refresh_history_btn = gr.Button("🔄 Refresh History")
            clear_history_btn = gr.Button("🗑️ Clear History")

        history_gallery = gr.Gallery(
            label="Previous Edits",
            show_label=False,
            columns=5,
            rows=2,
            height="auto",
            object_fit="contain"
        )

        # Tips section
        with gr.Accordion("💡 Tips for Best Results", open=False):
            gr.Markdown("""
            - **Be specific**: "turn the cat into a golden retriever" works better than "change animal"
            - **Lighting & style**: Try "make it nighttime", "add dramatic lighting", "make it look like anime"
            - **Season & weather**: "change to winter with snow", "make it sunny"
            - **Artistic styles**: "turn into watercolor painting", "make it cyberpunk style"
            - **Inference steps**: 30-50 is usually enough, more steps = slower but potentially better quality
            - **CFG Scale**: 4.0 is recommended, lower = more creative, higher = follows prompt more strictly
            - **Image size**: Large images are auto-resized to 1024px for optimal performance
            """)

        # Event handlers
        edit_btn.click(
            fn=edit_image,
            inputs=[input_image, prompt, num_steps, cfg_scale, seed],
            outputs=[output_image, info_box, gpu_memory]
        )

        clear_btn.click(
            fn=lambda: (None, "", "", get_gpu_memory()),
            outputs=[input_image, prompt, info_box, gpu_memory]
        )

        refresh_history_btn.click(
            fn=get_history_gallery,
            outputs=[history_gallery]
        )

        clear_history_btn.click(
            fn=clear_history,
            outputs=[history_gallery, info_box, gpu_memory]
        )

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen Image Edit Gradio Demo")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 60)

    # Create and launch interface
    demo = create_interface()
    demo.queue()  # Enable queuing for better handling of concurrent requests
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=False,  # Set to True to create a public link
        show_error=True,
    )
