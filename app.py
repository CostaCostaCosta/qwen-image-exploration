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
    """Load the Qwen Image Edit 2509 pipeline with memory optimizations"""
    global pipe

    if pipe is None:
        print("Loading Qwen-Image-Edit-2509 pipeline...")

        # Load the original pipeline
        pipe = QwenImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            torch_dtype=torch.bfloat16,
        )

        # Apply DFloat11 compression to transformer (32% size reduction, lossless)
        print("Applying DFloat11 compression...")
        DFloat11Model.from_pretrained(
            "DFloat11/Qwen-Image-Edit-2509-DF11",
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


def preprocess_images(images, max_size=1024):
    """Preprocess multiple images (1-3 images)"""
    if not images:
        return None

    # Filter out None values
    valid_images = [img for img in images if img is not None]

    if not valid_images:
        return None

    # Process each image
    processed = [preprocess_image(img, max_size) for img in valid_images]

    # Return single image or list based on count
    if len(processed) == 1:
        return processed[0]
    return processed


def edit_image(
    input_image1,
    input_image2,
    input_image3,
    prompt,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    seed=-1,
    progress=gr.Progress()
):
    """
    Main image editing function supporting 1-3 images

    Args:
        input_image1: First input image (required)
        input_image2: Second input image (optional)
        input_image3: Third input image (optional)
        prompt: Text prompt for editing
        num_inference_steps: Number of denoising steps
        true_cfg_scale: Guidance scale for editing
        seed: Random seed (-1 for random)
        progress: Gradio progress tracker
    """
    global edit_history

    try:
        # Collect all input images
        input_images = [input_image1, input_image2, input_image3]

        # Validation
        if input_image1 is None:
            return None, "Please upload at least one image!", get_gpu_memory()

        if not prompt or prompt.strip() == "":
            return None, "Please enter a prompt!", get_gpu_memory()

        # Load pipeline
        progress(0.1, desc="Loading pipeline...")
        pipeline = load_pipeline()

        # Preprocess images
        progress(0.2, desc="Preprocessing images...")
        processed_images = preprocess_images(input_images)

        # Count valid images
        num_images = len(processed_images) if isinstance(processed_images, list) else 1

        # Set seed
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Run inference
        progress(0.3, desc=f"Editing {num_images} image(s) ({num_inference_steps} steps)...")

        output = pipeline(
            image=processed_images,
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
            "original": processed_images if isinstance(processed_images, list) else [processed_images],
            "edited": edited_image,
            "seed": seed,
            "num_images": num_images,
        })

        # Keep only last 10 edits
        if len(edit_history) > 10:
            edit_history.pop(0)

        progress(1.0, desc="Done!")

        # Clean up
        gc.collect()
        torch.cuda.empty_cache()

        info_text = f"✓ Edit completed!\nImages used: {num_images}\nSeed: {seed}\nSteps: {num_inference_steps}\nCFG Scale: {true_cfg_scale}"

        return edited_image, info_text, get_gpu_memory()

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
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
        # 🎨 Qwen Image Edit 2509 Demo

        Upload 1-3 images and describe the changes you want to make. The model supports multi-image editing for combining, merging, and referencing multiple images.

        **Powered by:** Qwen-Image-Edit-2509 with DFloat11 Compression | **Supports:** 1-3 images
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### Input Images (1-3)")

                with gr.Row():
                    input_image1 = gr.Image(
                        label="Image 1 (Required)",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=200
                    )
                    input_image2 = gr.Image(
                        label="Image 2 (Optional)",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=200
                    )

                input_image3 = gr.Image(
                    label="Image 3 (Optional)",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=200
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
            **Single Image Editing:**
            - **Be specific**: "turn the cat into a golden retriever" works better than "change animal"
            - **Lighting & style**: Try "make it nighttime", "add dramatic lighting", "make it look like anime"
            - **Season & weather**: "change to winter with snow", "make it sunny"
            - **Artistic styles**: "turn into watercolor painting", "make it cyberpunk style"

            **Multi-Image Editing (2-3 images):**
            - **Person + Person**: "Place person from image 1 on the left, person from image 2 on the right"
            - **Person + Scene**: "Place the person in image 1 into the background from image 2"
            - **Person + Object**: "Have the person from image 1 wearing the outfit from image 2"
            - **Style transfer**: "Apply the artistic style from image 1 to the scene in image 2"
            - **Object combination**: "Merge the furniture from both images into a single room"

            **General Settings:**
            - **Inference steps**: 30-50 is usually enough, more steps = slower but potentially better quality
            - **CFG Scale**: 4.0 is recommended, lower = more creative, higher = follows prompt more strictly
            - **Image size**: Large images are auto-resized to 1024px for optimal performance
            """)

        # Event handlers
        edit_btn.click(
            fn=edit_image,
            inputs=[input_image1, input_image2, input_image3, prompt, num_steps, cfg_scale, seed],
            outputs=[output_image, info_box, gpu_memory]
        )

        clear_btn.click(
            fn=lambda: (None, None, None, "", "", get_gpu_memory()),
            outputs=[input_image1, input_image2, input_image3, prompt, info_box, gpu_memory]
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
    print("Qwen Image Edit 2509 Gradio Demo (Multi-Image Support)")
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
