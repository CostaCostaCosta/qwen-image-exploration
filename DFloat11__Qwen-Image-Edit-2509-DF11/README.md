---
base_model:
  - Qwen/Qwen-Image-Edit-2509
base_model_relation: quantized
tags:
- dfloat11
- df11
- lossless compression
- 70% size, 100% accuracy
---

# DFloat11 Compressed Model: `Qwen/Qwen-Image-Edit-2509`

This is a **DFloat11 losslessly compressed** version of the original `Qwen/Qwen-Image-Edit-2509` model. It reduces model size by **32%** compared to the original BFloat16 model, while maintaining **bit-identical outputs** and supporting **efficient GPU inference**.

🔥🔥🔥 Thanks to DFloat11 compression, Qwen-Image-Edit-2509 can now run on **a single 32GB GPU**, or on **a single 24GB GPU with CPU offloading**, while maintaining full model quality. 🔥🔥🔥

### 📊 Performance Comparison

| Model                                               | Model Size | Peak GPU Memory (1024x1024 image generation) | Image Editing Time (A100 GPU) |
|-----------------------------------------------------|------------|----------------------------------------------|-------------------------------|
| Qwen-Image-Edit-2509 (BFloat16)                     | ~41 GB     | OOM                                          | -                             |
| Qwen-Image-Edit-2509 (DFloat11)                     | 28.43 GB   | 30.20 GB                                     | 102 seconds                   |

### 🔧 How to Use

1. Install or upgrade the DFloat11 pip package *(installs the CUDA kernel automatically; requires a CUDA-compatible GPU and PyTorch installed)*:

    ```bash
    pip install -U dfloat11[cuda12]
    ```

2. Install or upgrade diffusers:

    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

3. Save the following code to a Python file `qwen_image_edit.py`:

    ```python
    import os
    import torch
    import argparse
    from diffusers import QwenImageEditPlusPipeline
    from diffusers.utils import load_image
    from dfloat11 import DFloat11Model

    parser = argparse.ArgumentParser(description="Qwen Image Edit with DFloat11")
    parser.add_argument("--image", default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png", help="Image URL or path")
    parser.add_argument("--prompt", default="Make this cat an astronaut gazing at planet earth from space", help="Edit prompt")
    parser.add_argument("--output", default="qwen_image_edit_output.png", help="Output image path")
    parser.add_argument("--steps", type=int, default=40, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0, help="True CFG scale")
    parser.add_argument("--negative_prompt", default=" ", help="Negative prompt")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--cpu_offload", action="store_true", help="Enable CPU offloading")
    parser.add_argument("--cpu_offload_blocks", type=int, default=20, help="Number of blocks to offload to CPU for block swapping")
    parser.add_argument("--cpu_offload_no_pin_memory", action="store_true", help="Disable memory pinning for CPU offloading")
    args = parser.parse_args()

    pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
    DFloat11Model.from_pretrained(
        "DFloat11/Qwen-Image-Edit-2509-DF11",
        bfloat16_model=pipeline.transformer,
        device="cpu",
        cpu_offload=args.cpu_offload,
        cpu_offload_blocks=args.cpu_offload_blocks,
        pin_memory=not args.cpu_offload_no_pin_memory,
    )
    pipeline.enable_model_cpu_offload()

    image = load_image(args.image)
    inputs = {
        "image": [image],
        "prompt": args.prompt,
        "generator": torch.manual_seed(args.seed),
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "num_images_per_prompt": 1,
    }
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
        output_image.save(args.output)
        print("Image saved at", os.path.abspath(args.output))

    max_memory = torch.cuda.max_memory_allocated()
    print(f"Max memory: {max_memory / (1000 ** 3):.2f} GB")
    ```

4. To run without CPU offloading (32GB VRAM required):
    ```bash
    python qwen_image_edit.py
    ```

    To run with CPU offloading (24GB VRAM required):
    ```bash
    python qwen_image_edit.py --cpu_offload
    ```

    If you are getting out-of-CPU-memory errors, try limiting the number of offloaded blocks or disabling memory-pinning:
    ```bash
    # Offload only 16 blocks (offloading more blocks uses less GPU memory and more CPU memory; offloading less blocks is faster):
    python qwen_image_edit.py --cpu_offload --cpu_offload_blocks 16

    # Disable memory-pinning (the most memory efficient way, but could be slower):
    python qwen_image_edit.py --cpu_offload --no_pin_memory
    ```


### 🔍 How It Works

We apply **Huffman coding** to losslessly compress the exponent bits of BFloat16 model weights, which are highly compressible (their 8 bits carry only ~2.6 bits of actual information). To enable fast inference, we implement a highly efficient CUDA kernel that performs on-the-fly weight decompression directly on the GPU.

The result is a model that is **~32% smaller**, delivers **bit-identical outputs**, and achieves performance **comparable to the original** BFloat16 model.

Learn more in our [research paper](https://arxiv.org/abs/2504.11651).

### 📄 Learn More

* **Paper**: [70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float](https://arxiv.org/abs/2504.11651)
* **GitHub**: [https://github.com/LeanModels/DFloat11](https://github.com/LeanModels/DFloat11)
* **HuggingFace**: [https://huggingface.co/DFloat11](https://huggingface.co/DFloat11)
