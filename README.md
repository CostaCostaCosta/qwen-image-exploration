# Qwen Image Edit Gradio Demo

A local GPU-accelerated Gradio demo for [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) running on RTX 5090 with Blackwell architecture.

## Features

- 🎨 **Interactive Image Editing**: Upload images and edit them with natural language prompts
- 🚀 **GPU Accelerated**: Optimized for RTX 5090 with CUDA 12.8 and BF16 precision
- 💾 **Memory Optimized**: DFloat11 compression + CPU offloading for 32GB VRAM cards
- 📊 **Real-time GPU Monitoring**: Track VRAM usage during inference
- 🎛️ **Adjustable Parameters**: Control inference steps and CFG scale
- 📜 **Edit History**: View your last 10 edits in a gallery
- 🎯 **Preset Prompts**: Quick-start with example editing prompts
- 📏 **Auto Image Resize**: Automatically resize large images for optimal performance
- 🌓 **Modern UI**: Clean Gradio 5 interface with dark mode support

## Hardware Requirements

- **GPU**: NVIDIA RTX 5090 or any GPU with 24GB+ VRAM and CUDA support
- **VRAM**: 24GB minimum (with DFloat11 + CPU offloading), 32GB recommended
- **CUDA**: 12.8 or higher
- **Driver**: NVIDIA driver 580.65.06 or newer

## Software Requirements

- **Python**: 3.12+
- **uv**: Package manager (installed)
- **PyTorch**: 2.7.0 with CUDA 12.8
- **Diffusers**: Latest from GitHub
- **Gradio**: 5.0+

## Installation

### 1. Clone or navigate to this repository

```bash
cd /home/eddie/repos/qwen-image-exploration
```

### 2. Create virtual environment and install dependencies with uv

```bash
# Install all dependencies (this will take a few minutes)
uv sync

# If you need to add diffusers manually (should be automatic):
uv pip install git+https://github.com/huggingface/diffusers
```

**Note**: The first run will download the Qwen-Image-Edit model (~10GB) from Hugging Face.

### 3. (Optional) Login to Hugging Face

If the model is gated or you want to use your HF token:

```bash
uv run huggingface-cli login
```

## Usage

### Start the Gradio Interface

```bash
uv run python app.py
```

The interface will launch at: **http://localhost:7860**

### Using the Demo

1. **Upload an Image**: Click the upload area or paste from clipboard
2. **Enter a Prompt**: Describe the edits you want (e.g., "turn this cat into a dog")
3. **Optional**: Choose a preset prompt from the dropdown
4. **Adjust Settings** (optional):
   - **Inference Steps**: 20-100 (default: 50) - more steps = better quality but slower
   - **CFG Scale**: 1.0-10.0 (default: 4.0) - how strictly to follow the prompt
   - **Seed**: Set for reproducible results (-1 for random)
5. **Click "Edit Image"**: Wait for processing (typically 10-30 seconds)
6. **Download**: Right-click the output image to save

### Example Prompts

- `turn this cat into a dog`
- `make it look like a watercolor painting`
- `change season to winter with snow`
- `make it nighttime with stars`
- `add sunglasses`
- `make it cyberpunk style`
- `turn into anime style`
- `change to autumn with fall colors`

## Quality of Life Features

### 1. **Image Comparison**
- View original and edited images side-by-side

### 2. **Edit History**
- Automatically saves your last 10 edits
- Click "Refresh History" to update the gallery
- Click "Clear History" to free up memory

### 3. **GPU Memory Monitoring**
- Real-time VRAM usage display
- Helps prevent out-of-memory errors

### 4. **Preset Prompts**
- Quick-start with 10 example prompts
- Select from dropdown to auto-fill prompt box

### 5. **Auto Image Preprocessing**
- Large images automatically resized to 1024px
- Maintains aspect ratio
- Improves inference speed

### 6. **Progress Tracking**
- Visual progress bar during inference
- Shows current step and ETA

### 7. **Reproducible Results**
- Set seed for consistent outputs
- Seed displayed after each edit

### 8. **Advanced Settings**
- Fine-tune inference parameters
- Collapse/expand advanced options

## Troubleshooting

### Out of Memory Errors

If you encounter CUDA out of memory errors:

1. Reduce inference steps to 30
2. Ensure large images are being resized (check console output)
3. Clear history to free VRAM: Click "Clear History"
4. Close other GPU-intensive applications

### Model Download Issues

If model download fails:

```bash
# Pre-download the model
uv run python -c "from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('Qwen/Qwen-Image-Edit')"
```

### Pipeline Loading Slow

First load downloads the model (~10GB). Subsequent launches are faster as the model is cached in `~/.cache/huggingface/`.

### CUDA Not Available

Check your installation:

```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

Expected output:
```
CUDA available: True
CUDA version: 12.8
```

## Performance Notes

- **First inference**: Slower due to model compilation (~30-60 seconds)
- **Subsequent inferences**: Much faster (~10-20 seconds with 50 steps)
- **Batch processing**: Process multiple images with the same prompt by running edits sequentially
- **Memory usage**:
  - Peak VRAM: ~22-28GB with DFloat11 + CPU offloading
  - Original model would require ~41GB (OOM on 32GB cards)
  - CPU offloading adds slight latency but enables 32GB card usage

## Configuration

### Enable Public Sharing

Edit `app.py` line 449:

```python
demo.launch(
    share=True,  # Change to True for public link
    ...
)
```

### Change Port

Edit `app.py` line 448:

```python
demo.launch(
    server_port=8080,  # Change port number
    ...
)
```

### Adjust Maximum Image Size

Edit `app.py` line 45:

```python
def preprocess_image(image, max_size=1024):  # Change max_size
```

## Technical Details

- **Model**: DFloat11/Qwen-Image-Edit-DF11 (32% compressed, lossless)
- **Compression**: DFloat11 Huffman-coded BFloat16 exponents
- **Precision**: BFloat16 (BF16)
- **Memory Optimizations**:
  - CPU offloading (moves inactive model parts to CPU)
  - Attention slicing (reduces peak memory during inference)
- **Framework**: Diffusers + PyTorch 2.7.0
- **Architecture Support**: CUDA compute capability 12.0 (sm_120)
- **Inference Engine**: CUDA 12.8 on Blackwell

## Credits

- **Model**: [Qwen Team](https://huggingface.co/Qwen)
- **Compression**: [DFloat11](https://huggingface.co/DFloat11/Qwen-Image-Edit-DF11)
- **Framework**: [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- **Interface**: [Gradio](https://www.gradio.app/)

## License

- **Code**: MIT License
- **Model**: [Qwen-Image-Edit Apache 2.0](https://huggingface.co/Qwen/Qwen-Image-Edit)

## Support

For issues related to:
- **Model**: Check [Qwen-Image-Edit discussions](https://huggingface.co/Qwen/Qwen-Image-Edit/discussions)
- **Installation**: Verify PyTorch and CUDA versions
- **Performance**: Ensure GPU drivers are up to date

---

**Enjoy editing with Qwen Image Edit! 🎨✨**
