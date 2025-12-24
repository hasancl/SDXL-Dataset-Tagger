# SDXL Dataset Tagging Assistant 🚀

A lightweight, high-performance tool developed to automate and curate image tagging for SDXL (Stable Diffusion XL) fine-tuning. Optimized for NVIDIA GPUs (RTX 3060+) and Ubuntu/Linux environments.

## Features

### 🏷️ Auto Tagging (Tab 1)
- **WD14 Tagger**: High-accuracy booru-style tagging using the `SwinV2` architecture (`wd-v1-4-swinv2-tagger-v2`).
- **BLIP Captioning**: Natural language image description using `blip-image-captioning-large`.
- **GPU Optimization**: Full support for **CUDA** with **FP16 (Half Precision)** to maximize speed and minimize VRAM usage.

### ✏️ Curation & Merge (Tab 2)
- **Real-time Preview**: Side-by-side view of image, WD14 tags, and BLIP caption.
- **Live Sync**: Editing tags or captions instantly updates the Final Prompt.
- **Jump to Image**: Slider to navigate directly to any image in the dataset.
- **🔄 Auto Merge All**: One-click batch merge of all BLIP + WD14 files into final `.txt` prompts.
- **Metadata Cleanup**: Option to automatically delete `.caption` and `.wd14` files after saving, leaving only `image.png` + `image.txt`.
- **Completion Status**: Clear notification when all images have been processed.

### 📁 Local Model Management
- Models are downloaded once and managed locally in a `models/` directory for portability.

## Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA drivers installed (e.g., RTX 3060)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/SDXL-Dataset-Tool.git
   cd SDXL-Dataset-Tool
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

1. Start the tool:
   ```bash
   python3 dataset_tool.py
   ```
2. Open the URL provided in the terminal (usually `http://127.0.0.1:7860`).

### Workflow
1. **Tab 1 (Auto Tagging)**: Provide the folder path and click "Start Batch Processing" to generate `.wd14` and `.caption` files.
2. **Tab 2 (Curation & Merge)**: 
   - Load the folder to review images.
   - Use **Auto Merge All** for quick batch processing, OR
   - Navigate image-by-image with **Save & Next** for manual curation.
   - Enable **Clean up Metadata** to keep your folder tidy (only `.png` + `.txt`).

## Project Structure
- `dataset_tool.py`: Main application logic.
- `requirements.txt`: Python package dependencies.
- `models/`: Local storage for downloaded AI models (created on first run).
- `photos/`: Example/Input folder for your images.

## License
MIT
