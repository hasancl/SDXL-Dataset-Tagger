import gradio as gr
import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
import numpy as np
import timm
from huggingface_hub import hf_hub_download
import cv2

# Models Directory
MODELS_DIR = os.path.join(os.getcwd(), "models")
WD14_DIR = os.path.join(MODELS_DIR, "wd14")
BLIP_DIR = os.path.join(MODELS_DIR, "blip")

# Global models to avoid reloading
blip_processor = None
blip_model = None
wd14_model = None
wd14_tags = None
wd14_transform = None
wd14_failed = False # Flag to prevent retry loop

def check_and_download_models():
    os.makedirs(WD14_DIR, exist_ok=True)
    os.makedirs(BLIP_DIR, exist_ok=True)
    
    print(f"Checking models in {MODELS_DIR}...")
    
    # --- WD14 Download ---
    # Attempt to download SwinV2 version which is known to work well with PyTorch/Timm
    wd14_repo = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
    
    try:
        if not os.path.exists(os.path.join(WD14_DIR, "pytorch_model.bin")) and not os.path.exists(os.path.join(WD14_DIR, "model.safetensors")): 
             print(f"Downloading WD14 model from {wd14_repo}...")
             try:
                 hf_hub_download(repo_id=wd14_repo, filename="model.safetensors", local_dir=WD14_DIR)
             except:
                 hf_hub_download(repo_id=wd14_repo, filename="pytorch_model.bin", local_dir=WD14_DIR)
             
             hf_hub_download(repo_id=wd14_repo, filename="selected_tags.csv", local_dir=WD14_DIR)
             hf_hub_download(repo_id=wd14_repo, filename="config.json", local_dir=WD14_DIR)
    except Exception as e:
        print(f"WD14 Download Warning: {e}. attempting fallback or proceed if files exist.")

    # --- BLIP Download ---
    print("BLIP model download is handled internally by transformers.")

def load_blip2():
    global blip_processor, blip_model
    if blip_model is None:
        print("Loading BLIP model (Lite version for CPU)...")
        model_id = "Salesforce/blip-image-captioning-large" 
        
        processor = BlipProcessor.from_pretrained(model_id, cache_dir=BLIP_DIR)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        model = BlipForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, cache_dir=BLIP_DIR
        )
        model.to(device)
        blip_processor = processor
        blip_model = model
        print("BLIP model loaded.")
    return blip_processor, blip_model

def load_wd14():
    global wd14_model, wd14_tags, wd14_transform, wd14_failed
    
    if wd14_failed:
        return None, None, None

    if wd14_model is None:
        check_and_download_models() 
        try:
            # Revert to SwinV2 as it's the only one with PyTorch weights on HF (SmilingWolf)
            repo_id = "hf_hub:SmilingWolf/wd-v1-4-swinv2-tagger-v2"
            
            # SwinV2 requires 448x448 to match window sizes.
            # We must force both the model creation AND the data transform to use 448.
            
            wd14_model = timm.create_model(repo_id, pretrained=False, img_size=448)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model_file = os.path.join(WD14_DIR, "model.safetensors")
            if not os.path.exists(model_file):
                 model_file = os.path.join(WD14_DIR, "pytorch_model.bin")
            
            if os.path.exists(model_file):
                print(f"Loading local weights from {model_file}...")
                if model_file.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    wd14_model.load_state_dict(load_file(model_file))
                else:
                    wd14_model.load_state_dict(torch.load(model_file, map_location="cpu"))
            else:
                print("Local weights missing, attempting timm auto-download...")
                wd14_model = timm.create_model(repo_id, pretrained=True, img_size=448)
            
            # Optimization for RTX 3060
            if device == "cuda":
                wd14_model = wd14_model.half() # Use FP16 for speed/memory efficiency on GPU
            
            wd14_model.eval()
            wd14_model.to(device)
            
            from timm.data import create_transform
            
            # Manually define transform to ensure 448x448 despite architecture defaults
            # Config found in models/wd14/config.json: 
            # input_size=[3, 448, 448], mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], interpolation='bicubic'
            
            wd14_transform = create_transform(
                input_size=(3, 448, 448),
                is_training=False,
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                interpolation='bicubic',
                crop_pct=1.0 # Use full image
            )
            print(f"WD14 loaded successfully on {device}.")
            
            # Tags
            tags_path = os.path.join(WD14_DIR, "selected_tags.csv")
            if not os.path.exists(tags_path):
                 hf_hub_download(repo_id="SmilingWolf/wd-v1-4-swinv2-tagger-v2", filename="selected_tags.csv", local_dir=WD14_DIR)
                 
            df = pd.read_csv(tags_path)
            wd14_tags = df['name'].tolist()
            
        except Exception as e:
            print(f"Failed to load WD14: {e}")
            wd14_failed = True # Stop retrying
            wd14_model = None

    return wd14_model, wd14_tags, wd14_transform

def run_blip2(image_path):
    try:
        processor, model = load_blip2()
        image = Image.open(image_path).convert('RGB')
        device = model.device
        dtype = model.dtype # This will be float16 on GPU
        
        inputs = processor(images=image, return_tensors="pt").to(device, dtype=dtype)
        
        generated_ids = model.generate(**inputs, max_new_tokens=50) # Added max_new_tokens for safety
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        base_path = os.path.splitext(image_path)[0]
        with open(f"{base_path}.caption", "w", encoding="utf-8") as f:
            f.write(caption)
        
        return caption
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ""

def run_wd14(image_path, threshold=0.35):
    try:
        model, tags, transform = load_wd14()
        if model is None:
            return ""

        image = Image.open(image_path).convert('RGB')
        
        # Get model precision (e.g. half/float16 or float32)
        param = next(model.parameters())
        device = param.device
        dtype = param.dtype
        
        input_tensor = transform(image).unsqueeze(0).to(device, dtype=dtype)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
        active_tags = []
        for i, p in enumerate(probs[4:], 4):
            if p >= threshold:
                if i < len(tags):
                    active_tags.append(tags[i])
        
        tag_string = ", ".join(active_tags)
        
        base_path = os.path.splitext(image_path)[0]
        # Use .wd14 for raw tags to avoid mixing with final .txt result
        with open(f"{base_path}.wd14", "w", encoding="utf-8") as f:
            f.write(tag_string)
            
        return tag_string
        
    except Exception as e:
        print(f"Error w/ WD14 {image_path}: {e}")
        return ""

def process_folder(folder_path, run_wd14_bool, run_blip_bool, progress=gr.Progress()):
    if not os.path.exists(folder_path):
        return "Folder not found."
    
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    logs = []
    
    for i, file in enumerate(progress.tqdm(files)):
        full_path = os.path.join(folder_path, file)
        log_entry = f"Processing {file}..."
        
        if run_wd14_bool:
            tags = run_wd14(full_path)
            log_entry += " [WD14 Done]"
        
        if run_blip_bool:
            caption = run_blip2(full_path)
            log_entry += " [BLIP2 Done]"
            
        logs_out = "\n".join(logs[-10:])
        yield "\n".join(logs) # Yield primarily for better gradio updates
        
    return "\n".join(logs)

# --- Curation Logic ---
current_files = []
current_index = 0
current_folder = ""

def load_curation_folder(folder_path):
    global current_files, current_index, current_folder
    if not os.path.exists(folder_path):
        return gr.update(value=None), "", "", "", "Folder not found"
    
    current_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    current_files.sort()
    current_index = 0
    current_folder = folder_path
    
    if not current_files:
        return gr.update(value=None), "", "", "", "No images found.", gr.update(maximum=0, value=0)
        
    img, wd14, blip, final, status, _ = load_image_data(0)
    return img, wd14, blip, final, status, gr.update(maximum=len(current_files)-1, value=0, visible=True)

def load_image_data(index):
    global current_files, current_folder, current_index
    current_index = index
    
    if index < 0 or index >= len(current_files):
        return None, "", "", "", f"Index {index} out of bounds"
        
    filename = current_files[index]
    filepath = os.path.join(current_folder, filename)
    basepath = os.path.splitext(filepath)[0]
    
    img = Image.open(filepath)
    
    wd14_content = ""
    if os.path.exists(f"{basepath}.wd14"):
        with open(f"{basepath}.wd14", "r", encoding="utf-8") as f:
            wd14_content = f.read()
    
    caption_content = ""
    if os.path.exists(f"{basepath}.caption"):
        with open(f"{basepath}.caption", "r", encoding="utf-8") as f:
            caption_content = f.read()

    # Priority 1: load existing .txt (final prompt)
    final_content = ""
    if os.path.exists(f"{basepath}.txt"):
        with open(f"{basepath}.txt", "r", encoding="utf-8") as f:
            final_content = f.read()

    # Priority 2: if .txt doesn't exist, auto-merge from metadata
    if not final_content.strip():
        parts = []
        if caption_content.strip():
            parts.append(caption_content.strip())
        if wd14_content.strip():
            parts.append(wd14_content.strip())
        final_content = ", ".join(parts)
    
    status = f"Image {index + 1}/{len(current_files)}: {filename}"
    return img, wd14_content, caption_content, final_content, status, index

def next_image():
    global current_index, current_files
    new_index = current_index + 1
    if new_index >= len(current_files):
        new_index = len(current_files) - 1
    return load_image_data(new_index)

def prev_image():
    global current_index
    new_index = current_index - 1
    if new_index < 0:
        new_index = 0
    return load_image_data(new_index)

def save_and_next(final_text, delete_blip):
    global current_index, current_files, current_folder
    
    # Check if we are on the last image and already processed
    is_last = current_index >= len(current_files) - 1
    
    if current_files and 0 <= current_index < len(current_files):
        filename = current_files[current_index]
        filepath = os.path.join(current_folder, filename)
        basepath = os.path.splitext(filepath)[0]
        
        # Only save if there's actual content
        if final_text.strip():
            with open(f"{basepath}.txt", "w", encoding="utf-8") as f:
                f.write(final_text)
            
            if delete_blip:
                for ext in [".caption", ".wd14"]:
                    meta_file = f"{basepath}{ext}"
                    if os.path.exists(meta_file):
                        os.remove(meta_file)
    
    # If last image, return "Completed" status
    if is_last:
        img, wd14, blip, final, status, idx = load_image_data(current_index)
        return img, wd14, blip, final, "✅ **Completed!** All images have been processed.", idx
                
    return next_image()

def append_tags(existing, new_tags):
    if not existing:
        return new_tags
    return existing.rstrip().rstrip(',') + ", " + new_tags

def update_final_prompt(caption, tags):
    parts = []
    if caption and caption.strip():
        parts.append(caption.strip())
    if tags and tags.strip():
        parts.append(tags.strip())
    return ", ".join(parts)

def auto_merge_all(folder_path, delete_metadata, progress=gr.Progress()):
    """Automatically merge BLIP and WD14 for all images in folder."""
    if not os.path.exists(folder_path):
        yield "Folder not found."
        return
    
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    files.sort()
    
    if not files:
        yield "No images found."
        return
    
    processed = 0
    for i, file in enumerate(progress.tqdm(files)):
        filepath = os.path.join(folder_path, file)
        basepath = os.path.splitext(filepath)[0]
        
        # Read metadata
        wd14_content = ""
        if os.path.exists(f"{basepath}.wd14"):
            with open(f"{basepath}.wd14", "r", encoding="utf-8") as f:
                wd14_content = f.read()
        
        caption_content = ""
        if os.path.exists(f"{basepath}.caption"):
            with open(f"{basepath}.caption", "r", encoding="utf-8") as f:
                caption_content = f.read()
        
        # Merge: BLIP first, then WD14
        parts = []
        if caption_content.strip():
            parts.append(caption_content.strip())
        if wd14_content.strip():
            parts.append(wd14_content.strip())
        
        final_prompt = ", ".join(parts)
        
        # Save final prompt
        if final_prompt.strip():
            with open(f"{basepath}.txt", "w", encoding="utf-8") as f:
                f.write(final_prompt)
            
            # Delete metadata if requested
            if delete_metadata:
                for ext in [".caption", ".wd14"]:
                    meta_file = f"{basepath}{ext}"
                    if os.path.exists(meta_file):
                        os.remove(meta_file)
            
            processed += 1
        
        yield f"Processing... {i+1}/{len(files)} - {file}"
    
    yield f"✅ **Completed!** Merged {processed}/{len(files)} images."

with gr.Blocks(title="Dataset Tagging Assistant") as demo:
    gr.Markdown("# Dataset Tagging Assistant (WD14 + BLIP2)")
    
    with gr.Tab("1. Auto Tagging"):
        with gr.Row():
            folder_input = gr.Textbox(label="Images Folder Path", value="images", placeholder="C:/path/to/img")
            btn_process = gr.Button("Start Batch Processing", variant="primary")
        
        with gr.Row():
            chk_wd14 = gr.Checkbox(label="Run WD14 Tagger", value=True)
            chk_blip = gr.Checkbox(label="Run BLIP-2 Captioning", value=True)
            
        logs_out = gr.Textbox(label="Logs", lines=10)
        
        btn_process.click(process_folder, inputs=[folder_input, chk_wd14, chk_blip], outputs=logs_out)
        
    with gr.Tab("2. Curation & Merge"):
        with gr.Row():
            curation_path = gr.Textbox(label="Images Folder Path", value="images")
            btn_load = gr.Button("Load Folder")
            chk_delete_blip = gr.Checkbox(label="Clean up Metadata (.caption & .wd14) after Save", value=True)
        
        status_info = gr.Markdown("Ready")
        image_index_slider = gr.Slider(minimum=0, maximum=100, step=1, label="Jump to Image Index", visible=False)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_display = gr.Image(type="pil", label="Image")
                with gr.Row():
                    btn_prev = gr.Button("<< Prev")
                    btn_next = gr.Button("Next >>")
            
            with gr.Column(scale=2):
                with gr.Row():
                    wd14_box = gr.Textbox(label="WD14 Tags (.txt)", lines=4, interactive=True)
                    blip_box = gr.Textbox(label="BLIP-2 Caption (.caption)", lines=4, interactive=True)
                
                gr.Markdown("### Merge Actions")
                with gr.Row():
                    btn_copy_blip = gr.Button("Copy BLIP to Final")
                    btn_append_wd14 = gr.Button("Append WD14 to Final")
                    
                final_box = gr.Textbox(label="Final Prompt (Saved to .txt)", lines=6, interactive=True)
                btn_save_next = gr.Button("Save & Next Image", variant="primary")
                btn_auto_merge = gr.Button("🔄 Auto Merge All", variant="secondary")
    
        btn_load.click(load_curation_folder, inputs=[curation_path], outputs=[image_display, wd14_box, blip_box, final_box, status_info, image_index_slider])
        btn_prev.click(prev_image, outputs=[image_display, wd14_box, blip_box, final_box, status_info, image_index_slider])
        btn_next.click(next_image, outputs=[image_display, wd14_box, blip_box, final_box, status_info, image_index_slider])
        
        image_index_slider.release(load_image_data, inputs=[image_index_slider], outputs=[image_display, wd14_box, blip_box, final_box, status_info, image_index_slider])

        # Real-time sync: When WD14 or BLIP2 text changes, update the Final Prompt
        wd14_box.change(update_final_prompt, inputs=[blip_box, wd14_box], outputs=[final_box])
        blip_box.change(update_final_prompt, inputs=[blip_box, wd14_box], outputs=[final_box])

        btn_copy_blip.click(lambda x: x, inputs=[blip_box], outputs=[final_box])
        btn_append_wd14.click(append_tags, inputs=[final_box, wd14_box], outputs=[final_box])
        btn_save_next.click(save_and_next, inputs=[final_box, chk_delete_blip], outputs=[image_display, wd14_box, blip_box, final_box, status_info, image_index_slider])
        btn_auto_merge.click(auto_merge_all, inputs=[curation_path, chk_delete_blip], outputs=[status_info])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
