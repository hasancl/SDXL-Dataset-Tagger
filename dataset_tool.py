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

# --- Edit Captions Logic ---
edit_files = []
edit_index = 0
edit_folder = ""

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

# --- Edit Captions Functions ---

def load_edit_folder(folder_path):
    """Load folder for caption editing."""
    global edit_files, edit_index, edit_folder
    if not os.path.exists(folder_path):
        return gr.update(value=None), "", "Folder not found", gr.update(maximum=0, value=0, visible=False)
    
    edit_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    edit_files.sort()
    edit_index = 0
    edit_folder = folder_path
    
    if not edit_files:
        return gr.update(value=None), "", "No images found.", gr.update(maximum=0, value=0, visible=False)
    
    return load_edit_image(0)

def load_edit_image(index):
    """Load image and its caption for editing."""
    global edit_files, edit_folder, edit_index
    edit_index = index
    
    if index < 0 or index >= len(edit_files):
        return None, "", f"Index {index} out of bounds", index
    
    filename = edit_files[index]
    filepath = os.path.join(edit_folder, filename)
    basepath = os.path.splitext(filepath)[0]
    
    img = Image.open(filepath)
    
    caption_content = ""
    txt_path = f"{basepath}.txt"
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            caption_content = f.read()
    
    status = f"📝 Image {index + 1}/{len(edit_files)}: {filename}"
    return img, caption_content, status, gr.update(maximum=len(edit_files)-1, value=index, visible=True)

def save_edit_caption(caption_text):
    """Save caption to current image's .txt file."""
    global edit_index, edit_files, edit_folder
    
    if not edit_files or edit_index < 0 or edit_index >= len(edit_files):
        return "❌ No image loaded"
    
    filename = edit_files[edit_index]
    filepath = os.path.join(edit_folder, filename)
    basepath = os.path.splitext(filepath)[0]
    
    with open(f"{basepath}.txt", "w", encoding="utf-8") as f:
        f.write(caption_text)
    
    return f"✅ Saved: {filename}"

def edit_next_image():
    """Navigate to next image in edit mode."""
    global edit_index, edit_files
    new_index = edit_index + 1
    if new_index >= len(edit_files):
        new_index = len(edit_files) - 1
    return load_edit_image(new_index)

def edit_prev_image():
    """Navigate to previous image in edit mode."""
    global edit_index
    new_index = edit_index - 1
    if new_index < 0:
        new_index = 0
    return load_edit_image(new_index)

def search_replace_current(caption, find_text, replace_text, case_sensitive):
    """Apply search and replace to current caption."""
    if not find_text:
        return caption, "⚠️ Find text is empty"
    
    if case_sensitive:
        count = caption.count(find_text)
        new_caption = caption.replace(find_text, replace_text)
    else:
        import re
        pattern = re.compile(re.escape(find_text), re.IGNORECASE)
        count = len(pattern.findall(caption))
        new_caption = pattern.sub(replace_text, caption)
    
    return new_caption, f"✅ Replaced {count} occurrence(s) in current caption"

def search_replace_all(folder_path, find_text, replace_text, case_sensitive, progress=gr.Progress()):
    """Apply search and replace to all caption files in folder."""
    if not find_text:
        yield "⚠️ Find text is empty"
        return
    
    if not os.path.exists(folder_path):
        yield "❌ Folder not found"
        return
    
    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
    
    if not txt_files:
        yield "⚠️ No .txt files found"
        return
    
    import re
    total_replacements = 0
    files_modified = 0
    
    for i, txt_file in enumerate(progress.tqdm(txt_files)):
        filepath = os.path.join(folder_path, txt_file)
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        if case_sensitive:
            count = content.count(find_text)
            new_content = content.replace(find_text, replace_text)
        else:
            pattern = re.compile(re.escape(find_text), re.IGNORECASE)
            count = len(pattern.findall(content))
            new_content = pattern.sub(replace_text, content)
        
        if count > 0:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            total_replacements += count
            files_modified += 1
        
        yield f"Processing... {i+1}/{len(txt_files)}"
    
    yield f"✅ **Completed!** Replaced {total_replacements} occurrence(s) in {files_modified} file(s)"

def add_prefix_all(folder_path, prefix, progress=gr.Progress()):
    """Add prefix to all caption files."""
    if not prefix:
        yield "⚠️ Prefix is empty"
        return
    
    if not os.path.exists(folder_path):
        yield "❌ Folder not found"
        return
    
    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
    
    for i, txt_file in enumerate(progress.tqdm(txt_files)):
        filepath = os.path.join(folder_path, txt_file)
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Avoid adding prefix if already present
        if not content.startswith(prefix):
            new_content = prefix + content
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
        
        yield f"Processing... {i+1}/{len(txt_files)}"
    
    yield f"✅ **Completed!** Added prefix to {len(txt_files)} file(s)"

def add_suffix_all(folder_path, suffix, progress=gr.Progress()):
    """Add suffix to all caption files."""
    if not suffix:
        yield "⚠️ Suffix is empty"
        return
    
    if not os.path.exists(folder_path):
        yield "❌ Folder not found"
        return
    
    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
    
    for i, txt_file in enumerate(progress.tqdm(txt_files)):
        filepath = os.path.join(folder_path, txt_file)
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Avoid adding suffix if already present
        if not content.rstrip().endswith(suffix.rstrip()):
            new_content = content.rstrip() + suffix
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
        
        yield f"Processing... {i+1}/{len(txt_files)}"
    
    yield f"✅ **Completed!** Added suffix to {len(txt_files)} file(s)"

def remove_tag_all(folder_path, tag_to_remove, progress=gr.Progress()):
    """Remove specific tag from all caption files."""
    if not tag_to_remove:
        yield "⚠️ Tag is empty"
        return
    
    if not os.path.exists(folder_path):
        yield "❌ Folder not found"
        return
    
    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
    total_removed = 0
    
    for i, txt_file in enumerate(progress.tqdm(txt_files)):
        filepath = os.path.join(folder_path, txt_file)
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split by comma, clean and filter
        tags = [t.strip() for t in content.split(",")]
        original_count = len(tags)
        tags = [t for t in tags if t.lower() != tag_to_remove.lower().strip()]
        
        if len(tags) < original_count:
            new_content = ", ".join(tags)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            total_removed += (original_count - len(tags))
        
        yield f"Processing... {i+1}/{len(txt_files)}"
    
    yield f"✅ **Completed!** Removed {total_removed} occurrence(s) of '{tag_to_remove}'"

def remove_duplicates_all(folder_path, progress=gr.Progress()):
    """Remove duplicate tags from all caption files."""
    if not os.path.exists(folder_path):
        yield "❌ Folder not found"
        return
    
    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
    total_duplicates = 0
    
    for i, txt_file in enumerate(progress.tqdm(txt_files)):
        filepath = os.path.join(folder_path, txt_file)
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split, clean, and remove duplicates while preserving order
        tags = [t.strip() for t in content.split(",") if t.strip()]
        seen = set()
        unique_tags = []
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower not in seen:
                seen.add(tag_lower)
                unique_tags.append(tag)
        
        duplicates_found = len(tags) - len(unique_tags)
        if duplicates_found > 0:
            new_content = ", ".join(unique_tags)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            total_duplicates += duplicates_found
        
        yield f"Processing... {i+1}/{len(txt_files)}"
    
    yield f"✅ **Completed!** Removed {total_duplicates} duplicate tag(s)"

def analyze_tag_frequency(folder_path):
    """Analyze tag frequency across all caption files."""
    if not os.path.exists(folder_path):
        return "❌ Folder not found"
    
    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
    
    if not txt_files:
        return "⚠️ No .txt files found"
    
    from collections import Counter
    tag_counter = Counter()
    
    for txt_file in txt_files:
        filepath = os.path.join(folder_path, txt_file)
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        tags = [t.strip().lower() for t in content.split(",") if t.strip()]
        tag_counter.update(tags)
    
    # Format results
    result = "## 📊 Tag Frequency Analysis\n\n"
    result += f"**Total unique tags:** {len(tag_counter)}\n\n"
    
    result += "### 🔝 Top 50 Most Common Tags\n"
    result += "| Rank | Tag | Count |\n|------|-----|-------|\n"
    for i, (tag, count) in enumerate(tag_counter.most_common(50), 1):
        result += f"| {i} | {tag} | {count} |\n"
    
    # Rare tags (only appear once)
    rare_tags = [tag for tag, count in tag_counter.items() if count == 1]
    result += f"\n### ⚠️ Rare Tags (appears only once): {len(rare_tags)}\n"
    if rare_tags[:20]:
        result += ", ".join(rare_tags[:20])
        if len(rare_tags) > 20:
            result += f" ... and {len(rare_tags) - 20} more"
    
    return result

def get_caption_statistics(folder_path):
    """Get statistics about captions in folder."""
    if not os.path.exists(folder_path):
        return "❌ Folder not found"
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
    
    # Find images without captions
    image_bases = set(os.path.splitext(f)[0] for f in image_files)
    txt_bases = set(os.path.splitext(f)[0] for f in txt_files)
    missing_captions = image_bases - txt_bases
    
    # Analyze caption lengths
    tag_counts = []
    char_counts = []
    shortest = ("", float('inf'))
    longest = ("", 0)
    empty_captions = []
    
    for txt_file in txt_files:
        filepath = os.path.join(folder_path, txt_file)
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        char_count = len(content)
        tags = [t.strip() for t in content.split(",") if t.strip()]
        tag_count = len(tags)
        
        tag_counts.append(tag_count)
        char_counts.append(char_count)
        
        if char_count == 0:
            empty_captions.append(txt_file)
        
        if char_count < shortest[1] and char_count > 0:
            shortest = (txt_file, char_count)
        if char_count > longest[1]:
            longest = (txt_file, char_count)
    
    avg_tags = sum(tag_counts) / len(tag_counts) if tag_counts else 0
    avg_chars = sum(char_counts) / len(char_counts) if char_counts else 0
    
    result = "## 📈 Caption Statistics\n\n"
    result += f"| Metric | Value |\n|--------|-------|\n"
    result += f"| Total Images | {len(image_files)} |\n"
    result += f"| Total Caption Files | {len(txt_files)} |\n"
    result += f"| Images without Captions | {len(missing_captions)} |\n"
    result += f"| Empty Caption Files | {len(empty_captions)} |\n"
    result += f"| Average Tags per Caption | {avg_tags:.1f} |\n"
    result += f"| Average Characters per Caption | {avg_chars:.1f} |\n"
    
    if shortest[0]:
        result += f"| Shortest Caption | {shortest[0]} ({shortest[1]} chars) |\n"
    if longest[0]:
        result += f"| Longest Caption | {longest[0]} ({longest[1]} chars) |\n"
    
    if missing_captions:
        result += f"\n### ⚠️ Images without captions ({len(missing_captions)}):\n"
        result += ", ".join(list(missing_captions)[:10])
        if len(missing_captions) > 10:
            result += f" ... and {len(missing_captions) - 10} more"
    
    if empty_captions:
        result += f"\n\n### ❌ Empty caption files ({len(empty_captions)}):\n"
        result += ", ".join(empty_captions[:10])
        if len(empty_captions) > 10:
            result += f" ... and {len(empty_captions) - 10} more"
    
    return result

def quality_check(folder_path):
    """Check quality issues in captions."""
    if not os.path.exists(folder_path):
        return "❌ Folder not found"
    
    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
    
    if not txt_files:
        return "⚠️ No .txt files found"
    
    issues = {
        "empty": [],
        "too_short": [],  # < 10 chars
        "too_long": [],   # > 500 chars
        "double_commas": [],
        "trailing_comma": [],
        "double_spaces": [],
        "leading_trailing_spaces": []
    }
    
    for txt_file in txt_files:
        filepath = os.path.join(folder_path, txt_file)
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        if len(content) == 0:
            issues["empty"].append(txt_file)
        elif len(content) < 10:
            issues["too_short"].append(txt_file)
        elif len(content) > 500:
            issues["too_long"].append(txt_file)
        
        if ",," in content:
            issues["double_commas"].append(txt_file)
        if content.rstrip().endswith(","):
            issues["trailing_comma"].append(txt_file)
        if "  " in content:
            issues["double_spaces"].append(txt_file)
        if content != content.strip():
            issues["leading_trailing_spaces"].append(txt_file)
    
    result = "## 🔍 Quality Check Report\n\n"
    
    total_issues = sum(len(v) for v in issues.values())
    if total_issues == 0:
        result += "✅ **No issues found!** All captions look good.\n"
        return result
    
    result += f"⚠️ **Found {total_issues} issue(s)**\n\n"
    
    issue_labels = {
        "empty": "❌ Empty captions",
        "too_short": "⚠️ Too short (< 10 chars)",
        "too_long": "⚠️ Too long (> 500 chars)",
        "double_commas": "🔧 Double commas (,,)",
        "trailing_comma": "🔧 Trailing comma",
        "double_spaces": "🔧 Double spaces",
        "leading_trailing_spaces": "🔧 Leading/trailing whitespace"
    }
    
    for key, label in issue_labels.items():
        if issues[key]:
            result += f"### {label}: {len(issues[key])}\n"
            result += ", ".join(issues[key][:5])
            if len(issues[key]) > 5:
                result += f" ... and {len(issues[key]) - 5} more"
            result += "\n\n"
    
    return result

def fix_common_issues(folder_path, progress=gr.Progress()):
    """Fix common issues in captions (double commas, trailing commas, double spaces, whitespace)."""
    if not os.path.exists(folder_path):
        yield "❌ Folder not found"
        return
    
    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
    fixed_count = 0
    
    for i, txt_file in enumerate(progress.tqdm(txt_files)):
        filepath = os.path.join(folder_path, txt_file)
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        original = content
        
        # Fix double spaces
        while "  " in content:
            content = content.replace("  ", " ")
        
        # Fix double commas
        while ",," in content:
            content = content.replace(",,", ",")
        
        # Fix ", ," pattern
        while ", ," in content:
            content = content.replace(", ,", ",")
        
        # Strip whitespace
        content = content.strip()
        
        # Remove trailing comma
        content = content.rstrip(",").strip()
        
        # Clean up tags
        tags = [t.strip() for t in content.split(",") if t.strip()]
        content = ", ".join(tags)
        
        if content != original:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            fixed_count += 1
        
        yield f"Processing... {i+1}/{len(txt_files)}"
    
    yield f"✅ **Completed!** Fixed issues in {fixed_count} file(s)"

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

    with gr.Tab("3. Edit Captions"):
        gr.Markdown("### ✏️ Caption Editor & Bulk Operations")
        
        with gr.Row():
            edit_folder_input = gr.Textbox(label="Images Folder Path", value="images", scale=3)
            btn_edit_load = gr.Button("📂 Load Folder", variant="primary")
        
        edit_status = gr.Markdown("Ready")
        edit_slider = gr.Slider(minimum=0, maximum=100, step=1, label="Jump to Image", visible=False)
        
        with gr.Row():
            with gr.Column(scale=1):
                edit_image_display = gr.Image(type="pil", label="Image Preview", height=400)
                with gr.Row():
                    btn_edit_prev = gr.Button("⬅️ Prev")
                    btn_edit_next = gr.Button("Next ➡️")
            
            with gr.Column(scale=2):
                edit_caption_box = gr.Textbox(label="Caption (.txt)", lines=6, interactive=True)
                with gr.Row():
                    btn_save_caption = gr.Button("💾 Save Caption", variant="primary")
                    save_result = gr.Markdown("")
                
                gr.Markdown("---")
                gr.Markdown("### 🔍 Search & Replace")
                
                with gr.Row():
                    find_text = gr.Textbox(label="Find", placeholder="araffe woman", scale=2)
                    replace_text = gr.Textbox(label="Replace with", placeholder="asian woman", scale=2)
                    chk_case_sensitive = gr.Checkbox(label="Case Sensitive", value=False)
                
                with gr.Row():
                    btn_replace_current = gr.Button("Replace in Current")
                    btn_replace_all = gr.Button("🔄 Replace in ALL Files", variant="secondary")
                
                replace_result = gr.Markdown("")
        
        with gr.Accordion("📦 Bulk Operations", open=False):
            gr.Markdown("Apply operations to **all** caption files in the folder.")
            
            with gr.Row():
                with gr.Column():
                    prefix_input = gr.Textbox(label="Prefix to Add", placeholder="masterpiece, best quality, ")
                    btn_add_prefix = gr.Button("Add Prefix to All")
                
                with gr.Column():
                    suffix_input = gr.Textbox(label="Suffix to Add", placeholder=", high resolution")
                    btn_add_suffix = gr.Button("Add Suffix to All")
            
            with gr.Row():
                with gr.Column():
                    tag_remove_input = gr.Textbox(label="Tag to Remove", placeholder="watermark")
                    btn_remove_tag = gr.Button("Remove Tag from All")
                
                with gr.Column():
                    gr.Markdown("**Remove Duplicate Tags**\n\nClean up repeated tags in all files.")
                    btn_remove_duplicates = gr.Button("Remove Duplicates from All")
            
            bulk_result = gr.Markdown("")
        
        with gr.Accordion("📊 Analytics", open=False):
            with gr.Row():
                btn_analyze_tags = gr.Button("📊 Tag Frequency Analysis")
                btn_statistics = gr.Button("📈 Caption Statistics")
                btn_quality_check = gr.Button("🔍 Quality Check")
                btn_fix_issues = gr.Button("🔧 Auto-Fix Common Issues", variant="secondary")
            
            analytics_result = gr.Markdown("")
        
        # Edit Captions Event Bindings
        btn_edit_load.click(
            load_edit_folder, 
            inputs=[edit_folder_input], 
            outputs=[edit_image_display, edit_caption_box, edit_status, edit_slider]
        )
        
        btn_edit_prev.click(
            edit_prev_image, 
            outputs=[edit_image_display, edit_caption_box, edit_status, edit_slider]
        )
        
        btn_edit_next.click(
            edit_next_image, 
            outputs=[edit_image_display, edit_caption_box, edit_status, edit_slider]
        )
        
        edit_slider.release(
            load_edit_image, 
            inputs=[edit_slider], 
            outputs=[edit_image_display, edit_caption_box, edit_status, edit_slider]
        )
        
        btn_save_caption.click(
            save_edit_caption, 
            inputs=[edit_caption_box], 
            outputs=[save_result]
        )
        
        btn_replace_current.click(
            search_replace_current, 
            inputs=[edit_caption_box, find_text, replace_text, chk_case_sensitive], 
            outputs=[edit_caption_box, replace_result]
        )
        
        btn_replace_all.click(
            search_replace_all, 
            inputs=[edit_folder_input, find_text, replace_text, chk_case_sensitive], 
            outputs=[replace_result]
        )
        
        # Bulk Operations
        btn_add_prefix.click(
            add_prefix_all, 
            inputs=[edit_folder_input, prefix_input], 
            outputs=[bulk_result]
        )
        
        btn_add_suffix.click(
            add_suffix_all, 
            inputs=[edit_folder_input, suffix_input], 
            outputs=[bulk_result]
        )
        
        btn_remove_tag.click(
            remove_tag_all, 
            inputs=[edit_folder_input, tag_remove_input], 
            outputs=[bulk_result]
        )
        
        btn_remove_duplicates.click(
            remove_duplicates_all, 
            inputs=[edit_folder_input], 
            outputs=[bulk_result]
        )
        
        # Analytics
        btn_analyze_tags.click(
            analyze_tag_frequency, 
            inputs=[edit_folder_input], 
            outputs=[analytics_result]
        )
        
        btn_statistics.click(
            get_caption_statistics, 
            inputs=[edit_folder_input], 
            outputs=[analytics_result]
        )
        
        btn_quality_check.click(
            quality_check, 
            inputs=[edit_folder_input], 
            outputs=[analytics_result]
        )
        
        btn_fix_issues.click(
            fix_common_issues, 
            inputs=[edit_folder_input], 
            outputs=[analytics_result]
        )

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
