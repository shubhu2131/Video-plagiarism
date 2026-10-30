import os
import glob
import torch
import numpy as np
from visual_plagiarism_detector import extract_keyframes, VisualEncoder

ORIGINAL_DIR = "C:/video plagiarism/dataset/original"
PLAGIARISED_DIR = "C:/video plagiarism/dataset/plagiarised"
OUTPUT_FILE = "C:/video plagiarism/main/video_embeddings.pt"

def extract_and_cache():
    encoder = VisualEncoder()
    embeddings = {}

    orig_files = glob.glob(os.path.join(ORIGINAL_DIR, "*.mp4"))
    plag_files = glob.glob(os.path.join(PLAGIARISED_DIR, "*.mp4"))

    print(f"[INFO] Found {len(orig_files)} original videos and {len(plag_files)} attack videos.")

    # 1. Process Originals
    print("\n--- Extracting Original Embeddings ---")
    for file_path in orig_files:
        name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Processing: {name}.mp4")
        frames = extract_keyframes(file_path, sample_interval_sec=4.0)
        vec = encoder.encode_video(frames)
        embeddings[name] = {"vector": vec, "type": "original", "base": name}

    # 2. Process Plagiarised Variations
    print("\n--- Extracting Plagiarised Attack Embeddings ---")
    for file_path in plag_files:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        # Identify the original base video name (before _attack_)
        base_name = filename.split("_attack_")[0]
        print(f"Processing: {filename}.mp4 (Base: {base_name})")
        frames = extract_keyframes(file_path, sample_interval_sec=4.0)
        vec = encoder.encode_video(frames)
        embeddings[filename] = {"vector": vec, "type": "plagiarised", "base": base_name}

    torch.save(embeddings, OUTPUT_FILE)
    print(f"\n[SUCCESS] Extracted vectors saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_and_cache()