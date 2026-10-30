import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
from transformers import ViTImageProcessor, ViTModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECTION_WEIGHTS = "C:/video plagiarism/main/metric_projection_head.pth"

# ---------------------------------------------------------------------
# 1. KEYFRAME EXTRACTION FUNCTION
# ---------------------------------------------------------------------
def extract_keyframes(video_path, sample_interval_sec=4.0):
    """
    Extracts keyframes at fixed time intervals (default: 1 frame every 4 seconds).
    Returns a list of BGR frame arrays.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    frame_interval = int(round(fps * sample_interval_sec))
    frame_interval = max(1, frame_interval)

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()

    # Fallback to the first frame if the video is extremely short
    if not frames:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        cap.release()

    return frames

# ---------------------------------------------------------------------
# 2. METRIC PROJECTION HEAD DEFINITION
# ---------------------------------------------------------------------
class MetricProjectionHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        features = self.network(x)
        return nn.functional.normalize(features, p=2, dim=1)

# ---------------------------------------------------------------------
# 3. ENCODER (ViT + PROJECTION HEAD)
# ---------------------------------------------------------------------
class VisualEncoder:
    def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
        print(f"[INFO] Initializing Vision Transformer on device: {DEVICE}")
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

        self.projection_head = None
        if os.path.exists(PROJECTION_WEIGHTS):
            self.projection_head = MetricProjectionHead().to(DEVICE)
            self.projection_head.load_state_dict(
                torch.load(PROJECTION_WEIGHTS, map_location=DEVICE, weights_only=True)
            )
            self.projection_head.eval()
            print("[INFO] Loaded trained MetricProjectionHead weights successfully.")
        else:
            print("[WARN] No projection head found. Using unprojected ViT features.")

    def encode_video(self, frames):
        if not frames:
            dim = 256 if self.projection_head else 768
            return torch.zeros((1, dim)).numpy()

        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        inputs = self.processor(images=pil_images, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Pool [CLS] token across keyframes
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            video_vector = torch.mean(cls_embeddings, dim=0, keepdim=True)

            if self.projection_head is not None:
                video_vector = self.projection_head(video_vector)
            else:
                video_vector = nn.functional.normalize(video_vector, p=2, dim=1)

        return video_vector.cpu().numpy()

# ---------------------------------------------------------------------
# 4. COMPARISON PIPELINE
# ---------------------------------------------------------------------
def compare_videos(video_1_path, video_2_path, threshold=0.85):
    encoder = VisualEncoder()

    print(f"\n[INFO] Extracting keyframes: {os.path.basename(video_1_path)}")
    frames_1 = extract_keyframes(video_1_path, sample_interval_sec=4.0)
    print(f"[INFO] Extracted {len(frames_1)} keyframes. Encoding vector...")
    vec_1 = encoder.encode_video(frames_1)

    print(f"\n[INFO] Extracting keyframes: {os.path.basename(video_2_path)}")
    frames_2 = extract_keyframes(video_2_path, sample_interval_sec=4.0)
    print(f"[INFO] Extracted {len(frames_2)} keyframes. Encoding vector...")
    vec_2 = encoder.encode_video(frames_2)

    # Dot product of normalized vectors equals Cosine Similarity
    sim = float((vec_1 @ vec_2.T)[0][0])
    sim_clamped = max(0.0, min(1.0, sim))

    print("\n" + "=" * 55)
    print("                ANALYSIS RESULTS")
    print("=" * 55)
    print(f"Visual Cosine Similarity Score : {sim_clamped:.4f}")
    print(f"Visual Match Percentage        : {sim_clamped * 100:.2f}%")
    print(f"Plagiarism Threshold           : {threshold * 100:.2f}%")
    print("-" * 55)

    if sim_clamped >= threshold:
        print("VERDICT: [ALERT] High visual similarity detected. Content is plagiarised.")
    else:
        print("VERDICT: [CLEARED] Visual similarity is below threshold. Content appears original.")
    print("=" * 55 + "\n")

# ---------------------------------------------------------------------
# 5. EXECUTION ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Normalized forward slashes eliminate escape sequence bugs
    VIDEO_FILE_1 = "C:/video plagiarism/dataset/original/maths.mp4"
    VIDEO_FILE_2 = "C:/video plagiarism/dataset/plagiarised/maths_V1.mp4"
    compare_videos(VIDEO_FILE_1, VIDEO_FILE_2, threshold=0.85)
    