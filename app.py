import streamlit as st
import os
import torch
import torch.nn.functional as F
import tempfile

# Import your core project functions and classes
from visual_plagiarism_detector import extract_keyframes, VisualEncoder
from train_triplet import MetricProjectionHead  # Ensure this matches your class file

# Streamlit Page Configuration
st.set_page_config(
    page_title="Video Plagiarism Detector",
    page_icon="🛡️",
    layout="centered"
)

# Cache model loading so it loads only once into memory
@st.cache_resource
def load_detection_models():
    print("[INFO] Loading ViT Encoder & Trained Metric Projection Head...")
    encoder = VisualEncoder()
    
    projection_head = MetricProjectionHead()
    weights_path = "C:/video plagiarism/main/metric_projection_head.pth"
    
    if os.path.exists(weights_path):
        try:
            projection_head.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            print("[INFO] Successfully loaded model weights.")
        except Exception as e:
            print(f"[WARNING] Could not load state dict directly: {e}")
    else:
        st.warning(f"Weights file not found at {weights_path}. Running with uninitialized weights!")
        
    projection_head.eval()
    return encoder, projection_head

# Load models on startup
encoder, projection_head = load_detection_models()

# UI Header
st.title("🛡️ Video Plagiarism & Tampering Detection System")
st.markdown("### **Project Kavach (प्रतिकृति AI)**")
st.write("Upload an **Original Lecture Video** and an **Edited/Tampered Video** below to analyze their structural and visual similarity using Vision Transformers and Triplet-Trained Metric Projection.")

# File Uploaders
st.subheader("1. Upload Video Files for Analysis")
col1, col2 = st.columns(2)

with col1:
    original_file = st.file_uploader("Original Video (.mp4)", type=["mp4", "avi", "mov"])

with col2:
    edited_file = st.file_uploader("Test / Edited Video (.mp4)", type=["mp4", "avi", "mov"])

# Helper function to write uploaded files to temporary storage for OpenCV processing
def save_uploaded_file(uploaded_file):
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        return tfile.name
    except Exception as e:
        st.error(f"Error handling file upload: {e}")
        return None

# Analysis Execution Button
if st.button("Run Plagiarism Analysis", type="primary"):
    if original_file is not None and edited_file is not None:
        with st.spinner("Extracting frames, passing through Vision Transformer, and computing similarity..."):
            
            # Save uploaded files temporarily
            orig_path = save_uploaded_file(original_file)
            edited_path = save_uploaded_file(edited_file)

            try:
                # 1. Extract Keyframes and Encode via ViT
                orig_frames = extract_keyframes(orig_path, sample_interval_sec=2.0)
                orig_vec = encoder.encode_video(orig_frames)

                edited_frames = extract_keyframes(edited_path, sample_interval_sec=2.0)
                edited_vec = encoder.encode_video(edited_frames)

                # Convert to PyTorch Tensors
                if not isinstance(orig_vec, torch.Tensor):
                    orig_vec = torch.tensor(orig_vec, dtype=torch.float32)
                if not isinstance(edited_vec, torch.Tensor):
                    edited_vec = torch.tensor(edited_vec, dtype=torch.float32)

                # Ensure tensors are 2D shape [num_frames, feature_dim]
                if orig_vec.dim() == 1:
                    orig_vec = orig_vec.unsqueeze(0)
                if edited_vec.dim() == 1:
                    edited_vec = edited_vec.unsqueeze(0)

                # 2. Safe Projection / Embedding Handling
                with torch.no_grad():
                    try:
                        # Attempt passing through projection head
                        orig_proj = projection_head(orig_vec)
                        edited_proj = projection_head(edited_vec)
                        
                        orig_embedding = F.normalize(orig_proj.mean(dim=0), p=2, dim=0)
                        edited_embedding = F.normalize(edited_proj.mean(dim=0), p=2, dim=0)
                    except Exception as proj_err:
                        # Fallback: if dimension mismatch occurs, use raw encoded vectors directly
                        st.info(f"Notice: Bypassing projection head due to shape adaptation ({proj_err}). Using raw feature embeddings.")
                        orig_embedding = F.normalize(orig_vec.mean(dim=0), p=2, dim=0)
                        edited_embedding = F.normalize(edited_vec.mean(dim=0), p=2, dim=0)

                # 3. Calculate Cosine Similarity
                similarity = torch.dot(orig_embedding, edited_embedding).item()
                similarity_percentage = similarity * 100

                # Clean up temporary files from disk
                if orig_path and os.path.exists(orig_path):
                    os.unlink(orig_path)
                if edited_path and os.path.exists(edited_path):
                    os.unlink(edited_path)

                # 4. Display Results on UI
                st.success("Analysis Successfully Completed!")
                st.metric(label="Calculated Cosine Similarity Score", value=f"{similarity_percentage:.2f}%")

                # Threshold verdict logic (85% boundary decision)
                if similarity_percentage >= 85.0:
                    st.error(f"**VERDICT: 🚨 PLAGIARIZED / TAMPERED COPY DETECTED!** (Exceeds the 85.0% threshold)")
                else:
                    st.success(f"**VERDICT: ✅ ORIGINAL / UNRELATED CONTENT** (Below the 85.0% threshold)")

            except Exception as e:
                st.error(f"An error occurred during video processing: {e}")
    else:
        st.warning("⚠️ Please upload both an **Original Video** and a **Test/Edited Video** before running the analysis.")