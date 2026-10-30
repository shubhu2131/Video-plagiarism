import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDINGS_FILE = "C:/video plagiarism/main/video_embeddings.pt"
MODEL_SAVE_PATH = "C:/video plagiarism/main/metric_projection_head.pth"

# Set random seeds for reproducible training
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------
# 1. NEURAL PROJECTION HEAD
# ---------------------------------------------------------------------
class MetricProjectionHead(nn.Module):
    """
    Projects raw 768-d ViT embeddings into a 256-d metric space.
    L2-normalization on output ensures vectors lie on a unit hypersphere
    where Euclidean distance directly mirrors Cosine Similarity.
    """
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
# 2. TRIPLET DATASET LOADER
# ---------------------------------------------------------------------
class TripletVideoDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anc, pos, neg = self.triplets[idx]
        return (
            torch.tensor(anc, dtype=torch.float32),
            torch.tensor(pos, dtype=torch.float32),
            torch.tensor(neg, dtype=torch.float32)
        )

def build_triplets(embeddings_dict):
    """
    Builds (Anchor, Positive, Negative) sets:
      - Anchor (A): Original video embedding.
      - Positive (P): Any attack variation of that exact original.
      - Negative (N): A completely different original video embedding.
    """
    originals = {k: v for k, v in embeddings_dict.items() if v.get("type") == "original"}
    plagiarised = {k: v for k, v in embeddings_dict.items() if v.get("type") == "plagiarised"}

    orig_keys = list(originals.keys())
    if len(orig_keys) < 2:
        raise ValueError(
            f"Insufficient original videos found ({len(orig_keys)}). "
            "You need at least 2 distinct original videos to form negative pairs."
        )

    triplets = []
    for plag_key, plag_meta in plagiarised.items():
        base_name = plag_meta["base"]
        if base_name not in originals:
            continue

        anchor_vec = originals[base_name]["vector"]
        positive_vec = plag_meta["vector"]

        # Negative must come from a completely different original video
        possible_negatives = [k for k in orig_keys if k != base_name]
        for neg_key in possible_negatives:
            negative_vec = originals[neg_key]["vector"]
            triplets.append((anchor_vec, positive_vec, negative_vec))

    random.shuffle(triplets)
    return triplets

# ---------------------------------------------------------------------
# 3. EVALUATION FUNCTION
# ---------------------------------------------------------------------
def evaluate_distances(model, dataloader):
    """Computes mean positive distance D(A, P) and negative distance D(A, N)."""
    model.eval()
    pos_dists = []
    neg_dists = []

    with torch.no_grad():
        for anc, pos, neg in dataloader:
            anc, pos, neg = anc.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            emb_a = model(anc)
            emb_p = model(pos)
            emb_n = model(neg)

            # Euclidean distance on normalized vectors
            d_ap = torch.norm(emb_a - emb_p, p=2, dim=1)
            d_an = torch.norm(emb_a - emb_n, p=2, dim=1)

            pos_dists.extend(d_ap.cpu().tolist())
            neg_dists.extend(d_an.cpu().tolist())

    return np.mean(pos_dists), np.mean(neg_dists)

# ---------------------------------------------------------------------
# 4. MAIN TRAINING LOOP
# ---------------------------------------------------------------------
def train():
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(
            f"Cannot find '{EMBEDDINGS_FILE}'. Please run extract_embeddings.py first!"
        )

    print(f"[INFO] Loading cached embeddings from: {EMBEDDINGS_FILE}")
    embeddings = torch.load(EMBEDDINGS_FILE, weights_only=False)

    triplets = build_triplets(embeddings)
    total_triplets = len(triplets)
    print(f"[INFO] Formed {total_triplets} unique (A, P, N) training triplets.")

    # Determine batch size dynamically
    batch_size = 8 if total_triplets >= 16 else 4
    
    # drop_last=True prevents single-sample batches from crashing BatchNorm1d
    dataloader = DataLoader(
        TripletVideoDataset(triplets),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True if total_triplets > batch_size else False
    )

    model = MetricProjectionHead().to(DEVICE)

    # Margin alpha = 0.4 ensures a solid separation gap between matches and negatives
    criterion = nn.TripletMarginLoss(margin=0.4, p=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Measure initial distances prior to training
    init_ap, init_an = evaluate_distances(model, dataloader)
    print(f"[INITIAL BASELINE] D(Anchor, Positive): {init_ap:.4f} | D(Anchor, Negative): {init_an:.4f}")

    print("\n" + "=" * 62)
    print("      TRAINING METRIC PROJECTION HEAD (TRIPLET LOSS)      ")
    print("=" * 62)

    epochs = 25
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for anc, pos, neg in dataloader:
            anc, pos, neg = anc.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)

            optimizer.zero_grad()
            emb_a = model(anc)
            emb_p = model(pos)
            emb_n = model(neg)

            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if epoch % 5 == 0 or epoch == 1:
            curr_ap, curr_an = evaluate_distances(model, dataloader)
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"| Loss: {avg_loss:.6f} "
                f"| D(A, P): {curr_ap:.4f} "
                f"| D(A, N): {curr_an:.4f}"
            )

    # Save model weights
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("=" * 62)
    print(f"[SUCCESS] Projection head weights saved to:\n  --> {MODEL_SAVE_PATH}")
    print("=" * 62)

if __name__ == "__main__":
    train()