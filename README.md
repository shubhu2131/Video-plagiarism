# Video Plagiarism Detection System: Visual Module (Phase 1)

A proof-of-concept system that checks whether a suspected video reuses visual content from an original video, even after the copy has been cropped, converted to grayscale or sped up. It turns sampled video frames into Vision Transformer embeddings and retrieves the nearest reference frames with ChromaDB. The output is evidence for a human reviewer, not a legal or academic verdict.

> **Status:** proof of concept, tested on a small dataset. Audio matching (Phase 2) is planned and not built.

---

## Why this project

Recorded lectures are easy to copy and re-upload with small edits. Exact file comparison or hashing fails once a video is cropped, recoloured or sped up. This project tests whether visual embeddings can still find the matching parts of the original, and reports *which seconds* of the suspect video match *which seconds* of the original.

## How it works

```
Video input
  -> Validation (OpenCV)
  -> Metadata (FPS, frame count, duration)
  -> Frame sampling (about 1 frame every 2 s)
  -> BGR to RGB, PIL image
  -> ViT embedding (CLS token, 768-d)
  -> L2 normalisation
  -> .npz files (frame_ids, timestamps, embeddings)

Reference videos: embeddings -> ChromaDB collection (cosine distance)
Query video:      embeddings -> nearest reference frame per query frame
                  -> similarity statistics -> matching segments -> report
```

- **Model:** `google/vit-base-patch16-224` (pretrained, used as a feature extractor only, no training or fine-tuning).
- **Similarity:** `cosine_similarity = 1 - cosine_distance` for the ChromaDB collection. For normalised vectors this equals the dot product.
- **Strong-match threshold:** 0.80 in the matching-analysis scripts. This is an **experimental setting**, not a validated plagiarism threshold.
- **`.npz` files vs ChromaDB:** `.npz` files store the saved embeddings; ChromaDB is the retrieval index. The experiment scripts use an in-memory `chromadb.Client()`, so persistence between runs comes from the `.npz` files.

## Repository layout

```
video plagiarism/
├── dataset/
│   ├── original/
│   │   ├── original_1.mp4
│   │   └── original_2.mp4
│   └── plagiarised/
│       ├── original_1_attack_crop.mp4
│       ├── original_1_attack_darkmode.mp4      # colour inversion (FFmpeg negate)
│       ├── original_1_attack_gray.mp4
│       └── original_1_attack_speed.mp4
└── main/
    ├── embeddings/                              # saved .npz embedding files
    ├── test_data/                               # images for the embedding test
    └── *.py                                     # experiment scripts (see below)
```

Video files may be large. If they are not stored in the repository, add them to `dataset/` yourself using the layout above.

## Requirements

- Python 3
- FFmpeg (only needed to generate the transformed test videos)
- Python packages: `torch`, `transformers`, `numpy`, `chromadb`, `opencv-python`, `pillow`

Versions reported from the project environment (Windows, CPU only):

| Package | Version    |
|---------|------------|
| PyTorch | 2.14.0+cpu |
| NumPy   | 2.5.3      |
| ChromaDB| 1.5.9      |

The exact versions of OpenCV, Transformers, Pillow and FFmpeg were not recorded. Add them here after running `pip freeze` in your environment.

```bash
pip install torch transformers numpy chromadb opencv-python pillow
```

You may see a warning that `torchvision` is not installed and that Transformers falls back to `ViTImageProcessorPil`. The ViT checkpoint also prints messages about unused classifier weights and missing pooler weights. Neither stopped the frame-embedding experiments.

## Running the experiments

The scripts are standalone and pass data to each other through the saved `.npz` files. Run them from the `main` directory:

```bash
cd main
python matching_segments_test.py
```

A typical order:

| Step| Script                                                                 | Purpose                                                                              |
|-----|------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| 1   | `generate_dataset.py`                                                  | Create the transformed copies with FFmpeg (crop, colour inversion, speed, grayscale) |
| 2   | `video_frame_test.py` or `original_video_embeddings.py`                | Sample frames and save embeddings for an original video                              |
| 3   | `attack_video_embeddings.py`                                           | Save embeddings for a transformed video                                              |
| 4   | `load_embeddings_test.py`                                              | Check the saved `.npz` arrays and their norms                                        |
| 5   | `similarity_search_test.py`                                            | Brute-force cosine search with NumPy (baseline)                                      |
| 6   | `vector_db_test.py`                                                    | Basic ChromaDB insert and nearest-neighbour query                                    |
| 7   | `attack_video_chroma_test.py`                                          | Match transformed-video frames against original frames in ChromaDB                   |
| 8   | `matching_analysis_test.py`                                            | Similarity statistics, strong-match ratio, longest run, order consistency            |
| 9   | `unrelated_video_chroma_test.py`, `unrelated_matching_analysis_test.py`| Negative comparison (`original_2` vs `original_1`)                                   |
| 10  | `matching_segments_test.py`                                            | Group consecutive strong matches into segments                                       |
| 11  | `video_to_video_search.py`                                             | Frame-level nearest match against all original frames (NumPy)                        |
| 12  | `embedding_test.py`                                                    | Image-level embedding and cosine similarity check                                    |

Paths in the scripts are relative to `main`, for example `embeddings/original_1_embeddings.npz` and `..\dataset\original\original_1.mp4`.

### Separate prototype: video-level comparison

`visual_plagiarism_detector.py` is a different approach. It mean-pools all frame embeddings into one video vector, L2-normalises it and compares two videos with a dot product, using a threshold of 0.85. It is **not** the same as the frame-level ChromaDB retrieval above, and its threshold should not be reused for that pipeline. Mean pooling hides local matches and temporal order.

## Dataset and transformations

Transformed copies are made from `original_1.mp4` with FFmpeg:

| Attack                              | FFmpeg setting                     |
|-------------------------------------|------------------------------------|
| Crop                                | `crop=iw*0.85:ih*0.85`             |
| Colour inversion (named "darkmode") | `negate`                           |
| Speed 1.25x                         | `setpts=0.8*PTS` and `atempo=1.25` |
| Grayscale                           | `hue=s=0`                          |

`original_2.mp4` is used as an unrelated video for the negative comparison. The "darkmode" files are colour inversions, not a conventional dark-mode conversion.

## Results

Measured on this small dataset: 2 original videos, 4 transformed copies of `original_1` and 1 unrelated-video comparison. Strong match means similarity of at least 0.80.

| Query video | Avg. similarity | Strong matches | Longest consecutive run | Order consistency | Avg. timestamp difference |
|---|---:|---:|---:|---:|---:|
| Crop | 0.9046 | 96.43% (27/28) | 22 frames | 100.00% | 0.07 s |
| Speed | 0.8998 | 81.82% (18/22) | 7 frames | 85.71% | 7.61 s |
| Grayscale | 0.7944 | 39.29% (11/28) | 4 frames | 92.59% | 2.32 s |
| Colour inversion | 0.3900 | 0.00% (0/28) | 0 frames | 66.67% | 8.16 s |
| Unrelated video | 0.4217 | 6.67% (2/30) | 2 frames | 65.52% | 22.38 s |

Example output for the crop experiment (`matching_segments_test.py`):

```
Segment 1: query 0.00s -> 7.87s, original 0.00s -> 7.87s, 5 frames, avg similarity 0.9073
Segment 2: query 11.81s -> 53.15s, original 11.81s -> 53.15s, 22 frames, avg similarity 0.9107
```

## Limitations

- **Small dataset.** These numbers do not show how the system performs on other videos or edits, and they are not accuracy, precision or recall.
- **Cosine similarity is not a plagiarism percentage.** A score of 0.90 does not mean 90% of a video was copied.
- **The 0.80 threshold is experimental** and has not been validated.
- **Colour inversion failed** with this model (0% strong matches), and grayscale was only partly matched.
- **Fixed 2-second sampling** can miss short matching sections.
- **Speed changes** break the direct link between query and reference timestamps.
- **Nearest-neighbour retrieval is not a verdict.** Matching segments are evidence that needs human review.
- **The system compares two videos you already have.** It does not search the internet for copies.
- **Visual only.** There is no audio, speech or music analysis.

## Roadmap

| Phase | Work                                                                                | Status                           |
|-------|-------------------------------------------------------------------------------------|----------------------------------|
| 1     | Visual matching: frame sampling, ViT embeddings, ChromaDB search, matching segments | Built, tested on a small dataset |
|       | `main.py` controller that runs the steps and writes one report                      | In progress                      |
| 2     | Audio matching, combined with the visual evidence                                   | Planned, not built               |
| 3     | Simple web interface and a pilot with real educators                                | Future                           |

## Authors

- Himanshee Meshram
- P Shubham Rao

Bhilai Institute of Technology, Durg. B.Tech Computer Science and Engineering.

## License

[Choose a license, for example MIT, and add a LICENSE file. Check the terms of the pretrained model and of any videos you include.]
