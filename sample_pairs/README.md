# Curated Benchmark Image Pairs for DUSt3R Enhancement Showcase

This directory contains curated image pairs selected from standard benchmark datasets (Middlebury Stereo 2014, OpenCV, and CROCO) to showcase our **Edge-Guided Depth Sharpening and Flying Pixel / Webbing Pruning** pipeline compared to baseline DUSt3R.

---

## 📁 Directory Structure & Scene Descriptions

| Folder | Source Dataset | Key Geometric Features | Why It Highlights Our Enhancement |
| :--- | :--- | :--- | :--- |
| **`01_chateau/`** | CROCO / DUSt3R Asset | Outdoor architecture, spires, sharp rooflines, stone edges against sky | Baseline DUSt3R produces soft, rounded building silhouettes. Our edge-guided filter snaps depth boundaries to the architectural silhouette. |
| **`02_motorcycle/`** | Middlebury Stereo 2014 | Thin handlebars, wheel spokes, rearview mirrors, cables against distant workshop background | **Classic failure mode for baseline unconstrained depth models:** Baseline DUSt3R bleeds depth across thin foreground objects into the background, creating dense "curtains" of webbing triangles and flying floating points. Our discontinuity filter prunes connecting webbing triangles cleanly. |
| **`03_playtable/`** | Middlebury Stereo 2014 | Narrow chair legs, table edges, blocks, sharp multi-plane occlusions | Chair legs and table corners showcase sharp step discontinuities (from foreground legs to distant floor). Webbing Triangle Ratio (WTR) drops significantly. |
| **`04_aloe_plant/`** | OpenCV Stereo Benchmark | Pointed, thin succulent leaves overlapping at diverse depth layers | Evaluates boundary precision ($F_1$) on high-frequency organic contours. Prevents adjacent leaf depths from merging into an amorphous blob. |
| **`05_pipes/`** | Middlebury Stereo 2014 | Overlapping cylindrical industrial pipes, metallic edges, sharp shadows | Cylindrical boundary silhouettes are sharpened; flying points in depth voids between parallel pipes are pruned. |

---

## 🚀 How to Run and Showcase

### 1. In the Interactive Gradio Web Demo
1. Ensure the web server is running (`http://localhost:7860`).
2. In the Gradio UI under **"Input Images"**, upload `view1` and `view2` from any folder above (e.g., `sample_pairs/02_motorcycle/view1.png` and `view2.png`).
3. Set **Confidence Threshold** to `1.0` or `1.5`.
4. First run with **"Enable Edge-Guided Sharpening"** unchecked.
5. In the 3D viewer, inspect thin edges (e.g. motorcycle handlebars, chair legs) — observe the "webbing" mesh connecting foreground and background.
6. Now check **"Enable Edge-Guided Sharpening"** (with default parameters: `Sobel + Canny`, Discontinuity Threshold `0.10 - 0.15`).
7. Re-run and click **"📊 Compute Quantitative Benchmark Metrics"**:
   - Notice the **pruned flying pixels**.
   - Notice the reduction in **Webbing Triangle Ratio (WTR)**.
   - Observe the increase in **Edge Gradient Sharpness (EGS)**.

### 2. Via CLI Quantitative Benchmark
You can run automated numerical evaluations directly from WSL:

```bash
# Evaluate Motorcycle scene
cd dust3r
~/.venvs/dust3r/bin/python benchmark.py \
  --images ../sample_pairs/02_motorcycle/view1.png ../sample_pairs/02_motorcycle/view2.png \
  --weights ../DUSt3R_ViTLarge_BaseDecoder_224_linear.pth \
  --device cuda

# Evaluate Playtable scene
~/.venvs/dust3r/bin/python benchmark.py \
  --images ../sample_pairs/03_playtable/view1.png ../sample_pairs/03_playtable/view2.png \
  --weights ../DUSt3R_ViTLarge_BaseDecoder_224_linear.pth \
  --device cuda
```
