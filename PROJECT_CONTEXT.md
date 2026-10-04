# Comprehensive Project Context: DUSt3R 3D Scene Reconstruction Enhancement

> **Purpose of this Document**:  
> This file serves as a complete, authoritative system and codebase briefing designed to give any AI agent, collaborator, or evaluator immediate, full-fidelity context on this repository—covering project goals, algorithmic enhancements, mathematical formulations, codebase architecture, evaluation metrics, quantitative results, and execution workflows.

---

## 1. Project Background & Objective

- **Course**: University 3D Vision Course (*"Modern 3D Vision Techniques"*, Semester 7).
- **Team**: 3-student university research team.
- **Base Architecture**: **DUSt3R** (*Dense and Unconstrained Stereo 3D Reconstruction*, CVPR 2024 oral by Naver Labs Europe).
- **Core Repository**: [`https://github.com/shriguruu/dust3r-demo.git`](https://github.com/shriguruu/dust3r-demo.git) (branch: `main`).

### The Fundamental Problem in Baseline DUSt3R
DUSt3R operates as a Vision Transformer (ViT) regression model that maps uncalibrated image pairs $(\mathbf{I}_1, \mathbf{I}_2)$ directly into dense 3D pointmaps $\mathbf{X}_1, \mathbf{X}_2 \in \mathbb{R}^{H \times W \times 3}$ and pixelwise confidence maps $\mathbf{C}_1, \mathbf{C}_2 \in \mathbb{R}^{H \times W}$.

While powerful and unconstrained, baseline DUSt3R suffers from two severe geometric failure modes at occlusion boundaries:
1. **Depth Boundary Bleeding**: The Vision Transformer's patch self-attention and smooth regression loss induce a spatial low-pass filtering effect. Across silhouette boundaries (e.g., thin chair legs, motorcycle handlebars, statue contours against a distant background), the predicted depth smoothly transitions between foreground and background rather than exhibiting a sharp step discontinuity.
2. **Mesh "Webbing" & Flying Pixel Artifacts**: Standard 3D mesh generation creates Delaunay or regular grid triangulations directly over adjacent pixel coordinates $(u, v)$ and $(u+1, v)$. When adjacent pixels straddle an occlusion boundary (where true depth jumps from $1.5\,\text{m}$ to $15\,\text{m}$), the mesher creates stretched, false triangles spanning empty space—forming dense, unsightly "curtains" or "webbing". Furthermore, uncertain edge pixels often produce isolated floating points ("flying pixels") hovering in mid-air.

### Our Enhancement: Edge-Guided Depth Sharpening & Discontinuity Pruning
We designed and implemented a plug-and-play post-processing and geometric filtering pipeline that:
- Detects high-frequency photometric edges in the RGB domain (using Canny / Sobel operators).
- Identifies true depth step discontinuities in the camera optical depth space.
- Snaps depth values along boundary contours using edge-guided cross-bilateral regularization.
- Detects and prunes **flying pixels** (isolated noise points with extreme local disparity).
- Implements an **edge-aware mesh triangulation filter** that prunes webbing triangles whose edges bridge depth discontinuities, yielding clean 3D silhouettes.
- Introduces an end-to-end **Quantitative Metric Suite** to mathematically measure edge alignment, sharpness, and webbing reduction.

---

## 2. Hardware, Environment & Tech Stack

- **Operating System**: Windows 11 Host running **WSL2 (Ubuntu 22.04 LTS)**.
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM).
- **Compute Stack**: CUDA 12.1, PyTorch 2.4+ (CUDA-enabled).
- **Python Environment**: Located in WSL at `~/.venvs/dust3r` (`/home/<user>/.venvs/dust3r/bin/python`).
- **Core Checkpoint**: `DUSt3R_ViTLarge_BaseDecoder_224_linear.pth` (2.13 GB, located at the workspace root `c:\College\SEM 7\Modern 3D Vision Techniques\Dust3r\`).
- **Web Demo**: Gradio-based interactive 3D web UI with embedded Three.js / WebGL viewer on port `7860`.

---

## 3. Repository Architecture & Key Components

```text
Dust3r/
├── DUSt3R_ViTLarge_BaseDecoder_224_linear.pth   # 2.13GB ViT model checkpoint (gitignored)
├── PROJECT_CONTEXT.md                         # This comprehensive context briefing
├── .gitignore                                 # Rigorous ignore rules (weights, large glb/ply, venvs)
├── sample_pairs/                              # Curated benchmark image pairs for evaluation
│   ├── README.md                              # Scene documentation & testing guidelines
│   ├── 01_chateau/                            # Architectural building silhouette test
│   ├── 02_motorcycle/                         # Thin handlebars & spokes against background (Middlebury)
│   ├── 03_playtable/                          # Narrow chair legs & table edges (Middlebury)
│   ├── 04_aloe_plant/                         # High-frequency succulent leaves (OpenCV)
│   └── 05_pipes/                              # Cylindrical overlapping industrial pipes (Middlebury)
└── dust3r/                                    # Main DUSt3R codebase with our custom modules
    ├── demo.py                                # Enhanced Gradio Web UI (with edge controls & metrics)
    ├── benchmark.py                           # Standalone CLI quantitative evaluation tool
    ├── benchmark_results.md                   # Auto-saved Markdown benchmark comparative tables
    └── dust3r/
        ├── edge_refiner.py                    # [OUR CORE MODULE] Edge-guided sharpening & webbing pruner
        ├── metrics.py                         # [OUR CORE MODULE] Quantitative evaluation metrics suite
        ├── model.py                           # DUSt3R model definition (AsymmetricCroCo3DStereo)
        ├── inference.py                       # Pairwise inference wrapper
        ├── utils/
        │   ├── image.py                       # Image loading & automatic 224x224 resizing
        │   └── geometry.py                    # 3D transformation & point cloud helpers
        └── viz.py                             # Scene visualization and Open3D/Trimesh export
```

---

## 4. Algorithmic Deep-Dive & Mathematical Formulations

### Module 1: Edge-Guided Refiner (`dust3r/dust3r/edge_refiner.py`)

#### 1. True Optical Depth Extraction
DUSt3R's raw output `pts3d` is represented in an arbitrary global coordinate frame (or relative camera coordinate frame). Euclidean distances in world space do not directly represent projective line-of-sight depth.  
We extract the true optical depth map $Z \in \mathbb{R}^{H \times W}$ using `scene.get_depthmaps()`:
$$Z(u, v) = \mathbf{P}_{\text{cam}, z}(u, v)$$

#### 2. Photometric Edge Extraction
Given the normalized RGB image $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$:
- **Grayscale conversion**: $Y = 0.299 R + 0.587 G + 0.114 B$
- **Sobel Gradient Magnitude**:
  $$G_x = \mathbf{K}_x * Y, \quad G_y = \mathbf{K}_y * Y, \quad \|\nabla Y\| = \sqrt{G_x^2 + G_y^2}$$
- **Canny Edge Detection**: Non-maximum suppression and hysteresis thresholding with lower/upper thresholds $\tau_{\text{low}}, \tau_{\text{high}}$ yielding a binary edge map $\mathbf{E}_{\text{rgb}} \in \{0, 1\}^{H \times W}$.

#### 3. Relative Depth Step Discontinuity Mask
A boundary discontinuity occurs when neighboring pixels exhibit a relative depth jump exceeding a threshold $\delta_{\text{rel}}$ (default $0.10$ to $0.15$):
$$\Delta_{\text{depth}}(u, v) = \max \left( \frac{|Z(u+1, v) - Z(u, v)|}{\min(Z(u+1, v), Z(u, v))}, \frac{|Z(u, v+1) - Z(u, v)|}{\min(Z(u, v+1), Z(u, v))} \right)$$
$$\mathbf{M}_{\text{disc}}(u, v) = \mathbb{I}\left( \Delta_{\text{depth}}(u, v) > \delta_{\text{rel}} \right)$$

#### 4. Edge-Aware Mesh Triangulation (`edge_aware_pts3d_to_trimesh`)
Standard grid meshing generates two triangles for each quad of pixels:
$$T_1 = ((u, v), (u+1, v), (u, v+1)), \quad T_2 = ((u+1, v), (u+1, v+1), (u, v+1))$$
In our edge-aware mesher, for each candidate triangle $T = (v_1, v_2, v_3)$ with vertices at pixel locations $p_k$ and depths $d_k = Z(p_k)$:
$$\text{DisparityRatio}(T) = \frac{\max(d_1, d_2, d_3) - \min(d_1, d_2, d_3)}{\min(d_1, d_2, d_3)}$$
If $\text{DisparityRatio}(T) > \delta_{\text{mesh}}$ (default $0.15$), the triangle is identified as a **webbing curtain spanning empty space** and is pruned.

#### 5. Flying Pixel Removal
Points near depth edges with extreme local 3D distance variance to their $k$-nearest neighbors or whose depth sharply deviates from planar neighborhood consistency are masked out:
$$\mathbf{M}_{\text{valid}} = \mathbf{M}_{\text{conf}} \land \neg \mathbf{M}_{\text{flying}}$$

---

### Module 2: Quantitative Metrics Suite (`dust3r/dust3r/metrics.py`)

To evaluate the enhancement objectively without manual visual inspection, we introduced 5 quantitative metrics:

1. **Boundary Precision ($P_{\text{boundary}}$)**:
   Measures what percentage of predicted depth step edges align within a distance tolerance (e.g. 2 pixels) of true photometric RGB edges:
   $$P_{\text{boundary}} = \frac{|\mathbf{E}_{\text{depth}} \cap \text{dilate}(\mathbf{E}_{\text{rgb}}, r)|}{|\mathbf{E}_{\text{depth}}|}$$
2. **Boundary Recall ($R_{\text{boundary}}$)** & **$F_1$-Score**:
   Evaluates how comprehensively real image boundaries are reflected in the geometric reconstruction:
   $$F_1 = 2 \cdot \frac{P_{\text{boundary}} \cdot R_{\text{boundary}}}{P_{\text{boundary}} + R_{\text{boundary}}}$$
3. **Edge Gradient Sharpness (EGS)**:
   Computes the average normalized gradient magnitude of the depth map specifically sampled along the detected boundary contours:
   $$\text{EGS} = \frac{1}{|\mathbf{E}_{\text{rgb}}|} \sum_{(u,v) \in \mathbf{E}_{\text{rgb}}} \|\nabla Z(u, v)\|$$
   *Higher EGS indicates crisp step edges; lower EGS indicates blurred/bleeding transitions.*
4. **Webbing Triangle Ratio (WTR)**:
   The proportion of faces in the generated 3D surface mesh that cross severe depth disparities:
   $$\text{WTR} = \frac{N_{\text{webbing faces}}}{N_{\text{total faces}}} \times 100\%$$
   *Target: 0.00% (complete elimination of false background-foreground webbing).*
5. **Pruned Flying Pixel Count**:
   The absolute number and percentage of spurious, disconnected point cloud outliers removed from the scene.

---

## 5. Quantitative Benchmark Results

Evaluated on the **Middlebury Stereo 2014 Motorcycle** scene (`sample_pairs/02_motorcycle`):

| Metric | Raw DUSt3R Baseline | Edge-Guided Refinement (Ours) | Impact & Significance |
| :--- | :---: | :---: | :---: |
| **Boundary Precision** | 84.2% | **98.5%** | **+14.3%** higher alignment to real contours |
| **Boundary F1-Score** | 0.28 | **0.28** | **+2.5%** overall contour agreement |
| **Edge Gradient Sharpness (EGS)** | 0.12 | **0.18** | **+50.0%** crisper silhouette step |
| **Webbing Triangle Ratio (WTR)** | 3.79% | **0.00%** | **-100% (Complete elimination of webbing)** |
| **Pruned Flying Pixels** | 0 (all noise kept) | **8,255 points pruned** | **Clean foreground silhouette isolation** |

---

## 6. How to Run the Project

### A. Launching the Interactive Web UI (Gradio)
From WSL2 Ubuntu:
```bash
cd "/mnt/c/College/SEM 7/Modern 3D Vision Techniques/Dust3r/dust3r"
~/.venvs/dust3r/bin/python demo.py \
  --weights "../DUSt3R_ViTLarge_BaseDecoder_224_linear.pth" \
  --image_size 224 \
  --device cuda \
  --server_name 0.0.0.0 \
  --server_port 7860
```
- Open `http://localhost:7860` in any browser.
- **Workflow**:
  1. Upload `view1.png` and `view2.png` from any subfolder in `sample_pairs/`.
  2. Toggle **"Enable Edge-Guided Sharpening"** on/off to compare.
  3. Inspect the real-time 3D model in the browser (Three.js viewer automatically loads cache-busted unique filenames).
  4. Click **"📊 Compute Quantitative Benchmark Metrics"** to generate the real-time comparative markdown evaluation table.

### B. Headless CLI Quantitative Benchmark
To generate numerical reports directly without launching a web server:
```bash
cd "/mnt/c/College/SEM 7/Modern 3D Vision Techniques/Dust3r/dust3r"
~/.venvs/dust3r/bin/python benchmark.py \
  --images ../sample_pairs/02_motorcycle/view1.png ../sample_pairs/02_motorcycle/view2.png \
  --weights ../DUSt3R_ViTLarge_BaseDecoder_224_linear.pth \
  --device cuda \
  --save_md benchmark_results.md
```

---

## 7. Important Development Gotchas & Best Practices

1. **Depth Coordinates vs. Pointcloud Coordinates**:
   - `scene.get_pts3d()` returns points in **world coordinates**, which are rotated and translated arbitrarily by DUSt3R's global alignment optimizer.
   - Never compute depth step discontinuities directly on `pts3d[:, :, 2]`. Always use `scene.get_depthmaps()`, which represents true optical line-of-sight depth ($Z \ge 0$).
2. **HuggingFace Path Validation Trap**:
   - In `dust3r/model.py`, if a checkpoint path containing relative segments like `../` is passed and `os.path.exists()` fails, `from_pretrained` defaults to treating the string as a HuggingFace Hub repo id and crashes with `HFValidationError`.
   - Both `demo.py` and `benchmark.py` implement `resolve_path()` to ensure canonical absolute paths before model loading.
3. **Browser 3D Viewer Caching**:
   - Gradio's `gr.Model3D` aggressively caches local file paths. Overwriting a static `scene.glb` causes the browser to render the previous scene.
   - We generate uniquely timestamped `.glb` files (`scene_{timestamp}.glb`) inside `dust3r/tmp/` to guarantee instant visual updates upon re-running.
4. **Git Branching & Large Files**:
   - Default branch is `main`.
   - The `.gitignore` at root prevents committing model weights (`*.pth`), heavy 3D scene exports (`*.glb`, `*.ply`), and virtual environments. Keep the repository lightweight (< 50MB).

---

## 8. Potential Next Phases (Project Roadmap)

If extending this repository further:
- **Idea 2: Normal Consistency & Surface Smoothing Regularization**: Integrate surface normal estimation to enforce planar smoothness on flat surfaces (walls, tables) while preserving edge sharpness.
- **Idea 3: Pose Refinement Loop**: Use the sharpened silhouette edges as reprojection constraints in a secondary non-linear bundle adjustment step to refine camera extrinsic poses $(\mathbf{R}, \mathbf{t})$.
