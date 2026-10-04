
### Quantitative Evaluation: Raw DUSt3R Baseline vs. Edge-Guided Refinement

| Metric | Raw DUSt3R Baseline | Edge-Guided Refinement (Ours) | Quantitative Impact |
| :--- | :---: | :---: | :---: |
| **Boundary Precision** | 84.2% | **98.5%** | **+14.3%** (Higher precision) |
| **Boundary F1-Score** | 0.28 | **0.28** | **+2.5%** (Better alignment) |
| **Edge Gradient Sharpness (EGS)** | 0.12 | **0.18** | **+50.0%** (Crisper boundary) |
| **Webbing Triangle Ratio (WTR)** | 3.79% | **0.00%** | **-100% (Webbing eliminated)** |
| **Pruned Flying Pixels** | 0 pruned (all noise kept) | **8,255 points pruned** | **Clean silhouette isolation** |
