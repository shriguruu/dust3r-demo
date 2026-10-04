
### Quantitative Evaluation: Raw DUSt3R Baseline vs. Edge-Guided Refinement

| Metric | Raw DUSt3R Baseline | Edge-Guided Refinement (Ours) | Quantitative Impact |
| :--- | :---: | :---: | :---: |
| **Boundary Precision** | 66.6% | **91.6%** | **+25.0%** (Higher precision) |
| **Boundary F1-Score** | 0.17 | **0.17** | **+4.8%** (Better alignment) |
| **Edge Gradient Sharpness (EGS)** | 0.12 | **0.18** | **+50.0%** (Crisper boundary) |
| **Webbing Triangle Ratio (WTR)** | 2.51% | **0.00%** | **-100% (Webbing eliminated)** |
| **Pruned Flying Pixels** | 0 pruned (all noise kept) | **6,264 points pruned** | **Clean silhouette isolation** |
