#!/usr/bin/env python3
"""
Quantitative Benchmarking Script for DUSt3R Edge Refinement
------------------------------------------------------------
Computes and reports academic metrics comparing Raw DUSt3R Baseline
vs. Edge-Guided Refined Output.
"""

import argparse
import os
import torch
import numpy as np

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
from dust3r.metrics import compute_full_comparison_table


def resolve_path(p, script_dir):
    if os.path.exists(p):
        return os.path.abspath(p)
    alt = os.path.join(script_dir, p)
    if os.path.exists(alt):
        return os.path.abspath(alt)
    alt_parent = os.path.join(script_dir, '..', os.path.basename(p))
    if os.path.exists(alt_parent):
        return os.path.abspath(alt_parent)
    return p


def benchmark_pair(weights_path, img_paths, image_size=224, device='cuda', flying_thresh=0.08, max_depth_ratio=0.08):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = resolve_path(weights_path, script_dir)
    img_paths = [resolve_path(p, script_dir) for p in img_paths]

    print("=" * 70)
    print("  DUSt3R QUANTITATIVE BENCHMARK: RAW BASELINE vs. EDGE-GUIDED REFINEMENT")
    print("=" * 70)
    print(f"Loading model from: {weights_path}")
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device)

    print(f"Loading input images: {img_paths}")
    imgs = load_images(img_paths, size=image_size, verbose=False)
    pairs = make_pairs(imgs, scene_graph='complete', symmetrize=True)

    print("Running inference on GPU...")
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    print("Solving global scene alignment...")
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer, verbose=False)

    rgb_imgs = scene.imgs
    pts3d = to_numpy(scene.get_pts3d())
    depths = to_numpy(scene.get_depthmaps())

    print("\nComputing quantitative metrics across all views...")
    md_table = compute_full_comparison_table(rgb_imgs, pts3d, depths,
                                            flying_thresh=flying_thresh,
                                            max_depth_ratio=max_depth_ratio)

    print(md_table)
    return md_table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark DUSt3R Edge-Refinement")
    parser.add_argument('--weights', type=str, default='../DUSt3R_ViTLarge_BaseDecoder_224_linear.pth', help='Path to weights')
    parser.add_argument('--images', nargs='+', default=['croco/assets/Chateau1.png', 'croco/assets/Chateau2.png'], help='Input images')
    parser.add_argument('--image_size', type=int, default=224, help='Image resolution')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--flying_thresh', type=float, default=0.08, help='Flying pixel threshold')
    parser.add_argument('--max_depth_ratio', type=float, default=0.08, help='Mesh webbing threshold')
    parser.add_argument('--save_md', type=str, default='benchmark_results.md', help='Save table to markdown file')

    args = parser.parse_args()
    table = benchmark_pair(args.weights, args.images, args.image_size, args.device, args.flying_thresh, args.max_depth_ratio)

    if args.save_md:
        with open(args.save_md, 'w') as f:
            f.write(table)
        print(f"\nSaved benchmark table to {args.save_md}")
