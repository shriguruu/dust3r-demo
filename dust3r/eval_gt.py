#!/usr/bin/env python3
"""
Ground-truth benchmark: raw DUSt3R vs. baselines vs. edge-guided sharpening
---------------------------------------------------------------------------
Runs DUSt3R once per Middlebury scene, then applies every method to the SAME output
and scores it against the Middlebury GT depth (see dust3r/gt_eval.py for metrics).

Methods
    raw                  DUSt3R depth as is
    conf (matched)       DUSt3R's own confidence masking, dropping as many pixels as
                         'prune (existing)' does -> same budget, only *which* pixels differs
    median5              5x5 median filter on depth (classic, edge-agnostic baseline)
    prune (existing)     edge_refiner.remove_flying_pixels + edge-aware mesh pruning
    raw + mesh prune     only the edge-aware mesh pruning, no point removal
    sharpen (ours)       edge_sharpen.sharpen_depth (re-assigns ramp pixels, deletes nothing)
    sharpen + mesh prune sharpening followed by edge-aware mesh pruning
    ablation: no colour  sharpening with the colour term disabled (pure depth mode filter)
    ablation: + conf     sharpening with DUSt3R confidence as an extra voter weight

Prerequisite: python download_middlebury_gt.py
Usage:        python eval_gt.py --weights ../DUSt3R_ViTLarge_BaseDecoder_224_linear.pth
"""

import argparse
import os

import numpy as np
import cv2

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
from dust3r.edge_refiner import remove_flying_pixels
from dust3r.edge_sharpen import sharpen_depth
from dust3r.gt_eval import (middlebury_depth, to_dust3r_frame, median_scale,
                            evaluate_view, mesh_prune_faces)

SCENES = ['02_motorcycle', '03_playtable', '05_pipes']

METRICS = [  # key, header, format, better
    ('coverage', 'Coverage %', '{:.1f}', 'higher'),
    ('absrel', 'AbsRel', '{:.4f}', 'lower'),
    ('absrel_edge', 'AbsRel@edge', '{:.4f}', 'lower'),
    ('delta1_edge', 'δ1@edge %', '{:.1f}', 'higher'),
    ('coverage_edge', 'Coverage@edge %', '{:.1f}', 'higher'),
    ('dbe_acc', 'DBE acc (px)', '{:.2f}', 'lower'),
    ('dbe_comp', 'DBE comp (px)', '{:.2f}', 'lower'),
    ('bridges', 'Webbing bridges %', '{:.1f}', 'lower'),
    ('false_cuts', 'False cuts %', '{:.2f}', 'lower'),
]


def resolve_path(p, script_dir):
    for cand in (p, os.path.join(script_dir, p), os.path.join(script_dir, '..', os.path.basename(p))):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return p


def run_methods(img, pts3d, depth, conf, args):
    """ returns {name: (depth, valid_mask, kept_faces or None)} """
    all_valid = np.isfinite(depth) & (depth > 0)
    clean, _ = remove_flying_pixels(pts3d, img, depth=depth, flying_pixel_thresh=args.prune_thr)
    # confidence threshold that removes the same number of pixels as the existing pruning
    conf_thr = np.quantile(conf[all_valid], 1.0 - clean[all_valid].mean())

    out = {}
    out['raw'] = (depth, all_valid, None)
    out['conf (matched)'] = (depth, all_valid & (conf >= conf_thr), None)
    out['median5'] = (cv2.medianBlur(depth.astype(np.float32), 5), all_valid, None)
    out['prune (existing)'] = (depth, clean, mesh_prune_faces(depth, clean, args.mesh_thr))
    out['raw + mesh prune'] = (depth, all_valid, mesh_prune_faces(depth, all_valid, args.mesh_thr))

    kw = dict(rel_jump=args.rel_jump, win_radius=args.win_radius, n_iters=args.n_iters)
    sharp, band = sharpen_depth(depth, img, sigma_color=args.sigma_color, **kw)
    out['sharpen (ours)'] = (sharp, all_valid, None)
    out['sharpen + mesh prune (ours)'] = (sharp, all_valid, mesh_prune_faces(sharp, all_valid, args.mesh_thr))

    # ablations: colour guidance off / DUSt3R confidence used as voter weight
    no_color, _ = sharpen_depth(depth, img, sigma_color=1e6, **kw)
    out['ablation: no colour'] = (no_color, all_valid, mesh_prune_faces(no_color, all_valid, args.mesh_thr))
    with_conf, _ = sharpen_depth(depth, img, conf=conf, sigma_color=args.sigma_color, **kw)
    out['ablation: + conf weight'] = (with_conf, all_valid, mesh_prune_faces(with_conf, all_valid, args.mesh_thr))
    return out, band


def colorize_depth(d, valid, lo, hi):
    inv = np.where(valid, 1.0 / np.maximum(d, 1e-9), 0)
    x = np.clip((inv - 1.0 / hi) / (1.0 / lo - 1.0 / hi + 1e-12), 0, 1)
    c = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    c[~valid] = 0
    return c


def colorize_err(d, gt, valid, vmax=0.3):
    e = np.where(valid & (gt > 0), np.abs(d - gt) / np.maximum(gt, 1e-9), 0)
    c = cv2.applyColorMap((np.clip(e / vmax, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    c[~(valid & (gt > 0))] = 0
    return c


def save_panel(path, img, gt, raw, sharp, band, scale, up=2):
    gv = gt > 0
    lo, hi = np.percentile(gt[gv], 1), np.percentile(gt[gv], 99)
    rv = np.isfinite(raw) & (raw > 0)
    tiles = [
        ('RGB', cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)),
        ('GT depth', colorize_depth(gt, gv, lo, hi)),
        ('raw DUSt3R', colorize_depth(raw * scale, rv, lo, hi)),
        ('sharpened', colorize_depth(sharp * scale, rv, lo, hi)),
        ('ramp band', cv2.cvtColor((band * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)),
        ('raw rel. error', colorize_err(raw * scale, gt, rv)),
        ('sharpened rel. error', colorize_err(sharp * scale, gt, rv)),
    ]
    rows = []
    for name, t in tiles:
        t = cv2.resize(t, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(t)
    cv2.imwrite(path, np.concatenate(rows, axis=1))


def to_markdown(title, results):
    head = '| Method | ' + ' | '.join(m[1] for m in METRICS) + ' |'
    sep = '| :--- |' + ' :---: |' * len(METRICS)
    arrows = '| *(better)* | ' + ' | '.join('↑' if m[3] == 'higher' else '↓' for m in METRICS) + ' |'
    best = {}
    for key, _, _, better in METRICS:
        vals = [r[key] for r in results.values() if np.isfinite(r[key])]
        if vals:
            best[key] = max(vals) if better == 'higher' else min(vals)
    lines = [f'### {title}', '', head, sep, arrows]
    for name, r in results.items():
        cells = []
        for key, _, fmt, _ in METRICS:
            s = fmt.format(r[key]) if np.isfinite(r[key]) else 'n/a'
            cells.append(f'**{s}**' if key in best and np.isclose(r[key], best[key]) else s)
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', default='../DUSt3R_ViTLarge_BaseDecoder_224_linear.pth')
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--device', default='cuda')
    p.add_argument('--scenes', nargs='+', default=SCENES)
    p.add_argument('--pairs', default=os.path.join(script_dir, '..', 'sample_pairs'))
    p.add_argument('--gt', default=os.path.join(script_dir, '..', 'data', 'middlebury_gt'))
    p.add_argument('--out', default=os.path.join(script_dir, 'results'))
    p.add_argument('--prune_thr', type=float, default=0.08, help='flying-pixel threshold of the existing pruning')
    p.add_argument('--mesh_thr', type=float, default=0.08, help='edge-aware mesh pruning threshold')
    p.add_argument('--rel_jump', type=float, default=0.10, help='sharpening: depth jump defining the ramp band')
    p.add_argument('--win_radius', type=int, default=4, help='sharpening: voting window half-size')
    p.add_argument('--sigma_color', type=float, default=12.0, help='sharpening: colour std (Lab)')
    p.add_argument('--n_iters', type=int, default=3, help='sharpening: iterations')
    args = p.parse_args()

    weights = resolve_path(args.weights, script_dir)
    os.makedirs(os.path.join(args.out, 'vis'), exist_ok=True)
    model = AsymmetricCroCo3DStereo.from_pretrained(weights).to(args.device)

    per_scene = {}
    for scene_name in args.scenes:
        gt_dir = os.path.join(args.gt, scene_name)
        if not os.path.exists(os.path.join(gt_dir, 'disp0.pfm')):
            raise SystemExit(f'missing GT for {scene_name}: run  python download_middlebury_gt.py  first')
        views = [os.path.join(args.pairs, scene_name, f'view{i}.png') for i in (1, 2)]
        print(f'== {scene_name}')

        imgs = load_images(views, size=args.image_size, verbose=False)
        pairs = make_pairs(imgs, scene_graph='complete', symmetrize=True)
        output = inference(pairs, model, args.device, batch_size=1, verbose=False)
        scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PairViewer, verbose=False)

        depths = [to_numpy(d) for d in scene.get_depthmaps()]
        confs = [to_numpy(c) for c in scene.im_conf]
        pts3d = [to_numpy(p) for p in scene.get_pts3d()]
        view_results = []
        for i in range(2):
            img = scene.imgs[i]
            H, W = img.shape[:2]
            depth = depths[i].reshape(H, W).astype(np.float32)
            gt = to_dust3r_frame(middlebury_depth(os.path.join(gt_dir, f'disp{i}.pfm'),
                                                  os.path.join(gt_dir, 'calib.txt')),
                                 size=args.image_size)
            assert gt.shape == (H, W), f'GT {gt.shape} vs pred {(H, W)}'
            scale = median_scale(depth, gt, gt > 0)

            methods, band = run_methods(img, pts3d[i], depth, confs[i], args)
            view_results.append({name: evaluate_view(d, v, gt, scale, kept_faces=f, rel_jump=args.rel_jump)
                                 for name, (d, v, f) in methods.items()})
            save_panel(os.path.join(args.out, 'vis', f'{scene_name}_view{i + 1}.png'),
                       img, gt, depth, methods['sharpen (ours)'][0], band, scale)

        per_scene[scene_name] = {name: {k: float(np.nanmean([vr[name][k] for vr in view_results]))
                                        for k, *_ in METRICS}
                                 for name in view_results[0]}

    overall = {name: {k: float(np.nanmean([per_scene[s][name][k] for s in per_scene])) for k, *_ in METRICS}
               for name in next(iter(per_scene.values()))}

    md = ['# Ground-truth evaluation on Middlebury 2014', '',
          f'Model: `{os.path.basename(weights)}` @ {args.image_size}px. '
          f'Each row is measured, not derived. Bold = best in column. '
          f'Edge band = ±3 px around true depth jumps > {args.rel_jump:.0%}.', '',
          to_markdown(f'Mean over {len(per_scene)} scenes × 2 views', overall)]
    for s, r in per_scene.items():
        md.append(to_markdown(s, r))
    md_text = '\n'.join(md)
    print(md_text)
    with open(os.path.join(args.out, 'gt_eval.md'), 'w', encoding='utf-8') as f:
        f.write(md_text)
    print(f'saved {os.path.join(args.out, "gt_eval.md")} and panels in {os.path.join(args.out, "vis")}')
