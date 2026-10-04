"""
Ground-Truth Evaluation of Depth Boundaries (Middlebury 2014)
-------------------------------------------------------------
Unlike metrics.py (which compares depth edges to RGB edges, i.e. to the same signal
an edge-guided method uses), everything here is measured against the true
Middlebury depth.

GT preparation:
    depth_gt = baseline * f / (disparity + doffs), then resized (nearest) and
    centre-cropped with exactly the same geometry as dust3r.utils.image.load_images,
    so GT pixel (u, v) is the same scene point as DUSt3R pixel (u, v).

Scale: DUSt3R depth is only defined up to scale, so each view is aligned with a
median ratio computed on the RAW prediction; the same scale is reused for every
method on that view, so pruning cannot "game" the alignment.

Metrics (per view):
    coverage        % of GT-valid pixels that still have a predicted depth
    AbsRel          mean |d - d_gt| / d_gt over all valid pixels
    AbsRel@edge     same, restricted to a band around true GT depth discontinuities
    delta1@edge     % of edge-band pixels with max(d/d_gt, d_gt/d) < 1.25
    coverage@edge   % of edge-band pixels that still have a predicted depth
    DBE acc / comp  Depth Boundary Errors (Koch et al., iBims-1, ECCVW 2018), in pixels:
                    acc  = mean distance from predicted depth edges to the nearest GT edge
                    comp = mean distance from GT depth edges to the nearest predicted edge
    bridges         % of mesh triangles spanning a TRUE depth gap that are still kept (webbing)
    false cuts      % of mesh triangles on a TRUE continuous surface that were removed
"""

import re

import numpy as np
import cv2
from PIL import Image


# ----------------------------------------------------------------------------- GT loading
def read_pfm(path):
    with open(path, 'rb') as f:
        header = f.readline().decode('latin-1').strip()
        assert header in ('Pf', 'PF'), f'not a PFM file: {path}'
        channels = 1 if header == 'Pf' else 3
        w, h = map(int, re.findall(r'\d+', f.readline().decode('latin-1')))
        scale = float(f.readline().decode('latin-1').strip())
        dtype = '<f4' if scale < 0 else '>f4'
        data = np.fromfile(f, dtype=dtype, count=w * h * channels)
    shape = (h, w, 3) if channels == 3 else (h, w)
    return np.flipud(data.reshape(shape)).astype(np.float32)


def read_calib(path):
    calib = {}
    with open(path) as f:
        for line in f:
            if '=' not in line:
                continue
            k, v = line.strip().split('=', 1)
            calib[k] = v
    cam0 = [float(x) for x in re.findall(r'[-\d.]+', calib['cam0'])]
    return dict(f=cam0[0], baseline=float(calib['baseline']), doffs=float(calib['doffs']),
                width=int(calib['width']), height=int(calib['height']))


def middlebury_depth(disp_path, calib_path):
    """ full-resolution GT depth in mm; 0 where unknown """
    disp = read_pfm(disp_path)
    c = read_calib(calib_path)
    valid = np.isfinite(disp) & (disp > 0)
    depth = np.zeros_like(disp)
    depth[valid] = c['baseline'] * c['f'] / (disp[valid] + c['doffs'])
    return depth


def to_dust3r_frame(depth_full, size=224, patch_size=16, square_ok=False):
    """ replicate the resize + centre-crop of dust3r.utils.image.load_images (nearest-neighbour for depth) """
    H1, W1 = depth_full.shape
    img = Image.fromarray(depth_full.astype(np.float32), mode='F')
    if size == 224:
        long_edge = round(size * max(W1 / H1, H1 / W1))
    else:
        long_edge = size
    S = max(img.size)
    new_size = tuple(int(round(x * long_edge / S)) for x in img.size)
    img = img.resize(new_size, Image.NEAREST)
    W, H = img.size
    cx, cy = W // 2, H // 2
    if size == 224:
        half = min(cx, cy)
        box = (cx - half, cy - half, cx + half, cy + half)
    else:
        halfw = ((2 * cx) // patch_size) * patch_size / 2
        halfh = ((2 * cy) // patch_size) * patch_size / 2
        if not square_ok and W == H:
            halfh = 3 * halfw / 4
        box = (cx - halfw, cy - halfh, cx + halfw, cy + halfh)
    return np.asarray(img.crop(box), dtype=np.float32)


# ----------------------------------------------------------------------------- helpers
def median_scale(pred, gt, mask):
    m = mask & (pred > 0) & (gt > 0) & np.isfinite(pred)
    return float(np.median(gt[m] / pred[m]))


def depth_edges(depth, valid, rel_jump=0.10):
    """ pixels with a > rel_jump relative depth jump to a valid 4-neighbour (both sides marked) """
    Z = depth.astype(np.float32)
    H, W = Z.shape
    edges = np.zeros((H, W), bool)
    for (a, b) in [((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
                   ((slice(0, -1), slice(None)), (slice(1, None), slice(None)))]:
        za, zb = Z[a], Z[b]
        ok = valid[a] & valid[b]
        jump = ok & (np.abs(za - zb) / (np.minimum(za, zb) + 1e-9) > rel_jump)
        edges[a] |= jump
        edges[b] |= jump
    return edges


def grid_faces(H, W):
    idx = np.arange(H * W).reshape(H, W)
    i1, i2 = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    i3, i4 = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    return np.concatenate([np.c_[i1, i2, i3], np.c_[i2, i3, i4]], axis=0)


def face_max_rel_jump(Z, faces):
    z = Z.reshape(-1)[faces]
    zmin = np.maximum(z.min(axis=1), 1e-9)
    return (z.max(axis=1) - z.min(axis=1)) / zmin


# ----------------------------------------------------------------------------- evaluation
def evaluate_view(pred, pred_valid, gt, scale, kept_faces=None,
                  rel_jump=0.10, band_radius=3, dbe_max_dist=10.0):
    """
    pred:        (H, W) predicted depth (unscaled DUSt3R units)
    pred_valid:  (H, W) bool, pixels the method keeps
    gt:          (H, W) GT depth in DUSt3R frame, 0 = unknown
    scale:       median scale (from the raw prediction) mapping pred -> GT units
    kept_faces:  (F,) bool over grid_faces(H, W): mesh triangles the method keeps.
                 Default = triangles whose 3 vertices are valid.
    """
    H, W = gt.shape
    gt_valid = gt > 0
    d = pred.astype(np.float32) * scale
    pv = pred_valid & np.isfinite(d) & (d > 0)

    m_all = gt_valid & pv
    absrel = np.abs(d - gt) / np.maximum(gt, 1e-9)

    gt_edge = depth_edges(gt, gt_valid, rel_jump)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * band_radius + 1, 2 * band_radius + 1))
    band = cv2.dilate(gt_edge.astype(np.uint8), k).astype(bool) & gt_valid
    m_band = band & pv
    ratio = np.maximum(d / np.maximum(gt, 1e-9), gt / np.maximum(d, 1e-9))

    res = dict(
        coverage=100.0 * m_all.sum() / max(gt_valid.sum(), 1),
        absrel=float(absrel[m_all].mean()) if m_all.any() else np.nan,
        absrel_edge=float(absrel[m_band].mean()) if m_band.any() else np.nan,
        delta1_edge=100.0 * float((ratio[m_band] < 1.25).mean()) if m_band.any() else np.nan,
        coverage_edge=100.0 * m_band.sum() / max(band.sum(), 1),
    )

    # Depth Boundary Errors: edges of the predicted depth (computed where GT is known)
    pred_edge = depth_edges(np.where(pv, d, 0), pv & gt_valid, rel_jump)
    dist_to_gt = cv2.distanceTransform((~gt_edge).astype(np.uint8), cv2.DIST_L2, 5)
    dist_to_pred = cv2.distanceTransform((~pred_edge).astype(np.uint8), cv2.DIST_L2, 5)
    acc_d = dist_to_gt[pred_edge]
    acc_d = acc_d[acc_d <= dbe_max_dist]
    res['dbe_acc'] = float(acc_d.mean()) if len(acc_d) else np.nan
    res['dbe_comp'] = float(np.minimum(dist_to_pred[gt_edge], dbe_max_dist).mean()) if gt_edge.any() else np.nan

    # Mesh webbing measured against the TRUE geometry
    faces = grid_faces(H, W)
    if kept_faces is None:
        kept_faces = pv.reshape(-1)[faces].all(axis=1)
    f_gt_ok = gt_valid.reshape(-1)[faces].all(axis=1)
    f_gap = f_gt_ok & (face_max_rel_jump(gt, faces) > rel_jump)
    f_cont = f_gt_ok & ~f_gap
    res['bridges'] = 100.0 * (kept_faces & f_gap).sum() / max(f_gap.sum(), 1)
    res['false_cuts'] = 100.0 * (~kept_faces & f_cont).sum() / max(f_cont.sum(), 1)
    return res


def mesh_prune_faces(depth, valid, max_rel_jump=0.08):
    """ same rule as edge_refiner.edge_aware_pts3d_to_trimesh, as a face mask over grid_faces """
    H, W = depth.shape
    faces = grid_faces(H, W)
    keep = valid.reshape(-1)[faces].all(axis=1)
    return keep & (face_max_rel_jump(depth, faces) <= max_rel_jump)
