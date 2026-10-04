"""
Edge-Guided Depth Sharpening for DUSt3R (joint-bilateral weighted-mode filter)
------------------------------------------------------------------------------
DUSt3R regresses depth patch-by-patch with a smooth regression loss, so at an
occlusion boundary it predicts a *ramp* of depths in between the foreground and
the background instead of a step. Those in-between pixels are exactly the
"flying pixels", and meshing them produces the "webbing" curtains.

Pruning those pixels hides the problem but deletes geometry. This module instead
*re-assigns* every ramp pixel to the side it actually belongs to:

  1. Ramp band: pixels whose local log-depth range (max - min in a small window)
     exceeds log(1 + rel_jump), i.e. pixels that sit on a depth transition.
  2. For each band pixel p, look at its neighbours q in a (2R+1)^2 window and
     split them into a NEAR and a FAR cluster at the middle of the local
     log-depth range.
  3. Each neighbour votes for its cluster with a joint-bilateral weight
         w(p, q) = exp(-|p-q|^2 / 2 s_s^2) * exp(-|Lab(p)-Lab(q)|^2 / 2 s_c^2) * c(q)
     (spatial closeness x colour similarity x optional DUSt3R confidence);
     neighbours that are themselves in the ramp band get down-weighted.
  4. p takes the weighted median depth of the winning cluster. A few Jacobi
     iterations collapse the ramp into a step aligned with the image edge.

This is a weighted-mode filter (Min, Lu & Do, "Depth Video Enhancement Based on
Weighted Mode Filtering", IEEE TIP 2012) restricted to the discontinuity band and
guided by the RGB image, so depth edges snap to colour edges.
"""

import numpy as np
import cv2

from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates


def rgb_to_lab(img):
    """ (H, W, 3) RGB in [0, 1] or uint8 -> float32 CIELab (L in [0, 100]) """
    img = to_numpy(img)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)


def detect_ramp_band(depth, radius=2, rel_jump=0.10, valid=None):
    """
    Pixels lying on a depth transition: max/min depth ratio in a (2r+1)^2 window > 1 + rel_jump.
    Returns (H, W) bool.
    """
    Z = to_numpy(depth).astype(np.float32)
    if valid is None:
        valid = np.isfinite(Z) & (Z > 0)
    logZ = np.where(valid, np.log(np.maximum(Z, 1e-6)), 0).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    hi = cv2.dilate(np.where(valid, logZ, -np.inf).astype(np.float32), k)
    lo = cv2.erode(np.where(valid, logZ, np.inf).astype(np.float32), k)
    return valid & np.isfinite(hi - lo) & ((hi - lo) > np.log1p(rel_jump))


def _weighted_median(values, weights):
    """ row-wise weighted median of (N, K) arrays (weights >= 0, every row has some weight) """
    order = np.argsort(values, axis=1)
    v = np.take_along_axis(values, order, axis=1)
    w = np.take_along_axis(weights, order, axis=1)
    cw = np.cumsum(w, axis=1)
    idx = np.argmax(cw >= 0.5 * cw[:, -1:], axis=1)
    return v[np.arange(len(v)), idx]


def sharpen_depth(depth, rgb, conf=None, valid=None,
                  rel_jump=0.10, band_radius=2, win_radius=4,
                  sigma_space=3.0, sigma_color=12.0, band_weight=0.25, n_iters=3):
    """
    Edge-guided sharpening of a single depth map.

    depth: (H, W) camera depth (use scene.get_depthmaps(), NOT pts3d[..., 2])
    rgb:   (H, W, 3) image in [0, 1]
    conf:  optional (H, W) DUSt3R confidence (>= 1) used as an extra voter weight. Off by
           default: on Middlebury it made every edge metric worse, because DUSt3R is least
           confident exactly at the boundaries where the votes are needed.
    rel_jump:    relative depth jump that counts as a discontinuity (0.10 = 10%)
    band_radius: half-size of the window used to find the ramp band
    win_radius:  half-size of the voting window
    sigma_space: spatial std (px) of the bilateral weight
    sigma_color: colour std (CIELab units) of the bilateral weight
    band_weight: vote multiplier for neighbours that are themselves in the ramp
    n_iters:     number of Jacobi iterations

    Returns: sharpened depth (H, W) float32, band mask (H, W) bool
    """
    Z = to_numpy(depth).astype(np.float32)
    H, W = Z.shape
    if valid is None:
        valid = np.isfinite(Z) & (Z > 0)
    valid = to_numpy(valid).astype(bool)

    band = detect_ramp_band(Z, radius=band_radius, rel_jump=rel_jump, valid=valid)
    ys, xs = np.nonzero(band)
    if len(ys) == 0:
        return Z.copy(), band

    logZ = np.where(valid, np.log(np.maximum(Z, 1e-6)), 0).astype(np.float32)
    lab = rgb_to_lab(rgb)
    if conf is None:
        cw = np.ones_like(Z)
    else:
        # DUSt3R confidence is 1 + exp(x); log(conf) is ~0 at uncertain edges and grows inside surfaces
        cw = np.log(np.maximum(to_numpy(conf).astype(np.float32), 1.0 + 1e-3))
        cw = cw / (cw.max() + 1e-6)
    voter_w = cw * np.where(band, band_weight, 1.0) * valid

    R = win_radius
    dy, dx = np.mgrid[-R:R + 1, -R:R + 1]
    dy, dx = dy.ravel(), dx.ravel()
    w_space = np.exp(-(dy ** 2 + dx ** 2) / (2 * sigma_space ** 2)).astype(np.float32)

    def pad(a, value):
        widths = ((R, R), (R, R)) + ((0, 0),) * (a.ndim - 2)
        return np.pad(a, widths, mode='constant', constant_values=value)

    # neighbour indices of every band pixel in padded coordinates: (N, K)
    ny = ys[:, None] + R + dy[None]
    nx = xs[:, None] + R + dx[None]

    lab_p = pad(lab, 0)
    voter_p = pad(voter_w.astype(np.float32), 0)

    # static part of the weights: space x colour x confidence (padding has weight 0)
    d_lab = lab_p[ny, nx] - lab[ys, xs][:, None, :]
    w_color = np.exp(-np.sum(d_lab ** 2, axis=-1) / (2 * sigma_color ** 2))
    w_static = w_space[None] * w_color * voter_p[ny, nx]

    cur = logZ.copy()
    for _ in range(n_iters):
        vals = pad(cur, 0)[ny, nx]
        w = w_static
        has_vote = w > 0
        lo = np.where(has_vote, vals, np.inf).min(axis=1, keepdims=True)
        hi = np.where(has_vote, vals, -np.inf).max(axis=1, keepdims=True)
        near = vals <= 0.5 * (lo + hi)

        w_near = np.where(near, w, 0).sum(axis=1)
        w_far = np.where(~near, w, 0).sum(axis=1)
        pick_near = (w_near >= w_far)[:, None]
        w_sel = np.where(near == pick_near, w, 0)

        ok = w_sel.sum(axis=1) > 0
        new = cur[ys, xs].copy()
        new[ok] = _weighted_median(vals[ok], w_sel[ok])
        cur[ys, xs] = new

    out = np.where(valid, np.exp(cur), Z).astype(np.float32)
    return out, band


def sharpen_scene(scene, use_conf=False, **kw):
    """
    Apply sharpen_depth to every view of a DUSt3R scene (PairViewer / global aligner output)
    and rebuild the 3D points from the sharpened depth with the scene's own cameras.

    Returns dict(depths=[...], pts3d=[...], bands=[...]) as numpy arrays.
    """
    imgs = scene.imgs
    depths = [to_numpy(d) for d in scene.get_depthmaps()]
    confs = [to_numpy(c) for c in scene.im_conf]
    K = to_numpy(scene.get_intrinsics())
    poses = to_numpy(scene.get_im_poses())

    out = dict(depths=[], pts3d=[], bands=[])
    for i in range(len(imgs)):
        H, W = imgs[i].shape[:2]
        d = depths[i].reshape(H, W)
        d_sharp, band = sharpen_depth(d, imgs[i], conf=confs[i] if use_conf else None, **kw)
        pts, _ = depthmap_to_absolute_camera_coordinates(d_sharp, K[i], poses[i])
        out['depths'].append(d_sharp)
        out['pts3d'].append(pts.astype(np.float32))
        out['bands'].append(band)
    return out
