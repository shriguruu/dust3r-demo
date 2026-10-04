"""
Edge-Guided Depth Sharpening and Flying Pixel Removal for DUSt3R
----------------------------------------------------------------
This module enhances DUSt3R 3D pointmaps and meshes by:
1. Identifying high-frequency RGB radiometric boundaries.
2. Pruning "flying pixels" and edge bleeding using true camera depth disparities.
3. Preventing triangle tearing ("webbing") across depth discontinuities in 3D meshes.
4. Supplying diagnostic visual feedback.
"""

import numpy as np
import cv2
import trimesh
from dust3r.utils.device import to_numpy


def compute_rgb_edges(img, low_thresh=50, high_thresh=150):
    """
    Compute edge map from RGB image using Canny and Sobel gradient magnitude.
    img: (H, W, 3) in range [0, 1] or [0, 255] uint8.
    Returns: (H, W) float32 normalized edge map in [0, 1].
    """
    img_np = to_numpy(img)
    if img_np.dtype != np.uint8:
        img_u8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    else:
        img_u8 = img_np

    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    
    # Canny edges
    canny = cv2.Canny(gray, low_thresh, high_thresh) / 255.0

    # Sobel gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    mag = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Combined edge map
    edges = np.clip(0.6 * canny + 0.4 * mag, 0.0, 1.0)
    return edges


def compute_depth_discontinuity_mask(depth, max_rel_diff=0.08):
    """
    Identifies pixels that have severe depth jumps compared to their immediate neighbors.
    depth: (H, W) camera depth.
    max_rel_diff: Maximum allowed relative difference |Z1 - Z2| / min(Z1, Z2).
    Returns:
        discontinuity_mask: (H, W) bool, True where a severe depth discontinuity exists.
        max_rel_diffs: (H, W) float32, relative depth difference field.
    """
    Z = to_numpy(depth).astype(np.float32)
    if Z.ndim == 3 and Z.shape[-1] == 3:
        Z = Z[..., 2]
    H, W = Z.shape

    # Pad depth for neighbor comparison
    Z_pad = np.pad(Z, 1, mode='edge')
    
    # Compare with 4 immediate neighbors: up, down, left, right
    center = Z_pad[1:-1, 1:-1]
    up     = Z_pad[:-2,  1:-1]
    down   = Z_pad[2:,   1:-1]
    left   = Z_pad[1:-1, :-2]
    right  = Z_pad[1:-1, 2:]

    neighbors = [up, down, left, right]
    rel_diffs = []
    for n in neighbors:
        denom = np.minimum(np.abs(center), np.abs(n)) + 1e-6
        diff = np.abs(center - n) / denom
        rel_diffs.append(diff)

    max_rel = np.maximum.reduce(rel_diffs)
    discontinuity = max_rel > max_rel_diff
    return discontinuity, max_rel.astype(np.float32)


def remove_flying_pixels(pts3d, rgb_img, depth=None, confidence=None, 
                         flying_pixel_thresh=0.08, 
                         conf_thresh=0.0):
    """
    Prunes floating interpolated points along silhouette boundaries (flying pixels).
    """
    pts3d = to_numpy(pts3d)
    rgb_img = to_numpy(rgb_img)
    H, W, _ = pts3d.shape

    if depth is not None:
        Z = to_numpy(depth).astype(np.float32)
        if Z.ndim == 3:
            Z = Z[..., 2]
    else:
        Z = pts3d[..., 2].astype(np.float32)

    # 1. Compute depth discontinuity using true camera depth
    discontinuity, max_rel = compute_depth_discontinuity_mask(Z, max_rel_diff=flying_pixel_thresh)

    # 2. Flying pixels are boundary pixels with large disparity jumps
    flying_pixels = discontinuity

    # Dilate flying pixel mask by 1 pixel to clean transition ramps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    flying_pixels_dilated = cv2.dilate(flying_pixels.astype(np.uint8), kernel).astype(bool)

    # Base valid mask
    clean_mask = ~flying_pixels_dilated

    # If confidence is provided, enforce it too
    if confidence is not None:
        confidence = to_numpy(confidence)
        clean_mask = clean_mask & (confidence > conf_thresh)

    # Ensure valid finite depth
    valid_depth = np.isfinite(Z) & (Z > 1e-3)
    clean_mask = clean_mask & valid_depth

    # 3. Create diagnostic visualization map
    edge_map = compute_rgb_edges(rgb_img)
    edge_viz = np.zeros((H, W, 3), dtype=np.float32)
    edge_viz[..., 0] = flying_pixels_dilated.astype(np.float32)  # Red = removed flying pixels
    edge_viz[..., 1] = edge_map                                  # Green = RGB edges
    edge_viz[..., 2] = np.clip(max_rel / 0.2, 0.0, 1.0)          # Blue = depth disparity
    edge_viz = np.clip(edge_viz * 255.0, 0, 255).astype(np.uint8)

    return clean_mask, edge_viz


def edge_aware_pts3d_to_trimesh(img, pts3d, valid=None, depth=None, max_edge_depth_ratio=0.08):
    """
    Converts a pointmap to a trimesh representation while pruning
    stretched 'webbing' triangles that cross depth boundaries.
    """
    img = to_numpy(img)
    pts3d = to_numpy(pts3d)
    H, W, THREE = img.shape
    assert THREE == 3

    vertices = pts3d.reshape(-1, 3)

    if depth is not None:
        Z = to_numpy(depth).astype(np.float32).reshape(-1)
    else:
        Z = pts3d[..., 2].astype(np.float32).reshape(-1)

    # 2 triangles per quad
    idx = np.arange(len(vertices)).reshape(H, W)
    idx1 = idx[:-1, :-1].ravel()  # top-left
    idx2 = idx[:-1, +1:].ravel()  # top-right
    idx3 = idx[+1:, :-1].ravel()  # bottom-left
    idx4 = idx[+1:, +1:].ravel()  # bottom-right

    t1 = np.c_[idx1, idx2, idx3]
    t2 = np.c_[idx2, idx3, idx4]
    faces = np.concatenate((t1, t2), axis=0)

    # Triangle colors from vertex RGB
    colors_quad = img[:-1, :-1].reshape(-1, 3)
    colors_quad4 = img[+1:, +1:].reshape(-1, 3)
    face_colors = np.concatenate((colors_quad, colors_quad4), axis=0)

    # Double-sided faces to cancel culling
    faces_rev = faces[:, ::-1]
    faces = np.concatenate((faces, faces_rev), axis=0)
    face_colors = np.concatenate((face_colors, face_colors), axis=0)

    # 1. Mask-based validity
    if valid is not None:
        valid_flat = to_numpy(valid).ravel()
        valid_faces = valid_flat[faces].all(axis=-1)
        faces = faces[valid_faces]
        face_colors = face_colors[valid_faces]

    # 2. Depth discontinuity pruning (prevents tearing / webbing across boundaries)
    if len(faces) > 0 and max_edge_depth_ratio > 0:
        z_v0 = Z[faces[:, 0]]
        z_v1 = Z[faces[:, 1]]
        z_v2 = Z[faces[:, 2]]

        # Max relative depth difference across any edge in the triangle
        d01 = np.abs(z_v0 - z_v1) / (np.minimum(np.abs(z_v0), np.abs(z_v1)) + 1e-6)
        d12 = np.abs(z_v1 - z_v2) / (np.minimum(np.abs(z_v1), np.abs(z_v2)) + 1e-6)
        d20 = np.abs(z_v2 - z_v0) / (np.minimum(np.abs(z_v2), np.abs(z_v0)) + 1e-6)

        max_tri_disparity = np.maximum(d01, np.maximum(d12, d20))
        non_webbing_faces = max_tri_disparity <= max_edge_depth_ratio

        faces = faces[non_webbing_faces]
        face_colors = face_colors[non_webbing_faces]

    return dict(vertices=vertices, face_colors=face_colors, faces=faces)
