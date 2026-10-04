# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------
import argparse
import math
import builtins
import datetime
import gradio
import os
import torch
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation

from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.edge_refiner import remove_flying_pixels, edge_aware_pts3d_to_trimesh
from dust3r.metrics import compute_full_comparison_table

import matplotlib.pyplot as pl


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser_weights.add_argument("--model_name", type=str, help="name of the model weights",
                                choices=["DUSt3R_ViTLarge_BaseDecoder_512_dpt",
                                         "DUSt3R_ViTLarge_BaseDecoder_512_linear",
                                         "DUSt3R_ViTLarge_BaseDecoder_224_linear"])
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default=None, help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    return parser


def set_print_with_timestamp(time_format="%Y-%m-%d %H:%M:%S"):
    builtin_print = builtins.print

    def print_with_timestamp(*args, **kwargs):
        now = datetime.datetime.now()
        formatted_date_time = now.strftime(time_format)

        builtin_print(f'[{formatted_date_time}] ', end='')  # print with time stamp
        builtin_print(*args, **kwargs)

    builtins.print = print_with_timestamp


import time

def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False,
                                 enable_edge_sharpening=True,
                                 flying_pixel_thresh=0.08,
                                 max_depth_ratio=0.08,
                                 depths=None):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)
    if depths is not None:
        depths = [to_numpy(d) for d in depths]

    scene = trimesh.Scene()

    # Apply edge-guided flying pixel removal on masks if enabled
    clean_masks = []
    for i in range(len(imgs)):
        m_i = mask[i].copy() if mask is not None else np.ones(pts3d[i].shape[:2], dtype=bool)
        d_i = depths[i] if depths is not None else None
        if enable_edge_sharpening:
            flying_clean_m, _ = remove_flying_pixels(pts3d[i], imgs[i], depth=d_i, flying_pixel_thresh=flying_pixel_thresh)
            m_i = m_i & flying_clean_m
        clean_masks.append(m_i)

    # full pointcloud
    if as_pointcloud:
        used_masks = clean_masks if enable_edge_sharpening else mask
        pts = np.concatenate([p[m] for p, m in zip(pts3d, used_masks)])
        col = np.concatenate([p[m] for p, m in zip(imgs, used_masks)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            d_i = depths[i] if depths is not None else None
            if enable_edge_sharpening:
                meshes.append(edge_aware_pts3d_to_trimesh(imgs[i], pts3d[i], valid=clean_masks[i], depth=d_i, max_edge_depth_ratio=max_depth_ratio))
            else:
                # Raw DUSt3R standard trimesh with original mask (shows full webbing)
                meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], valid=mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    # Unique timestamp prevents browser Model3D caching
    outfile = os.path.join(outdir, f'scene_{int(time.time() * 1000)}.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05,
                            enable_edge_sharpening=True, flying_pixel_thresh=0.08, max_depth_ratio=0.08):
    """
    extract 3D_model (glb file) from a reconstructed scene with edge-guided refinement
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    depths = to_numpy(scene.get_depthmaps())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent,
                                        enable_edge_sharpening=enable_edge_sharpening,
                                        flying_pixel_thresh=flying_pixel_thresh,
                                        max_depth_ratio=max_depth_ratio,
                                        depths=depths)


def get_reconstructed_scene(outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid,
                            enable_edge_sharpening=True, flying_pixel_thresh=0.08, max_depth_ratio=0.08):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    try:
        square_ok = model.square_ok
    except Exception as e:
        square_ok = False
    imgs = load_images(filelist, size=image_size, verbose=not silent, patch_size=model.patch_size, square_ok=square_ok)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=not silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size,
                                      enable_edge_sharpening=enable_edge_sharpening,
                                      flying_pixel_thresh=flying_pixel_thresh,
                                      max_depth_ratio=max_depth_ratio)

    # also return rgb, depth, confidence imgs, and edge diagnostics
    rgbimg = scene.imgs
    pts3d = to_numpy(scene.get_pts3d())
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d / confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))
        if enable_edge_sharpening:
            _, edge_viz = remove_flying_pixels(pts3d[i], rgbimg[i], flying_pixel_thresh=flying_pixel_thresh)
            imgs.append(edge_viz)

    return scene, outfile, imgs


def set_scenegraph_options(inputfiles, winsize, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize = max(2, math.ceil((num_files - 1) / 2))
    max_refid = max(1, num_files - 1)
    if scenegraph_type == "swin":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max(1, math.ceil((num_files - 1) / 2)),
                                minimum=1, maximum=max_winsize, step=1, visible=True)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=max_refid, step=1, visible=False)
    elif scenegraph_type == "oneref":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max(1, math.ceil((num_files - 1) / 2)),
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=max_refid, step=1, visible=True)
    else:
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max(1, math.ceil((num_files - 1) / 2)),
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=max_refid, step=1, visible=False)
    return winsize, refid


def main_demo(tmpdirname, model, device, image_size, server_name, server_port, silent=False):
    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, model, device, silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent)
    with gradio.Blocks(title="DUSt3R Demo") as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML('<h2 style="text-align: center;">DUSt3R Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            with gradio.Row():
                schedule = gradio.Dropdown(["linear", "cosine"],
                                           value='linear', label="schedule", info="For global alignment!")
                niter = gradio.Number(value=300, precision=0, minimum=0, maximum=5000,
                                      label="num_iterations", info="For global alignment!")
                scenegraph_type = gradio.Dropdown([("complete: all possible image pairs", "complete"),
                                                   ("swin: sliding window", "swin"),
                                                   ("oneref: match one image with all", "oneref")],
                                                  value='complete', label="Scenegraph",
                                                  info="Define how to make pairs",
                                                  interactive=True)
                winsize = gradio.Slider(label="Scene Graph: Window Size", value=1,
                                        minimum=1, maximum=2, step=1, visible=False)
                refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0, maximum=1, step=1, visible=False)

            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the confidence threshold
                min_conf_thr = gradio.Slider(label="min_conf_thr", value=3.0, minimum=1.0, maximum=20, step=0.1)
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.05, minimum=0.001, maximum=0.1, step=0.001)
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=False, label="As pointcloud")
                # two post process implemented
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")

            with gradio.Accordion("Edge-Guided Sharpening & Flying Pixel Filter (Enhancement Component)", open=True):
                with gradio.Row():
                    enable_edge_sharpening = gradio.Checkbox(value=True, label="Enable Edge-Guided Sharpening")
                    flying_pixel_thresh = gradio.Slider(label="Flying Pixel Disparity Threshold", value=0.08,
                                                        minimum=0.01, maximum=0.25, step=0.01,
                                                        info="Removes floating interpolated points near silhouette edges")
                    max_depth_ratio = gradio.Slider(label="Max Depth Discontinuity (Mesh Webbing)", value=0.08,
                                                   minimum=0.01, maximum=0.25, step=0.01,
                                                   info="Prevents stretched triangles across depth steps")
                with gradio.Row():
                    calc_metrics_btn = gradio.Button("📊 Compute Quantitative Benchmark Metrics (Raw vs. Refined)", variant="secondary")
                metrics_output = gradio.Markdown(value="*Click the button above to calculate quantitative academic metrics for this reconstruction.*")

            outmodel = gradio.Model3D()
            outgallery = gradio.Gallery(label='RGB / Depth / Confidence / Edge & Flying Pixel Diagnostics', columns=4, height="100%")

            # events
            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, winsize, refid, scenegraph_type],
                                   outputs=[winsize, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, winsize, refid, scenegraph_type],
                              outputs=[winsize, refid])

            scene_update_inputs = [scene, min_conf_thr, as_pointcloud, mask_sky,
                                   clean_depth, transparent_cams, cam_size,
                                   enable_edge_sharpening, flying_pixel_thresh, max_depth_ratio]

            recon_inputs = [inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                            mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid,
                            enable_edge_sharpening, flying_pixel_thresh, max_depth_ratio]

            run_btn.click(fn=recon_fun,
                          inputs=recon_inputs,
                          outputs=[scene, outmodel, outgallery])

            min_conf_thr.release(fn=model_from_scene_fun, inputs=scene_update_inputs, outputs=outmodel)
            cam_size.change(fn=model_from_scene_fun, inputs=scene_update_inputs, outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun, inputs=scene_update_inputs, outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun, inputs=scene_update_inputs, outputs=outmodel)
            clean_depth.change(fn=model_from_scene_fun, inputs=scene_update_inputs, outputs=outmodel)
            transparent_cams.change(fn=model_from_scene_fun, inputs=scene_update_inputs, outputs=outmodel)
            enable_edge_sharpening.change(fn=model_from_scene_fun, inputs=scene_update_inputs, outputs=outmodel)
            flying_pixel_thresh.release(fn=model_from_scene_fun, inputs=scene_update_inputs, outputs=outmodel)
            max_depth_ratio.release(fn=model_from_scene_fun, inputs=scene_update_inputs, outputs=outmodel)

            def on_compute_metrics(scene_state, flying_thresh, max_depth_ratio):
                if scene_state is None:
                    return "⚠️ *Please upload images and click 'Run' first to reconstruct the 3D scene before computing metrics.*"
                rgb_imgs = scene_state.imgs
                pts3d = to_numpy(scene_state.get_pts3d())
                depths = to_numpy(scene_state.get_depthmaps())
                return compute_full_comparison_table(rgb_imgs, pts3d, depths, flying_thresh=flying_thresh, max_depth_ratio=max_depth_ratio)

            calc_metrics_btn.click(fn=on_compute_metrics, inputs=[scene, flying_pixel_thresh, max_depth_ratio], outputs=metrics_output)
    demo.launch(share=False, server_name=server_name, server_port=server_port,
                css=""".gradio-container {margin: 0 !important; min-width: 100%};""")
