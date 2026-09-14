# --------------------------------------------------------
# Multi-view RGB-D tracking demo for Any4D with Rerun
#
# Runs RGB-D inference with the official checkpoint (no MoGe) and logs the
# reconstruction together with tracking results to Rerun:
#   - current-frame RGB and color-mapped input depth side by side
#   - current-frame camera transform, metric point cloud and scene flow
#   - trajectories of a fixed set of reference-view points over time
#   - all-frame overlays and predicted depth are opt-in
#
# Point tracks are obtained by adding the predicted world-frame scene flow of
# every view to the pointmap of the reference view (view 0), following the
# tracking logic of scripts/demo_inference.py.
# --------------------------------------------------------
import argparse

import numpy as np
import rerun as rr
import torch
from rerun.blueprint import (
    Blueprint,
    Horizontal,
    Spatial2DView,
    Spatial3DView,
    Vertical,
)

from any4d.utils.geometry import (
    quaternion_to_rotation_matrix,
    recover_pinhole_intrinsics_from_ray_directions,
)
from any4d.utils.inference import (
    loss_of_one_batch_multi_view,
    postprocess_model_outputs_for_inference,
    preprocess_input_views_for_inference,
    validate_input_views_for_inference,
)
from any4d.utils.misc import seed_everything
from any4d.utils.rgbd import (
    build_image_normalizer,
    compute_target_size,
    find_pairs,
    init_inference_model,
    load_intrinsics,
    load_rgbd_view,
    resolve_session_paths,
)
from any4d.utils.viz import script_add_rerun_args

DEFAULT_RERUN_URL = "rerun+http://127.0.0.1:9877/proxy"


def get_parser():
    parser = argparse.ArgumentParser(description="Multi-view RGB-D tracking demo with Any4D + Rerun.")
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Session root directory. Resolves <session>/rgb (or Color_*.jpg), "
        "<session>/mapped_depth and <session>/camera_parameters/rgb_camera_param.yaml automatically.",
    )
    parser.add_argument("--rgb_dir", type=str, default=None, help="RGB directory (alternative to --session).")
    parser.add_argument("--depth_dir", type=str, default=None, help="Depth directory (alternative to --session).")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/any4d_4v_combined.pth",
        help="Path to the official Any4D pretrained checkpoint.",
    )
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--task", type=str, default="rgbd")
    parser.add_argument("--machine", type=str, default="local")

    # Camera / depth conventions
    parser.add_argument(
        "--camera_params",
        type=str,
        default=None,
        help="Camera parameter file (YAML/JSON) with ROS-style 'K', 'width' and 'height' fields.",
    )
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=0.001,
        help="Scale factor that converts stored depth values to meters (default: 0.001 for mm uint16 PNGs).",
    )
    parser.add_argument("--max_depth", type=float, default=None, help="Optional maximum valid depth in meters.")

    # Data selection
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None, help="Exclusive end index (default: all).")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Load every nth frame. Increase it to cover a longer sequence with fewer views.",
    )
    parser.add_argument("--resolution_set", type=int, default=518, choices=[518, 512])

    # Tracking / visualization
    parser.add_argument(
        "--max_track_points",
        type=int,
        default=500,
        help="Maximum number of reference-view points to track and draw as trajectories.",
    )
    parser.add_argument(
        "--motion_threshold",
        type=float,
        default=0.05,
        help="Minimum scene flow magnitude in meters for a scene flow arrow to be drawn.",
    )
    parser.add_argument(
        "--show_all_frames",
        action="store_true",
        help="Also keep the point cloud and scene flow of every view as a static overlay (off by default).",
    )
    parser.add_argument(
        "--show_tracks",
        action="store_true",
        help="Show the trajectories of the tracked reference-view points (off by default).",
    )
    parser.add_argument(
        "--show_camera_path",
        action="store_true",
        help="Show the camera trajectory line (off by default).",
    )
    parser.add_argument(
        "--show_predicted_depth",
        action="store_true",
        help="Also show the predicted depth next to the RGB and input depth (off by default).",
    )
    parser.add_argument("--no_amp", action="store_true", help="Disable bfloat16 autocast.")
    parser.add_argument("--seed", type=int, default=0)

    script_add_rerun_args(parser)
    # Default to the port used by `pixi run rerun-serve`/`rerun-view` and make
    # `--save` the default mode instead of silently connecting to a viewer.
    parser.set_defaults(connect=False, headless=True, url=DEFAULT_RERUN_URL)
    return parser


def resolve_inputs(args):
    "Resolve rgb/depth/camera-parameter paths from --session or explicit directories."
    rgb_dir, depth_dir, camera_params = args.rgb_dir, args.depth_dir, args.camera_params
    if args.session is not None:
        rgb_dir, depth_dir, camera_params = resolve_session_paths(args.session, camera_params)
    if rgb_dir is None or depth_dir is None:
        raise ValueError("Provide --session or both --rgb_dir and --depth_dir.")
    return rgb_dir, depth_dir, camera_params


def get_colormap(name):
    "Return a matplotlib colormap callable."
    import matplotlib

    if hasattr(matplotlib, "colormaps"):
        return matplotlib.colormaps[name]
    return matplotlib.cm.get_cmap(name)


def cmap_colors(values, name="rainbow"):
    "Map scalar values to uint8 RGB colors using a matplotlib colormap."
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    vmin, vmax = float(values.min()), float(values.max())
    normalized = (values - vmin) / max(vmax - vmin, 1e-8)
    return (get_colormap(name)(normalized)[:, :3] * 255).astype(np.uint8)


def depth_color_range(depths, low_percentile=2.0, high_percentile=98.0):
    "Fixed color range (meters) for a sequence of depth maps."
    valid = np.concatenate([depth[depth > 0].reshape(-1) for depth in depths])
    if valid.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(valid, [low_percentile, high_percentile])
    if vmax <= vmin:
        vmax = vmin + 1e-3
    return float(vmin), float(vmax)


def colorize_depth(depth, vmin, vmax, name="turbo"):
    "Color-map a depth map with a fixed range; invalid pixels become black."
    normalized = np.clip((depth - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    colors = (get_colormap(name)(normalized)[..., :3] * 255).astype(np.uint8)
    colors[depth <= 0] = 0
    return colors


def build_blueprint(show_predicted_depth=False):
    "Fixed layout: RGB | input depth on top, 3D reconstruction and tracks below."
    views_2d = [
        Spatial2DView(origin="world/camera/pinhole/rgb", name="RGB"),
        Spatial2DView(origin="world/camera/pinhole/depth_input", name="Input depth"),
    ]
    if show_predicted_depth:
        views_2d.append(Spatial2DView(origin="world/camera/pinhole/depth_pred", name="Predicted depth"))
    return Blueprint(
        Vertical(
            Horizontal(*views_2d),
            Spatial3DView(origin="world", name="3D + tracks"),
        ),
        collapse_panels=True,
    )


def gather_view_data(processed, input_views):
    "Collect the per-view prediction arrays used for logging."
    data = {
        "images": [],
        "pts3d": [],
        "depth_z": [],
        "input_depths": [],
        "cam_trans": [],
        "cam_quats": [],
        "intrinsics": [],
        "scene_flows": [],
    }
    for prediction, view in zip(processed, input_views):
        data["images"].append(prediction["img_no_norm"][0].detach().float().cpu().numpy())
        data["pts3d"].append(prediction["pts3d"][0].detach().float().cpu().numpy())
        data["depth_z"].append(prediction["depth_z"][0, ..., 0].detach().float().cpu().numpy())
        data["input_depths"].append(view["depth_z"][0, ..., 0].detach().float().cpu().numpy())
        data["cam_trans"].append(prediction["cam_trans"][0].detach().float().cpu().numpy())
        data["cam_quats"].append(prediction["cam_quats"][0].detach().float().cpu().numpy())
        data["intrinsics"].append(
            recover_pinhole_intrinsics_from_ray_directions(prediction["ray_directions"])[0].cpu().numpy()
        )
        data["scene_flows"].append(prediction["scene_flow"][0].detach().float().cpu().numpy())
    return data


def select_track_indices(valid_mask, max_track_points, rng):
    "Pick flat pixel indices of the reference view that are tracked across views."
    candidates = np.flatnonzero(valid_mask.reshape(-1))
    if candidates.size > max_track_points:
        candidates = rng.choice(candidates, size=max_track_points, replace=False)
        candidates.sort()
    return candidates


def log_tracking(
    data,
    valid_masks,
    max_track_points=500,
    motion_threshold=0.05,
    show_all_frames=False,
    show_predicted_depth=False,
    show_tracks=False,
    show_camera_path=False,
):
    "Log cameras, images, point clouds, scene flow and point tracks to Rerun."
    images = [(image.clip(0, 1) * 255).astype(np.uint8) for image in data["images"]]
    pts3d = data["pts3d"]
    num_views = len(images)

    # Fixed depth color range over the whole sequence so colors are comparable.
    depth_vmin, depth_vmax = depth_color_range(data["input_depths"])

    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    rng = np.random.default_rng(0)
    ref_points = pts3d[0].reshape(-1, 3)
    track_indices = select_track_indices(valid_masks[0] > 0, max_track_points, rng)
    track_colors = cmap_colors(ref_points[track_indices, 0], "rainbow")
    track_history = [ref_points[track_indices]]
    camera_positions = []

    for view_idx in range(num_views):
        rr.set_time("view", sequence=view_idx)

        height, width = images[view_idx].shape[:2]
        cam_rot = quaternion_to_rotation_matrix(torch.from_numpy(data["cam_quats"][view_idx])).numpy()

        # `world/points`, `world/scene_flow` and `world/point_tracks` are already
        # in world coordinates, so they are logged outside the camera transform.
        rr.log("world/camera", rr.Transform3D(translation=data["cam_trans"][view_idx], mat3x3=cam_rot))
        rr.log(
            "world/camera/pinhole",
            rr.Pinhole(
                image_from_camera=data["intrinsics"][view_idx],
                height=height,
                width=width,
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
        )
        rr.log("world/camera/pinhole/rgb", rr.Image(images[view_idx]))
        rr.log(
            "world/camera/pinhole/depth_input",
            rr.Image(colorize_depth(data["input_depths"][view_idx], depth_vmin, depth_vmax)),
        )
        if show_predicted_depth:
            rr.log(
                "world/camera/pinhole/depth_pred",
                rr.Image(colorize_depth(data["depth_z"][view_idx], depth_vmin, depth_vmax)),
            )
        camera_positions.append(data["cam_trans"][view_idx])

        valid = valid_masks[view_idx] > 0
        rr.log(
            "world/points",
            rr.Points3D(positions=pts3d[view_idx][valid], colors=images[view_idx][valid]),
        )
        if show_all_frames:
            rr.log(
                f"world/all_frames/view_{view_idx:03d}/points",
                rr.Points3D(positions=pts3d[view_idx][valid], colors=images[view_idx][valid]),
                static=True,
            )

        if view_idx == 0:
            continue

        # World-frame scene flow of the tracked reference points.
        scene_flow = data["scene_flows"][view_idx].reshape(-1, 3)[track_indices]
        if show_tracks:
            track_history.append(ref_points[track_indices] + scene_flow)

        magnitude = np.linalg.norm(scene_flow, axis=-1)
        moving = magnitude > motion_threshold
        if moving.any():
            rr.log(
                "world/scene_flow",
                rr.Arrows3D(
                    origins=ref_points[track_indices][moving],
                    vectors=scene_flow[moving],
                    colors=cmap_colors(magnitude[moving], "turbo"),
                ),
            )
            if show_all_frames:
                rr.log(
                    f"world/all_frames/view_{view_idx:03d}/scene_flow",
                    rr.Arrows3D(
                        origins=ref_points[track_indices][moving],
                        vectors=scene_flow[moving],
                        colors=cmap_colors(magnitude[moving], "turbo"),
                    ),
                    static=True,
                )
        else:
            # Clear the previous frame's arrows instead of leaving them behind.
            rr.log("world/scene_flow", rr.Clear(recursive=False))

        # Trajectories of the tracked points as growing line strips.
        if show_tracks:
            history = np.stack(track_history, axis=0)  # (T, N, 3)
            strips = [history[:, track] for track in range(history.shape[1])]
            rr.log("world/point_tracks", rr.LineStrips3D(strips=strips, colors=track_colors))

    if show_camera_path:
        rr.log(
            "world/camera_path",
            rr.LineStrips3D(strips=[np.stack(camera_positions, axis=0)]),
            static=True,
        )


def main():
    args = get_parser().parse_args()
    seed_everything(args.seed)

    rgb_dir, depth_dir, camera_params = resolve_inputs(args)

    blueprint = build_blueprint(args.show_predicted_depth)
    rr.script_setup(args, "any4d_rgbd_tracking", default_blueprint=blueprint)
    if not (args.save or args.connect or args.serve or args.stdout or not args.headless):
        print("Rerun logging is disabled: pass --save, --connect, --serve, --stdout or --headless false.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = find_pairs(rgb_dir, depth_dir)
    if not pairs:
        raise ValueError(
            f"No RGB/depth pairs found between {rgb_dir} and {depth_dir}. "
            "RGB and depth files must share the same stem."
        )
    pairs = pairs[args.start_idx : args.end_idx : args.stride]
    print(f"Selected {len(pairs)} RGB-D pairs")

    target_size = compute_target_size(pairs, args.resolution_set)
    print(f"Using target resolution {target_size[0]}x{target_size[1]} (W x H)")

    intrinsics = load_intrinsics(
        camera_params=camera_params, fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy
    )
    model, data_norm_type = init_inference_model(
        args.config_dir, args.checkpoint_path, task=args.task, machine=args.machine, device=device
    )
    img_norm = build_image_normalizer(data_norm_type)

    views, valid_masks = [], []
    for key, rgb_path, depth_path in pairs:
        view, valid_mask = load_rgbd_view(
            rgb_path,
            depth_path,
            intrinsics,
            target_size,
            depth_scale=args.depth_scale,
            max_depth=args.max_depth,
            img_norm=img_norm,
            data_norm_type=data_norm_type,
        )
        views.append(view)
        valid_masks.append(valid_mask)
        print(f"  loaded {key}")

    views = validate_input_views_for_inference(views)
    input_views = views
    views = preprocess_input_views_for_inference(views)

    with torch.no_grad():
        result = loss_of_one_batch_multi_view(views, model, None, device, use_amp=not args.no_amp)
    raw_outputs = [result[f"pred{i + 1}"] for i in range(len(views))]
    processed = postprocess_model_outputs_for_inference(
        raw_outputs, views, apply_mask=False, mask_edges=False
    )

    log_tracking(
        gather_view_data(processed, input_views),
        valid_masks,
        max_track_points=args.max_track_points,
        motion_threshold=args.motion_threshold,
        show_all_frames=args.show_all_frames,
        show_predicted_depth=args.show_predicted_depth,
        show_tracks=args.show_tracks,
        show_camera_path=args.show_camera_path,
    )
    print(f"Logged {len(views)} views to Rerun")

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
