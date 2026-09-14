# --------------------------------------------------------
# Multi-view RGB-D inference script for Any4D
#
# Conditions the official Any4D checkpoint on RGB images plus metric depth
# maps aligned to the RGB camera (for example "mapped depth" PNGs). Unlike
# scripts/demo_inference.py this script does not use MoGe: the depth validity
# mask is used in place of the MoGe non-ambiguous mask, and the depth itself
# can be fed to the model as a geometric input.
# --------------------------------------------------------
import argparse
import json
import os

import numpy as np
import torch

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


def get_parser():
    parser = argparse.ArgumentParser(description="Multi-view RGB-D inference with Any4D.")
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Session root directory. Resolves <session>/rgb (or Color_*.jpg), "
        "<session>/mapped_depth and <session>/camera_parameters/rgb_camera_param.yaml automatically.",
    )
    parser.add_argument(
        "--rgb_dir", type=str, default=None, help="RGB directory (alternative to --session)."
    )
    parser.add_argument(
        "--depth_dir", type=str, default=None, help="Depth directory (alternative to --session)."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/any4d_4v_combined.pth",
        help="Path to the official Any4D pretrained checkpoint.",
    )
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument(
        "--task",
        type=str,
        default="rgbd",
        help="Hydra task config under configs/model/task. Use 'rgbd' to condition on depth and ray "
        "directions, or 'images_only' for RGB-only inference.",
    )
    parser.add_argument("--machine", type=str, default="local")

    # Camera / depth conventions
    parser.add_argument(
        "--camera_params",
        type=str,
        default=None,
        help="Camera parameter file (YAML/JSON) with ROS-style 'K', 'width' and 'height' fields.",
    )
    parser.add_argument("--fx", type=float, default=None, help="Focal length x (overrides --camera_params).")
    parser.add_argument("--fy", type=float, default=None, help="Focal length y (overrides --camera_params).")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x (overrides --camera_params).")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y (overrides --camera_params).")
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=0.001,
        help="Scale factor that converts stored depth values to meters (default: 0.001 for mm uint16 PNGs).",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=None,
        help="Optional maximum valid depth in meters. Larger values (and 0) are treated as invalid.",
    )

    # Data selection
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None, help="Exclusive end index (default: all).")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help="Number of views per forward pass (0 = all selected views in a single pass).",
    )
    parser.add_argument("--resolution_set", type=int, default=518, choices=[518, 512])

    # Runtime
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save predictions.")
    parser.add_argument("--save_ply", action="store_true", help="Also export a combined point cloud as PLY.")
    parser.add_argument("--no_amp", action="store_true", help="Disable bfloat16 autocast.")
    parser.add_argument("--seed", type=int, default=0)

    return parser


@torch.no_grad()
def run_inference(model, views, device, use_amp):
    return loss_of_one_batch_multi_view(views, model, None, device, use_amp=use_amp)


def depth_agreement(predicted_depth_z, input_depth, valid_mask):
    "Median/mean relative error between predicted and input Z depth on valid pixels."
    predicted_depth_z = (
        np.squeeze(predicted_depth_z, axis=-1)
        if predicted_depth_z.ndim == 3 and predicted_depth_z.shape[-1] == 1
        else predicted_depth_z
    )
    valid = (valid_mask > 0) & (input_depth > 0)
    if valid.sum() == 0:
        return None
    predicted = predicted_depth_z[valid]
    target = input_depth[valid]
    relative_error = np.abs(predicted - target) / target
    return float(np.median(relative_error)), float(np.mean(relative_error))


def save_ply(path, point_clouds):
    "Export a list of (points, colors) tuples as a single PLY file."
    import trimesh

    meshes = []
    for points, colors in point_clouds:
        if points.shape[0] == 0:
            continue
        meshes.append(trimesh.PointCloud(vertices=points, colors=colors))
    if not meshes:
        print("No valid points to export, skipping PLY.")
        return
    trimesh.Scene(meshes).export(path)
    print(f"Saved point cloud to {path}")


def save_predictions(output_dir, keys, processed, views, valid_masks):
    "Save per-view predictions as compressed npz plus a JSON summary."
    os.makedirs(output_dir, exist_ok=True)
    summary = []

    for key, prediction, view, valid_mask in zip(keys, processed, views, valid_masks):
        arrays = {}
        for name in (
            "pts3d",
            "pts3d_cam",
            "depth_z",
            "depth_along_ray",
            "ray_directions",
            "cam_quats",
            "cam_trans",
            "camera_poses",
            "metric_scaling_factor",
        ):
            if name in prediction:
                arrays[name] = prediction[name][0].detach().float().cpu().numpy()

        input_depth = view["depth_z"][0, ..., 0].detach().float().cpu().numpy()
        arrays["input_depth_m"] = input_depth
        arrays["valid_mask"] = valid_mask
        arrays["intrinsics"] = view["intrinsics"][0].detach().float().cpu().numpy()
        if "img_no_norm" in prediction:
            image = prediction["img_no_norm"][0].detach().float().cpu().numpy()
            arrays["image"] = (image.clip(0, 1) * 255).astype(np.uint8)

        np.savez_compressed(os.path.join(output_dir, f"{key}.npz"), **arrays)

        agreement = depth_agreement(arrays["depth_z"], input_depth, valid_mask)
        entry = {
            "key": key,
            "depth_agreement_median_relative_error": agreement[0] if agreement else None,
            "depth_agreement_mean_relative_error": agreement[1] if agreement else None,
            "valid_fraction": float((valid_mask > 0).mean()),
        }
        if "metric_scaling_factor" in arrays:
            entry["metric_scaling_factor"] = float(np.asarray(arrays["metric_scaling_factor"]).reshape(-1)[0])
        summary.append(entry)
        print(
            f"  {key}: valid={entry['valid_fraction']:.3f}"
            + (
                f", depth rel. err (median/mean)={agreement[0]:.4f}/{agreement[1]:.4f}"
                if agreement
                else ", no valid depth pixels"
            )
        )

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(f"Saved predictions for {len(summary)} views to {output_dir}")


def main():
    args = get_parser().parse_args()
    seed_everything(args.seed)

    rgb_dir, depth_dir, camera_params = args.rgb_dir, args.depth_dir, args.camera_params
    if args.session is not None:
        rgb_dir, depth_dir, camera_params = resolve_session_paths(args.session, camera_params)
    if rgb_dir is None or depth_dir is None:
        raise ValueError("Provide --session or both --rgb_dir and --depth_dir.")

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
    if intrinsics[0, 2] >= target_size[0] * 2 or intrinsics[1, 2] >= target_size[1] * 2:
        raise ValueError(
            f"Principal point {intrinsics[:2, 2]} looks inconsistent with the input images. "
            "Make sure the intrinsics correspond to the original RGB resolution."
        )

    model, data_norm_type = init_inference_model(
        args.config_dir, args.checkpoint_path, task=args.task, machine=args.machine, device=device
    )
    img_norm = build_image_normalizer(data_norm_type)

    chunk_size = args.chunk_size if args.chunk_size > 0 else len(pairs)
    use_amp = not args.no_amp

    for chunk_start in range(0, len(pairs), chunk_size):
        chunk = pairs[chunk_start : chunk_start + chunk_size]
        views, valid_masks, keys = [], [], []
        for key, rgb_path, depth_path in chunk:
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
            keys.append(key)

        views = validate_input_views_for_inference(views)
        input_views = views
        views = preprocess_input_views_for_inference(views)

        result = run_inference(model, views, device, use_amp)
        raw_outputs = [result[f"pred{i + 1}"] for i in range(len(views))]
        processed = postprocess_model_outputs_for_inference(
            raw_outputs, views, apply_mask=False, mask_edges=False
        )

        if args.output_dir is not None:
            save_predictions(args.output_dir, keys, processed, input_views, valid_masks)

            if args.save_ply:
                point_clouds = []
                for prediction, valid_mask in zip(processed, valid_masks):
                    points = prediction["pts3d"][0].detach().float().cpu().numpy()
                    colors = prediction["img_no_norm"][0].detach().float().cpu().numpy()
                    colors = (colors.clip(0, 1) * 255).astype(np.uint8)
                    mask = valid_mask > 0
                    point_clouds.append((points[mask], colors[mask]))
                save_ply(os.path.join(args.output_dir, "pointcloud.ply"), point_clouds)


if __name__ == "__main__":
    main()
