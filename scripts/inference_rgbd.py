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
import gc
import json
import os
from glob import glob
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image
from PIL.ImageOps import exif_transpose

from any4d.models import init_model
from any4d.utils.cropping import crop_resize_if_necessary
from any4d.utils.image import find_closest_aspect_ratio
from any4d.utils.inference import (
    loss_of_one_batch_multi_view,
    postprocess_model_outputs_for_inference,
    preprocess_input_views_for_inference,
    validate_input_views_for_inference,
)
from any4d.utils.misc import seed_everything
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def get_parser():
    parser = argparse.ArgumentParser(description="Multi-view RGB-D inference with Any4D.")
    parser.add_argument("--rgb_dir", type=str, required=True, help="Directory containing RGB images.")
    parser.add_argument(
        "--depth_dir",
        type=str,
        required=True,
        help="Directory containing depth maps aligned to the RGB camera. Files are paired with the RGB "
        "images by matching file stems (a trailing '_rgb'/'_depth' suffix is ignored).",
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


def init_hydra_config(config_path, overrides=None):
    "Initialize a Hydra config relative to this script (same convention as scripts/demo_inference.py)."
    config_dir = os.path.dirname(os.path.abspath(config_path))
    config_name = os.path.basename(config_path).split(".")[0]
    relative_path = os.path.relpath(config_dir, os.path.dirname(os.path.abspath(__file__)))
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path=relative_path)
    if overrides is not None:
        return hydra.compose(config_name=config_name, overrides=overrides)
    return hydra.compose(config_name=config_name)


def init_inference_model(config_dir, task, machine, checkpoint_path, device):
    "Build the Any4D model from the config and load the official checkpoint."
    model_args = init_hydra_config(
        os.path.join(config_dir, "train.yaml"),
        overrides=[
            f"machine={machine}",
            "model=any4d",
            "model.encoder.uses_torch_hub=false",
            f"model/task={task}",
        ],
    )
    model = init_model(model_args.model.model_str, model_args.model.model_config)
    model.to(device)

    if checkpoint_path is not None:
        print(f"Loading model from: {checkpoint_path}")
        try:
            # Memory-map the checkpoint when possible to avoid a full extra copy in RAM.
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        except (RuntimeError, TypeError, ValueError):
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(model.load_state_dict(ckpt["model"], strict=False))
        del ckpt
        gc.collect()
    model.to(device)
    model.eval()

    return model, model_args.model.data_norm_type


def stem_key(path):
    "Pair RGB and depth files by their stem, ignoring '_rgb'/'_depth' suffixes and 'color_'/'depth_' prefixes."
    stem = Path(path).stem
    lowered = stem.lower()
    for suffix in ("_rgb", "_depth"):
        if lowered.endswith(suffix):
            stem = stem[: -len(suffix)]
            lowered = stem.lower()
            break
    for prefix in ("color_", "depth_", "rgb_"):
        if lowered.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return stem


def find_pairs(rgb_dir, depth_dir):
    "Return a sorted list of (key, rgb_path, depth_path) tuples."
    rgb_files = sorted(
        p
        for p in glob(os.path.join(rgb_dir, "*"))
        if p.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS) and "depth" not in os.path.basename(p).lower()
    )
    depth_files = sorted(
        p
        for p in glob(os.path.join(depth_dir, "*"))
        if p.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS) and "rgb" not in os.path.basename(p).lower()
    )
    if os.path.abspath(rgb_dir) == os.path.abspath(depth_dir):
        # When RGB and depth live in the same folder, keep only depth-like file names.
        depth_files = [p for p in depth_files if "depth" in os.path.basename(p).lower()]
    depth_by_key = {stem_key(p): p for p in depth_files}

    pairs = []
    for rgb_path in rgb_files:
        key = stem_key(rgb_path)
        if key in depth_by_key:
            pairs.append((key, rgb_path, depth_by_key[key]))
    pairs.sort(key=lambda pair: pair[0])
    return pairs


def load_intrinsics(args):
    "Build a 3x3 intrinsics matrix from CLI values or a camera parameter file."
    if args.fx is not None:
        if None in (args.fy, args.cx, args.cy):
            raise ValueError("--fx requires --fy, --cx and --cy to be provided as well.")
        return np.array(
            [[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]], dtype=np.float32
        )
    if args.camera_params is None:
        raise ValueError(
            "Provide camera intrinsics through --camera_params or --fx/--fy/--cx/--cy. "
            "Depth conditioning requires camera calibration to convert Z depth to depth along the ray."
        )

    with open(args.camera_params, encoding="utf-8") as stream:
        if str(args.camera_params).lower().endswith((".yaml", ".yml")):
            import yaml

            params = yaml.safe_load(stream)
        else:
            params = json.load(stream)

    if "K" in params:
        intrinsics = np.array(params["K"], dtype=np.float32).reshape(3, 3)
    elif all(k in params for k in ("fx", "fy", "cx", "cy")):
        intrinsics = np.array(
            [
                [params["fx"], 0.0, params["cx"]],
                [0.0, params["fy"], params["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(
            f"{args.camera_params} must contain either 'K' or 'fx'/'fy'/'cx'/'cy' entries."
        )
    return intrinsics


def compute_target_size(pairs, resolution_set):
    "Pick the model input resolution from the average aspect ratio, as in any4d.utils.image.load_images."
    aspect_ratios = []
    for _, rgb_path, _ in pairs:
        with Image.open(rgb_path) as image:
            width, height = exif_transpose(image).size
        aspect_ratios.append(width / height)
    average_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
    return find_closest_aspect_ratio(average_aspect_ratio, resolution_set)


def load_rgbd_view(rgb_path, depth_path, intrinsics, target_size, depth_scale, max_depth, img_norm):
    "Load, resize and normalize an RGB-D pair into a model view dictionary."
    image = exif_transpose(Image.open(rgb_path)).convert("RGB")

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise OSError(f"Could not load depth map: {depth_path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32) * depth_scale
    if max_depth is not None:
        depth[depth > max_depth] = 0.0

    if depth.shape[:2] != (image.size[1], image.size[0]):
        depth = cv2.resize(depth, image.size, interpolation=cv2.INTER_NEAREST)

    valid_mask = (depth > 0).astype(np.uint8)
    image, depth, intrinsics, additional = crop_resize_if_necessary(
        image,
        target_size,
        depthmap=depth,
        intrinsics=intrinsics.copy(),
        additional_quantities=[valid_mask],
    )
    valid_mask = additional[0].astype(np.float32)

    view = {
        "img": img_norm(image)[None],
        "depth_z": torch.from_numpy(depth)[None, ..., None].float(),
        "intrinsics": torch.from_numpy(np.asarray(intrinsics, dtype=np.float32))[None],
        "is_metric_scale": torch.ones(1, dtype=torch.bool),
        "data_norm_type": [None],  # replaced by the caller
        "true_shape": np.int32([image.size[::-1]]),
    }
    return view, valid_mask


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = find_pairs(args.rgb_dir, args.depth_dir)
    if not pairs:
        raise ValueError(
            f"No RGB/depth pairs found between {args.rgb_dir} and {args.depth_dir}. "
            "RGB and depth files must share the same stem."
        )
    pairs = pairs[args.start_idx : args.end_idx : args.stride]
    print(f"Selected {len(pairs)} RGB-D pairs")

    target_size = compute_target_size(pairs, args.resolution_set)
    print(f"Using target resolution {target_size[0]}x{target_size[1]} (W x H)")

    intrinsics = load_intrinsics(args)
    if (intrinsics[0, 2] >= target_size[0] * 2 or intrinsics[1, 2] >= target_size[1] * 2):
        raise ValueError(
            f"Principal point {intrinsics[:2, 2]} looks inconsistent with the input images. "
            "Make sure the intrinsics correspond to the original RGB resolution."
        )

    model, data_norm_type = init_inference_model(
        args.config_dir, args.task, args.machine, args.checkpoint_path, device
    )
    img_norm = tvf.Compose(
        [
            tvf.ToTensor(),
            tvf.Normalize(
                mean=IMAGE_NORMALIZATION_DICT[data_norm_type].mean,
                std=IMAGE_NORMALIZATION_DICT[data_norm_type].std,
            ),
        ]
    )

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
                args.depth_scale,
                args.max_depth,
                img_norm,
            )
            view["data_norm_type"] = [data_norm_type]
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
