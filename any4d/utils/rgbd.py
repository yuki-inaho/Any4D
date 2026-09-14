"""
Utilities for loading RGB-D image pairs for Any4D inference.

RGB images and depth maps aligned to the RGB camera are paired by file stem,
resized with the same geometric transform as the images, and converted into
the view dictionaries expected by the Any4D inference API.
"""

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
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def stem_key(path):
    "Pair RGB and depth files by stem, ignoring '_rgb'/'_depth' suffixes and 'color_'/'depth_' prefixes."
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


def resolve_session_paths(session_dir, camera_params=None):
    """
    Resolve rgb/depth/camera-parameter paths from a single session directory.

    Supported layouts:
        - <session>/rgb/*_rgb.png + <session>/mapped_depth/*_depth.png
        - <session>/Color_*.{jpg,png} + <session>/mapped_depth/Depth_*.png
    """
    if not os.path.isdir(session_dir):
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    rgb_dir = os.path.join(session_dir, "rgb")
    if not os.path.isdir(rgb_dir):
        rgb_dir = session_dir

    depth_dir = os.path.join(session_dir, "mapped_depth")
    if not os.path.isdir(depth_dir):
        raise FileNotFoundError(
            f"No 'mapped_depth' directory under {session_dir}. "
            "Pass --rgb_dir/--depth_dir explicitly for other layouts."
        )

    if camera_params is None:
        candidate = os.path.join(session_dir, "camera_parameters", "rgb_camera_param.yaml")
        if os.path.exists(candidate):
            camera_params = candidate
    if camera_params is None:
        raise FileNotFoundError(
            f"No camera parameters found under {session_dir}. "
            "Pass --camera_params pointing to a ROS-style camera info file or --fx/--fy/--cx/--cy."
        )
    return rgb_dir, depth_dir, camera_params


def load_intrinsics(camera_params=None, fx=None, fy=None, cx=None, cy=None):
    "Build a 3x3 intrinsics matrix from explicit values or a camera parameter file."
    if fx is not None:
        if None in (fy, cx, cy):
            raise ValueError("--fx requires --fy, --cx and --cy to be provided as well.")
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    if camera_params is None:
        raise ValueError(
            "Provide camera intrinsics through --camera_params or --fx/--fy/--cx/--cy. "
            "Depth conditioning requires camera calibration to convert Z depth to depth along the ray."
        )

    with open(camera_params, encoding="utf-8") as stream:
        if str(camera_params).lower().endswith((".yaml", ".yml")):
            import yaml

            params = yaml.safe_load(stream)
        else:
            params = json.load(stream)

    if "K" in params:
        intrinsics = np.array(params["K"], dtype=np.float32).reshape(3, 3)
    elif all(k in params for k in ("fx", "fy", "cx", "cy")):
        intrinsics = np.array(
            [[params["fx"], 0.0, params["cx"]], [0.0, params["fy"], params["cy"]], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"{camera_params} must contain either 'K' or 'fx'/'fy'/'cx'/'cy' entries.")
    return intrinsics


def compute_target_size(pairs, resolution_set=518):
    "Pick the model input resolution from the average aspect ratio, as in any4d.utils.image.load_images."
    aspect_ratios = []
    for _, rgb_path, _ in pairs:
        with Image.open(rgb_path) as image:
            width, height = exif_transpose(image).size
        aspect_ratios.append(width / height)
    average_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
    return find_closest_aspect_ratio(average_aspect_ratio, resolution_set)


def build_image_normalizer(data_norm_type):
    "Build the torchvision normalization transform for an encoder data_norm_type."
    if data_norm_type not in IMAGE_NORMALIZATION_DICT:
        raise ValueError(
            f"Unknown image normalization type: {data_norm_type}. "
            f"Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
        )
    image_norm = IMAGE_NORMALIZATION_DICT[data_norm_type]
    return tvf.Compose(
        [tvf.ToTensor(), tvf.Normalize(mean=image_norm.mean, std=image_norm.std)]
    )


def load_rgbd_view(
    rgb_path,
    depth_path,
    intrinsics,
    target_size,
    depth_scale=0.001,
    max_depth=None,
    img_norm=None,
    data_norm_type="dinov2",
):
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
        "data_norm_type": [data_norm_type],
        "true_shape": np.int32([image.size[::-1]]),
    }
    return view, valid_mask


def init_hydra_config(config_dir, config_name="train", overrides=None):
    "Compose a Hydra config from an absolute config directory."
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(version_base=None, config_dir=os.path.abspath(config_dir))
    return hydra.compose(config_name=config_name, overrides=overrides or [])


def init_inference_model(config_dir, checkpoint_path=None, task="rgbd", machine="local", device="cpu"):
    "Build the Any4D model from the config and load the official checkpoint."
    model_args = init_hydra_config(
        config_dir,
        "train",
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
