"""Pseudo scene-flow teacher built from RAFT correspondences and RGB-D poses.

The teacher estimates, for every pixel of the first view, the 3D
displacement of the corresponding surface point in the first camera
frame. It follows section 5.0 of the geometry handoff workdoc:

    X0(p) = Z0(p) K0^-1 [p,1]
    Xi(q) = R_i^rel (Z_i(q) Ki^-1 [q,1]) + t_i^rel
    F_i(p) = Xi(q) - X0(p)

Unobserved, ambiguous or geometrically inconsistent pixels are dropped
instead of being replaced by a zero-flow teacher.
"""

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

FLOW_SCHEMA = "any4d_pseudo_sf_v1"
POSE_CONVENTION = "stored_extrinsics_are_w2c"
TEACHER_NAME = "torchvision_raft_large_C_T_SKHT_V2"
RAFT_WEIGHTS_FILE = "raft_large_C_T_SKHT_V2-ff5fadd5.pth"
RAFT_WEIGHTS_SHA256 = (
    "ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322"
)
FLOW_TARGET_VIEWS = (1, 2, 3)


@dataclass
class FlowTeacherConfig:
    max_depth_m: float = 1.3
    fb_tolerance_px: float = 1.5
    static_threshold_m: float = 0.01
    dynamic_threshold_m: float = 0.05
    depth_edge_abs_m: float = 0.02
    depth_edge_rel: float = 0.03
    num_flow_updates: int = 12


def _as_float32(value):
    return torch.as_tensor(value, dtype=torch.float32)


def _bilinear(field, y0, x0, wy, wx):
    top = field[y0, x0] * (1.0 - wx) + field[y0, x0 + 1] * wx
    bottom = field[y0 + 1, x0] * (1.0 - wx) + field[y0 + 1, x0 + 1] * wx
    return top * (1.0 - wy) + bottom * wy


def estimate_sparse_targets(
    depth0_m,
    depth_i_m,
    intrinsics0,
    intrinsics_i,
    c2w_i,
    flow_0i,
    flow_i0,
    config=None,
):
    """Build the sparse first-view-grid displacement teacher for one pair.

    ``flow_0i`` and ``flow_i0`` are (2,H,W) pixel flows in dx/dy order.
    Returns numpy arrays in the cache contract plus a stats dictionary.
    """
    config = config or FlowTeacherConfig()
    depth0 = _as_float32(depth0_m)
    depth_i = _as_float32(depth_i_m)
    intrinsics0 = _as_float32(intrinsics0)
    intrinsics_i = _as_float32(intrinsics_i)
    c2w_i = _as_float32(c2w_i)
    flow_0i = _as_float32(flow_0i)
    flow_i0 = _as_float32(flow_i0)
    height, width = depth0.shape

    xs, ys = torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        indexing="xy",
    )
    qx = xs + flow_0i[0]
    qy = ys + flow_0i[1]

    valid = torch.isfinite(depth0) & (depth0 > 0) & (depth0 <= config.max_depth_m)
    valid &= torch.isfinite(flow_0i).all(dim=0)
    valid &= (qx >= 0) & (qx < width - 1) & (qy >= 0) & (qy < height - 1)

    x0 = qx.floor().long().clamp(0, width - 2)
    y0 = qy.floor().long().clamp(0, height - 2)
    wx = qx - x0.float()
    wy = qy - y0.float()

    d00 = depth_i[y0, x0]
    d01 = depth_i[y0, x0 + 1]
    d10 = depth_i[y0 + 1, x0]
    d11 = depth_i[y0 + 1, x0 + 1]
    neighbors = torch.stack([d00, d01, d10, d11], dim=0)
    neighbor_valid = (
        torch.isfinite(neighbors).all(dim=0)
        & (neighbors > 0).all(dim=0)
        & (neighbors <= config.max_depth_m).all(dim=0)
    )
    ordered, _ = neighbors.sort(dim=0)
    median_depth = 0.5 * (ordered[1] + ordered[2])
    edge_threshold = torch.clamp(
        config.depth_edge_rel * median_depth, min=config.depth_edge_abs_m
    )
    edge = (ordered[3] - ordered[0]) > edge_threshold
    valid &= neighbor_valid & ~edge

    z_i = _bilinear(depth_i, y0, x0, wy, wx)
    reverse_x = _bilinear(flow_i0[0], y0, x0, wy, wx)
    reverse_y = _bilinear(flow_i0[1], y0, x0, wy, wx)
    fb_error = torch.sqrt((flow_0i[0] + reverse_x) ** 2 + (flow_0i[1] + reverse_y) ** 2)
    valid &= fb_error <= config.fb_tolerance_px

    ray0 = torch.stack(
        [
            (xs - intrinsics0[0, 2]) / intrinsics0[0, 0],
            (ys - intrinsics0[1, 2]) / intrinsics0[1, 1],
            torch.ones_like(xs),
        ],
        dim=-1,
    )
    ray_i = torch.stack(
        [
            (qx - intrinsics_i[0, 2]) / intrinsics_i[0, 0],
            (qy - intrinsics_i[1, 2]) / intrinsics_i[1, 1],
            torch.ones_like(qx),
        ],
        dim=-1,
    )
    points0 = depth0[..., None] * ray0
    points_i = z_i[..., None] * ray_i
    rotation = c2w_i[:3, :3]
    translation = c2w_i[:3, 3]
    points_i_world = points_i @ rotation.transpose(0, 1) + translation
    displacement = points_i_world - points0

    norms = displacement.norm(dim=-1)
    labels = torch.full_like(norms, -1, dtype=torch.int8)
    labels[norms <= config.static_threshold_m] = 0
    labels[norms >= config.dynamic_threshold_m] = 1

    selected = valid.nonzero(as_tuple=False)
    stats = {
        "candidates": int(selected.shape[0]),
        "coverage": float(valid.float().mean()),
        "static": int((labels[valid] == 0).sum()),
        "dynamic": int((labels[valid] == 1).sum()),
        "ambiguous": int((labels[valid] == -1).sum()),
    }
    return {
        "source_yx": selected.numpy().astype(np.uint16),
        "flow_m": displacement[valid].detach().numpy().astype(np.float32),
        "motion_label": labels[valid].detach().numpy().astype(np.int8),
        "fb_error_px": fb_error[valid].detach().numpy().astype(np.float32),
        "stats": stats,
    }


def weights_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_teacher_weights(allow_weight_download):
    path = os.path.join(
        torch.hub.get_dir(), "checkpoints", RAFT_WEIGHTS_FILE
    )
    if not os.path.isfile(path):
        if not allow_weight_download:
            raise FileNotFoundError(
                f"RAFT weights are not cached at {path}; rerun with "
                "--allow-weight-download to fetch them explicitly"
            )
        return None
    digest = weights_sha256(path)
    if digest != RAFT_WEIGHTS_SHA256:
        raise RuntimeError(
            f"cached RAFT weights hash mismatch: {digest} != {RAFT_WEIGHTS_SHA256}"
        )
    return path


def load_raft_teacher(device="cuda"):
    """Load the pinned torchvision RAFT teacher on the given device."""
    from torchvision.models.optical_flow import (
        raft_large,
        Raft_Large_Weights,
    )

    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights, progress=False).to(device).eval()
    transforms = weights.transforms()
    return model, transforms


def compute_flow_pair(model, transforms, image0, image1, device, config):
    """Estimate 0->1 and 1->0 flows for one image pair (dx/dy, pixels)."""
    transformed_a, transformed_b = transforms(image0, image1)
    with torch.no_grad():
        forward = model(
            transformed_a[None].to(device),
            transformed_b[None].to(device),
            num_flow_updates=config.num_flow_updates,
        )[-1]
        backward = model(
            transformed_b[None].to(device),
            transformed_a[None].to(device),
            num_flow_updates=config.num_flow_updates,
        )[-1]
    return forward[0].float().cpu(), backward[0].float().cpu()


def _load_native_pair(data_root, scene_dir, frame_id):
    import cv2
    from PIL import Image

    image_path = os.path.join(scene_dir, "rgb", f"frame_{frame_id:06d}.png")
    depth_path = os.path.join(scene_dir, "depth", f"frame_{frame_id:06d}.png")
    image = Image.open(image_path).convert("RGB")
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise OSError(f"could not read depth: {depth_path}")
    return (
        torch.from_numpy(np.array(image)).permute(2, 0, 1),
        depth.astype(np.float32) * 0.001,
    )


def _atomic_savez(payload, path):
    temporary = f"{path}.tmp.npz"
    np.savez(temporary, **payload)
    os.replace(temporary, path)


def _settings_record(data_root, splits, config, dataset_digest):
    return {
        "schema": FLOW_SCHEMA,
        "teacher": {"name": TEACHER_NAME, "sha256": RAFT_WEIGHTS_SHA256},
        "dataset_digest": dataset_digest,
        "pose_convention": POSE_CONVENTION,
        "splits": list(splits),
        "config": asdict(config),
    }


def _settings_hash(settings):
    return hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _summarize_flow_cache(output_root, splits):
    """Scan the cache directories so resume runs recompute full quality."""
    quality = {}
    for split in splits:
        entry = {
            "pairs": 0,
            "usable_pairs": 0,
            "empty_pairs": 0,
            "points": 0,
            "static": 0,
            "dynamic": 0,
            "ambiguous": 0,
        }
        split_dir = os.path.join(output_root, split)
        if os.path.isdir(split_dir):
            for sequence in sorted(os.listdir(split_dir)):
                sequence_dir = os.path.join(split_dir, sequence)
                if not os.path.isdir(sequence_dir):
                    continue
                for name in sorted(os.listdir(sequence_dir)):
                    if not name.endswith(".npz"):
                        continue
                    with np.load(
                        os.path.join(sequence_dir, name), allow_pickle=False
                    ) as data:
                        labels = np.asarray(data["motion_label"])
                        count = len(np.asarray(data["source_yx"]))
                    entry["pairs"] += 1
                    entry["points"] += count
                    entry["static"] += int((labels == 0).sum())
                    entry["dynamic"] += int((labels == 1).sum())
                    entry["ambiguous"] += int((labels == -1).sum())
                    if count:
                        entry["usable_pairs"] += 1
                    else:
                        entry["empty_pairs"] += 1
        quality[split] = entry
    return quality


def generate_flow_cache(
    data_root,
    output_root,
    splits=("train", "val", "smoke"),
    resume=False,
    max_pairs=None,
    device=None,
    allow_weight_download=False,
    log=print,
):
    """Generate the pseudo scene-flow cache for every fixed-4 candidate."""
    from any4d.training.geometry_data import compute_dataset_digest, GeometryDataset

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset_digest = compute_dataset_digest(data_root)
    config = FlowTeacherConfig()
    settings = _settings_record(data_root, splits, config, dataset_digest)
    settings_hash = _settings_hash(settings)
    manifest_path = os.path.join(output_root, "manifest.json")
    total_pairs = 0
    for split in splits:
        total_pairs += len(GeometryDataset(data_root, split).sequence_records) * len(
            FLOW_TARGET_VIEWS
        )

    if os.path.exists(manifest_path):
        if not resume:
            raise FileExistsError(
                f"flow cache already exists at {output_root}; pass --resume"
            )
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("settings_hash") != settings_hash:
            raise ValueError(
                "existing flow cache was generated with different settings; "
                "refuse to reuse it"
            )
    elif resume:
        raise FileNotFoundError(f"no flow cache to resume at {output_root}")
    else:
        os.makedirs(output_root, exist_ok=True)
        initial = {
            **settings,
            "settings_hash": settings_hash,
            "completed": False,
            "generated_pairs": 0,
            "skipped_pairs": 0,
            "total_pairs": total_pairs,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z%z"),
        }
        temporary = f"{manifest_path}.tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(initial, stream, indent=2)
        os.replace(temporary, manifest_path)

    model = None
    transforms = None
    generated = 0
    skipped = 0
    quality = {
        split: {
            "pairs": 0,
            "usable_pairs": 0,
            "empty_pairs": 0,
            "points": 0,
            "static": 0,
            "dynamic": 0,
            "ambiguous": 0,
        }
        for split in splits
    }
    started = time.time()

    for split in splits:
        dataset = GeometryDataset(data_root, split)
        scene_dir = dataset.scene_dir
        for record in dataset.sequence_records:
            for target_view in FLOW_TARGET_VIEWS:
                if max_pairs is not None and generated + skipped >= max_pairs:
                    break
                pair_path = os.path.join(
                    output_root,
                    split,
                    f"sequence_{record['row']:06d}",
                    f"view_0_to_{target_view}.npz",
                )
                if os.path.isfile(pair_path):
                    skipped += 1
                    continue
                if model is None:
                    _ensure_teacher_weights(allow_weight_download)
                    log(f"loading RAFT teacher on {device}")
                    model, transforms = load_raft_teacher(device)
                frame0, frame_i = (
                    record["frame_ids"][0],
                    record["frame_ids"][target_view],
                )
                image0, depth0 = _load_native_pair(data_root, scene_dir, frame0)
                image_i, depth_i = _load_native_pair(data_root, scene_dir, frame_i)
                flow_0i, flow_i0 = compute_flow_pair(
                    model, transforms, image0, image_i, device, config
                )
                w2c0 = _frame_pose(dataset, frame0)[1]
                w2c_i = _frame_pose(dataset, frame_i)[1]
                c2w_rel = relative_c2w_from_stored(w2c0, w2c_i)
                intrinsics0 = np.asarray(dataset.intrinsics[dataset.frame_row[frame0]])
                intrinsics_i = np.asarray(dataset.intrinsics[dataset.frame_row[frame_i]])
                result = estimate_sparse_targets(
                    depth0,
                    depth_i,
                    intrinsics0,
                    intrinsics_i,
                    c2w_rel,
                    flow_0i,
                    flow_i0,
                    config,
                )
                os.makedirs(os.path.dirname(pair_path), exist_ok=True)
                _atomic_savez(
                    {
                        "source_yx": result["source_yx"],
                        "flow_m": result["flow_m"],
                        "motion_label": result["motion_label"],
                        "fb_error_px": result["fb_error_px"],
                        "frame_ids": np.asarray(record["frame_ids"], dtype=np.int64),
                        "target_view": np.asarray(target_view, dtype=np.int64),
                    },
                    pair_path,
                )
                stats = result["stats"]
                entry = quality[split]
                entry["pairs"] += 1
                entry["points"] += stats["candidates"]
                entry["static"] += stats["static"]
                entry["dynamic"] += stats["dynamic"]
                entry["ambiguous"] += stats["ambiguous"]
                if stats["candidates"] == 0:
                    entry["empty_pairs"] += 1
                else:
                    entry["usable_pairs"] += 1
                generated += 1
                if generated % 50 == 0:
                    log(
                        f"{split} generated={generated} skipped={skipped} "
                        f"elapsed={time.time() - started:.0f}s"
                    )
            if max_pairs is not None and generated + skipped >= max_pairs:
                break
        if max_pairs is not None and generated + skipped >= max_pairs:
            break

    manifest = {
        **settings,
        "settings_hash": settings_hash,
        "completed": False,
        "generated_pairs": generated,
        "skipped_pairs": skipped,
        "total_pairs": total_pairs,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z%z"),
    }
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    with open(
        os.path.join(output_root, "flow_quality.json"), "w", encoding="utf-8"
    ) as stream:
        json.dump(_summarize_flow_cache(output_root, splits), stream, indent=2)
    manifest["completed"] = (generated + skipped) >= total_pairs
    manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    temporary = f"{manifest_path}.tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    os.replace(temporary, manifest_path)
    return manifest


def relative_c2w_from_stored(w2c_reference, w2c_target):
    """First-camera-relative C2W from stored world-to-camera matrices.

    The staged ``extrinsics_w2c`` array stores OpenCV world-to-camera
    matrices E_i (X_cam = E X_world). With C_i = inv(E_i) the first-camera
    relative pose is ``T_i^rel = inv(C_0) @ C_i = E_0 @ inv(E_i)``, which
    the empirical RAFT projection check reproduces at sub-pixel error.
    """
    return w2c_reference @ np.linalg.inv(w2c_target)


def _frame_pose(dataset, frame_id):
    row = dataset.frame_row[int(frame_id)]
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :4] = dataset.w2c[row]
    return row, w2c
