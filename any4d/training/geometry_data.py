"""RGB-D geometry dataset for Any4D head-only fine-tuning.

Implements the input/target contract of the geometry handoff workdoc:
fixed 4-frame sequences at the native 480x640 resolution, padded to
490x644 for the patch-14 encoder, with the depth conditioning input
partially held out so that the depth and pose losses are not leaked
through the input.

``GeometryDataset`` returns per-sample ``(model_views, targets)`` where
``model_views`` is a list of four unbatched view dictionaries and
``targets`` is a dictionary of metric geometry teachers. Use
:func:`collate_geometry` to build a batch of a fixed size.
"""

import hashlib
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from any4d.training.flow_teacher import (
    POSE_CONVENTION,
    RAFT_WEIGHTS_SHA256,
    TEACHER_NAME,
)
from any4d.utils.geometry import get_rays_in_camera_frame
from any4d.utils.rgbd import build_image_normalizer

BASE_H = 480
BASE_W = 640
PAD_BOTTOM = 10
PAD_RIGHT = 4
PADDED_H = BASE_H + PAD_BOTTOM
PADDED_W = BASE_W + PAD_RIGHT
DEPTH_MM_TO_M = 0.001
MAX_DEPTH_MM = 1300
HOLDOUT_BLOCK = 16
HOLDOUT_PROB = 0.2
NUM_VIEWS = 4
SPLIT_IDS = {"train": 0, "val": 1, "smoke": 2}
DATA_NORM_TYPE = "dinov2"
SCENE_NAME = "scene_000000"
FLOW_SCHEMA = "any4d_pseudo_sf_v1"
FLOW_TARGET_VIEWS = (1, 2, 3)


def _read_split_rows(data_root, split):
    path = os.path.join(data_root, "splits", f"{split}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing split file: {path}")
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            scene, sequence = line.split("/")
            if scene != SCENE_NAME:
                raise ValueError(
                    f"Unexpected scene '{scene}' in {path}; only {SCENE_NAME} is staged."
                )
            rows.append(int(sequence.rsplit("_", 1)[1]))
    if not rows:
        raise ValueError(f"Split file is empty: {path}")
    return rows


class GeometryDataset(torch.utils.data.Dataset):
    """Fixed 4-view geometry dataset over one staged scene."""

    def __init__(self, data_root, split, seed=42, epoch=0, flow_root=None, dataset_digest=None):
        if split not in SPLIT_IDS:
            raise ValueError(f"split must be one of {sorted(SPLIT_IDS)}, got {split!r}")
        self.data_root = data_root
        self.split = split
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.flow_root = flow_root
        self.dataset_digest = dataset_digest
        self.scene_dir = os.path.join(data_root, "scenes", SCENE_NAME)
        if not os.path.isdir(self.scene_dir):
            raise FileNotFoundError(f"Missing scene directory: {self.scene_dir}")

        cameras = np.load(
            os.path.join(self.scene_dir, "cameras.npz"), allow_pickle=False
        )
        sequences = np.load(
            os.path.join(self.scene_dir, "sequences.npz"), allow_pickle=False
        )
        self.frame_ids = cameras["frame_ids"]
        self.intrinsics = cameras["intrinsics"]
        self.w2c = cameras["extrinsics_w2c"]
        self.quality = cameras["quality_flags"]
        self.frame_chunks = cameras["chunk_ids"]
        self.frame_row = {int(fid): i for i, fid in enumerate(self.frame_ids)}
        frame_order = self.frame_row

        seq = sequences["sequences"]
        lengths = sequences["lengths"]
        split_ids = sequences["split_ids"]
        chunk_ids = sequences["chunk_ids"]

        self.sequence_records = []
        self.filtered = {"length": 0, "quality": 0}
        for row in _read_split_rows(data_root, split):
            if not 0 <= row < len(seq):
                raise ValueError(f"sequence row {row} out of range for split {split}")
            if int(split_ids[row]) != SPLIT_IDS[split]:
                raise ValueError(
                    f"split_ids[{row}]={int(split_ids[row])} does not match {split}"
                )
            if int(lengths[row]) != NUM_VIEWS:
                self.filtered["length"] += 1
                continue
            frame_ids = [int(fid) for fid in seq[row]]
            if any(fid < 0 for fid in frame_ids):
                raise ValueError(f"sequence {row} is padded but has length 4")
            frame_rows = []
            for fid in frame_ids:
                if fid not in frame_order:
                    raise ValueError(f"sequence {row} references unknown frame {fid}")
                frame_rows.append(frame_order[fid])
            if not all(bool(self.quality[fr]) for fr in frame_rows):
                self.filtered["quality"] += 1
                continue
            for fr in frame_rows:
                if int(self.frame_chunks[fr]) != int(chunk_ids[row]):
                    raise ValueError(
                        f"frame {int(self.frame_ids[fr])} chunk {int(self.frame_chunks[fr])} "
                        f"differs from sequence {row} chunk {int(chunk_ids[row])}"
                    )
            self.sequence_records.append(
                {
                    "row": row,
                    "frame_ids": frame_ids,
                    "frame_rows": frame_rows,
                    "chunk_id": int(chunk_ids[row]),
                }
            )
        if not self.sequence_records:
            raise ValueError(f"No valid 4-frame sequences for split {split}")

        self.flow_filtered = {"all_empty": 0, "missing_pairs": 0}
        self.flow_stats = None
        if flow_root is not None:
            self._filter_by_flow_cache(flow_root)

        self.img_norm = build_image_normalizer(DATA_NORM_TYPE)

    def __len__(self):
        return len(self.sequence_records)

    def _flow_pair_path(self, sequence_row, target_view):
        return os.path.join(
            self.flow_root,
            self.split,
            f"sequence_{sequence_row:06d}",
            f"view_0_to_{target_view}.npz",
        )

    def _validate_flow_manifest(self, flow_root):
        manifest_path = os.path.join(flow_root, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"flow cache manifest not found: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("schema") != FLOW_SCHEMA:
            raise ValueError(
                f"flow cache schema mismatch: {manifest.get('schema')!r} != {FLOW_SCHEMA}"
            )
        if not manifest.get("completed"):
            raise ValueError(f"flow cache is not completed: {manifest_path}")
        if manifest.get("pose_convention") != POSE_CONVENTION:
            raise ValueError(
                "flow cache pose convention mismatch: "
                f"{manifest.get('pose_convention')!r} != {POSE_CONVENTION!r}"
            )
        teacher = manifest.get("teacher", {})
        if teacher.get("name") != TEACHER_NAME:
            raise ValueError(
                f"flow cache teacher mismatch: {teacher.get('name')!r} != {TEACHER_NAME!r}"
            )
        if teacher.get("sha256") != RAFT_WEIGHTS_SHA256:
            raise ValueError(
                "flow cache teacher weight hash mismatch: "
                f"{teacher.get('sha256')!r} != {RAFT_WEIGHTS_SHA256!r}"
            )
        expected_digest = self.dataset_digest
        if expected_digest is None:
            expected_digest = compute_dataset_digest(self.data_root)
        if manifest.get("dataset_digest") != expected_digest:
            raise ValueError(
                "flow cache was generated for a different dataset digest: "
                f"{manifest.get('dataset_digest')!r} != {expected_digest!r}"
            )
        return manifest

    def _filter_by_flow_cache(self, flow_root):
        self.manifest = self._validate_flow_manifest(flow_root)
        kept = []
        total_pairs = 0
        empty_pairs = 0
        dynamic_points = 0
        usable_points = 0
        for record in self.sequence_records:
            pair_stats = []
            record_dynamic = 0
            record_usable = 0
            for target_view in FLOW_TARGET_VIEWS:
                path = self._flow_pair_path(record["row"], target_view)
                if not os.path.isfile(path):
                    self.flow_filtered["missing_pairs"] += 1
                    raise FileNotFoundError(
                        f"flow cache is incomplete, missing pair: {path}"
                    )
                with np.load(path, allow_pickle=False) as data:
                    self._validate_flow_pair(data, record, target_view, path)
                    count = len(np.asarray(data["source_yx"]))
                    labels = np.asarray(data["motion_label"])
                    dynamic = int((labels == 1).sum())
                    usable = int((labels != -1).sum())
                total_pairs += 1
                empty_pairs += int(count == 0)
                dynamic_points += dynamic
                usable_points += usable
                record_dynamic += dynamic
                record_usable += usable
                pair_stats.append(count)
            record = dict(record)
            record["flow_dynamic_points"] = record_dynamic
            record["flow_usable_points"] = record_usable
            if all(count == 0 for count in pair_stats) or record_usable == 0:
                self.flow_filtered["all_empty"] += 1
                continue
            kept.append(record)
        if not kept:
            raise ValueError(
                f"no sequence with usable flow targets for split {self.split}"
            )
        if self.split == "smoke":
            kept.sort(key=lambda item: (-item["flow_dynamic_points"], item["row"]))
        self.sequence_records = kept
        self.flow_stats = {
            "sequences": len(kept),
            "excluded_all_empty": self.flow_filtered["all_empty"],
            "pairs": total_pairs,
            "empty_pairs": empty_pairs,
            "dynamic_points": dynamic_points,
            "usable_points": usable_points,
        }

    def _validate_flow_pair(self, data, record, target_view, path):
        required = (
            "source_yx",
            "flow_m",
            "motion_label",
            "fb_error_px",
            "frame_ids",
            "target_view",
        )
        for key in required:
            if key not in data:
                raise ValueError(f"{path}: missing key {key}")
        frame_ids = np.asarray(data["frame_ids"])
        if (
            frame_ids.shape != (NUM_VIEWS,)
            or [int(value) for value in frame_ids] != list(record["frame_ids"])
        ):
            raise ValueError(f"{path}: frame_ids do not match the sequence")
        if int(np.asarray(data["target_view"])) != target_view:
            raise ValueError(f"{path}: target_view mismatch")
        source = np.asarray(data["source_yx"])
        flow = np.asarray(data["flow_m"])
        labels = np.asarray(data["motion_label"])
        fb_error = np.asarray(data["fb_error_px"])
        count = len(source)
        if source.dtype.kind not in "iu":
            raise ValueError(f"{path}: source_yx must be an integer array")
        if source.shape != (count, 2):
            raise ValueError(f"{path}: inconsistent sparse tensor shapes")
        if len(np.unique(source, axis=0)) != count:
            raise ValueError(f"{path}: duplicate source coordinates")
        if (
            flow.shape != (count, 3)
            or labels.shape != (count,)
            or fb_error.shape != (count,)
        ):
            raise ValueError(f"{path}: inconsistent sparse tensor shapes")
        if not np.isfinite(flow).all() or not np.isfinite(fb_error).all():
            raise ValueError(f"{path}: non-finite flow values")
        if count and (
            (source[:, 0] < 0).any()
            or (source[:, 0] >= BASE_H).any()
            or (source[:, 1] < 0).any()
            or (source[:, 1] >= BASE_W).any()
        ):
            raise ValueError(f"{path}: source_yx out of bounds")
        if not np.isin(labels, (-1, 0, 1)).all():
            raise ValueError(f"{path}: invalid motion labels")

    def _load_flow_pair(self, record, target_view):
        path = self._flow_pair_path(record["row"], target_view)
        with np.load(path, allow_pickle=False) as data:
            self._validate_flow_pair(data, record, target_view, path)
            pair = {
                "source_yx": torch.from_numpy(
                    np.asarray(data["source_yx"]).astype(np.int64)
                ),
                "flow_m": torch.from_numpy(
                    np.asarray(data["flow_m"]).astype(np.float32)
                ),
                "motion_label": torch.from_numpy(
                    np.asarray(data["motion_label"]).astype(np.int8)
                ),
                "fb_error_px": torch.from_numpy(
                    np.asarray(data["fb_error_px"]).astype(np.float32)
                ),
                "frame_ids": torch.from_numpy(
                    np.asarray(data["frame_ids"]).astype(np.int64)
                ),
                "target_view": torch.tensor(target_view),
            }
        return pair

    def _holdout_pattern(self, sequence_row, view_index):
        text = f"{self.seed}/{self.epoch}/{SCENE_NAME}/{sequence_row}/{view_index}"
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little") & (2**63 - 1)
        generator = torch.Generator()
        generator.manual_seed(seed)
        blocks = (
            torch.rand(
                (BASE_H // HOLDOUT_BLOCK, BASE_W // HOLDOUT_BLOCK),
                generator=generator,
            )
            < HOLDOUT_PROB
        )
        pattern = blocks.repeat_interleave(HOLDOUT_BLOCK, dim=0)
        pattern = pattern.repeat_interleave(HOLDOUT_BLOCK, dim=1)
        return pattern.numpy()

    def _load_arrays(self, frame_id):
        image_path = os.path.join(self.scene_dir, "rgb", f"frame_{frame_id:06d}.png")
        depth_path = os.path.join(self.scene_dir, "depth", f"frame_{frame_id:06d}.png")
        image = Image.open(image_path).convert("RGB")
        if image.size != (BASE_W, BASE_H):
            raise ValueError(f"{image_path}: expected 640x480, got {image.size}")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise OSError(f"Could not read depth map: {depth_path}")
        if depth.shape != (BASE_H, BASE_W) or depth.dtype != np.uint16:
            raise ValueError(
                f"{depth_path}: expected uint16 480x640, got {depth.dtype} {depth.shape}"
            )
        return image, depth

    def __getitem__(self, index):
        record = self.sequence_records[index]
        views = []
        depth_targets = []
        valid_targets = []
        holdout_targets = []
        ray_targets = []
        c2w_targets = []

        w2c_reference = np.eye(4, dtype=np.float64)
        w2c_reference[:3, :4] = self.w2c[record["frame_rows"][0]]
        t0_inv = w2c_reference  # E_0; T_i^rel = E_0 @ inv(E_i)

        for view_index, (frame_id, frame_row) in enumerate(
            zip(record["frame_ids"], record["frame_rows"])
        ):
            image, depth_mm = self._load_arrays(frame_id)
            depth_m = depth_mm.astype(np.float32) * DEPTH_MM_TO_M
            valid = (depth_mm > 0) & (depth_mm <= MAX_DEPTH_MM)
            held_blocks = self._holdout_pattern(record["row"], view_index)
            holdout = valid & held_blocks

            depth_input = depth_m.copy()
            depth_input[held_blocks] = 0.0

            img = self.img_norm(image)
            img = F.pad(img, (0, PAD_RIGHT, 0, PAD_BOTTOM), mode="replicate")
            depth_padded = F.pad(
                torch.from_numpy(depth_input)[None],
                (0, PAD_RIGHT, 0, PAD_BOTTOM),
                mode="constant",
                value=0.0,
            )[0, ..., None]

            intrinsics = torch.from_numpy(
                np.asarray(self.intrinsics[frame_row], dtype=np.float32)
            )
            _, rays = get_rays_in_camera_frame(
                intrinsics, BASE_H, BASE_W, normalize_to_unit_sphere=True
            )

            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :4] = self.w2c[frame_row]
            c2w_ref = t0_inv @ np.linalg.inv(w2c)

            views.append(
                {
                    "img": img,
                    "depth_z": depth_padded,
                    "intrinsics": intrinsics,
                    "is_metric_scale": torch.tensor(True),
                    "data_norm_type": [DATA_NORM_TYPE],
                }
            )
            depth_targets.append(torch.from_numpy(depth_m))
            valid_targets.append(torch.from_numpy(valid))
            holdout_targets.append(torch.from_numpy(holdout))
            ray_targets.append(rays)
            c2w_targets.append(torch.from_numpy(c2w_ref.astype(np.float32)))

        targets = {
            "depth_z_m": torch.stack(depth_targets)[..., None],
            "valid_mask": torch.stack(valid_targets),
            "holdout_mask": torch.stack(holdout_targets),
            "rays": torch.stack(ray_targets),
            "c2w_ref": torch.stack(c2w_targets),
            "frame_ids": torch.tensor(record["frame_ids"], dtype=torch.int64),
            "sequence_id": torch.tensor(record["row"], dtype=torch.int64),
            "chunk_id": torch.tensor(record["chunk_id"], dtype=torch.int64),
        }
        if self.flow_root is not None:
            targets["flow_pairs"] = [
                self._load_flow_pair(record, target_view)
                for target_view in FLOW_TARGET_VIEWS
            ]
        return views, targets

    def manifest_digest(self):
        digest = hashlib.sha256()
        digest.update(f"split={self.split}\n".encode())
        for record in self.sequence_records:
            digest.update(
                ("/".join(str(fid) for fid in record["frame_ids"]) + "\n").encode("utf-8")
            )
        return digest.hexdigest()


def compute_dataset_digest(data_root):
    """Digest of the staged dataset with relative POSIX paths only.

    Follows the workdoc contract: sort every regular file by its path
    relative to ``data_root`` and hash the concatenation of
    ``relative_path + NUL + SHA256(file_bytes) + LF``.
    """
    root = os.path.abspath(data_root)
    entries = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            path = os.path.join(directory, filename)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            with open(path, "rb") as stream:
                file_digest = hashlib.sha256(stream.read()).hexdigest()
            entries.append((relative, file_digest))
    entries.sort()
    digest = hashlib.sha256()
    for relative, file_digest in entries:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def collate_geometry(samples):
    """Stack per-sample ``(views, targets)`` pairs into a batch."""
    if not samples:
        raise ValueError("collate_geometry received an empty sample list")
    views = []
    for view_index in range(NUM_VIEWS):
        view = {}
        for key in ("img", "depth_z", "intrinsics", "is_metric_scale"):
            view[key] = torch.stack([sample[0][view_index][key] for sample in samples])
        view["data_norm_type"] = [
            sample[0][view_index]["data_norm_type"][0] for sample in samples
        ]
        views.append(view)
    targets = {}
    for key in samples[0][1]:
        if key == "flow_pairs":
            targets[key] = [sample[1][key] for sample in samples]
        else:
            targets[key] = torch.stack([sample[1][key] for sample in samples])
    return views, targets
