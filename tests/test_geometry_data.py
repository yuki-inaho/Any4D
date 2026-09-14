"""Tests for the geometry dataset.

These tests require the local RGB-D handoff dataset. Point
``GEOMETRY_DATA_ROOT`` at the ``colmap_rgbd_640x480_v1`` directory before
running them, e.g.::

    GEOMETRY_DATA_ROOT=/path/to/colmap_rgbd_640x480_v1 \
        python -m unittest tests.test_geometry_data -v
"""

import hashlib
import json
import os
import tempfile
import unittest

import cv2
import numpy as np
import torch

from any4d.training.flow_teacher import estimate_sparse_targets, FlowTeacherConfig
from any4d.training.geometry_data import (
    BASE_H,
    BASE_W,
    collate_geometry,
    compute_dataset_digest,
    DEPTH_MM_TO_M,
    FLOW_SCHEMA,
    GeometryDataset,
    PADDED_H,
    PADDED_W,
)

DATA_ROOT = os.environ.get("GEOMETRY_DATA_ROOT")
EXPECTED_FIXED4 = {"train": 879, "val": 163, "smoke": 51}
SCENE_DIR = "scenes/scene_000000"


def pixel_grid(height, width):
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    return xs, ys


def camera_matrix(fx=60.0, fy=60.0, cx=32.0, cy=24.0):
    return torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32
    )


def project_view0(points, c2w, intrinsics):
    rotation = c2w[:3, :3]
    translation = c2w[:3, 3]
    camera = (points - translation) @ rotation
    uv = camera @ intrinsics.T
    return uv[:, :2] / uv[:, 2:3], camera[:, 2]


def make_static_inverse_fields(height, width, intrinsics, c2w):
    """Exact static-plane fields by inverse mapping target pixels to view 0."""
    xs, ys = pixel_grid(height, width)
    rays = torch.stack(
        [
            (xs - intrinsics[0, 2]) / intrinsics[0, 0],
            (ys - intrinsics[1, 2]) / intrinsics[1, 1],
            torch.ones_like(xs),
        ],
        dim=-1,
    )
    rotation = c2w[:3, :3]
    translation = c2w[:3, 3]
    denominator = (rays @ rotation.transpose(0, 1))[..., 2]
    depth_i = (1.0 - translation[2]) / denominator
    points_world = (depth_i[..., None] * rays) @ rotation.transpose(0, 1) + translation
    uv0 = points_world @ intrinsics.transpose(0, 1)
    uv0 = uv0[..., :2] / uv0[..., 2:3]
    flow_i0 = torch.stack([uv0[..., 0] - xs, uv0[..., 1] - ys], dim=0)
    return depth_i, flow_i0


class FlowTeacherTest(unittest.TestCase):
    H, W = 48, 64

    def setUp(self):
        self.config = FlowTeacherConfig()
        angle = torch.tensor(0.05)
        self.c2w = torch.eye(4)
        self.c2w[:3, :3] = torch.tensor(
            [
                [torch.cos(angle), 0.0, torch.sin(angle)],
                [0.0, 1.0, 0.0],
                [-torch.sin(angle), 0.0, torch.cos(angle)],
            ]
        )
        self.c2w[:3, 3] = torch.tensor([0.06, -0.02, 0.03])
        self.intrinsics = camera_matrix(cx=self.W / 2, cy=self.H / 2)

    def _static_inputs(self):
        depth0 = torch.ones(self.H, self.W)
        depth_i, flow_i0 = make_static_inverse_fields(
            self.H, self.W, self.intrinsics, self.c2w
        )
        xs, ys = pixel_grid(self.H, self.W)
        rays = torch.stack(
            [
                (xs - self.intrinsics[0, 2]) / self.intrinsics[0, 0],
                (ys - self.intrinsics[1, 2]) / self.intrinsics[1, 1],
                torch.ones_like(xs),
            ],
            dim=-1,
        )
        rotation = self.c2w[:3, :3]
        translation = self.c2w[:3, 3]
        camera = (rays - translation) @ rotation
        uv_i = camera @ self.intrinsics.T
        uv_i = uv_i[..., :2] / uv_i[..., 2:3]
        flow_0i = torch.stack([uv_i[..., 0] - xs, uv_i[..., 1] - ys], dim=0)
        return depth0, depth_i, flow_0i, flow_i0

    def test_camera_motion_static_zero_flow(self):
        depth0, depth_i, flow_0i, flow_i0 = self._static_inputs()
        result = estimate_sparse_targets(
            depth0,
            depth_i,
            self.intrinsics,
            self.intrinsics,
            self.c2w,
            flow_0i,
            flow_i0,
            self.config,
        )
        count = result["source_yx"].shape[0]
        self.assertGreater(count, self.H * self.W * 0.5)
        norms = torch.from_numpy(result["flow_m"]).norm(dim=-1)
        labels = torch.from_numpy(result["motion_label"].astype(np.int64))
        self.assertLess(float(norms.mean()), 5e-3)
        self.assertLess(float(norms.max()), 2e-2)
        self.assertGreater(float((labels == 0).float().mean()), 0.9)

    def test_camera_and_object_motion_recovers_displacement(self):
        shift = torch.tensor([4.0, 2.0])
        depth0 = torch.ones(self.H, self.W)
        depth_i = torch.ones(self.H, self.W)
        flow_0i = torch.zeros(2, self.H, self.W)
        flow_i0 = torch.zeros(2, self.H, self.W)
        block = (slice(12, 28), slice(24, 40))
        flow_0i[:, block[0], block[1]] = shift[:, None, None]
        flow_i0[:, 14:30, 28:44] = -shift[:, None, None]
        result = estimate_sparse_targets(
            depth0,
            depth_i,
            self.intrinsics,
            self.intrinsics,
            torch.eye(4),
            flow_0i,
            flow_i0,
            self.config,
        )
        source = torch.from_numpy(result["source_yx"].astype(np.int64))
        flow = torch.from_numpy(result["flow_m"])
        in_block = (
            (source[:, 0] >= 12)
            & (source[:, 0] < 28)
            & (source[:, 1] >= 24)
            & (source[:, 1] < 40)
        )
        self.assertGreater(int(in_block.sum()), 100)
        delta = torch.tensor(
            [shift[0] / self.intrinsics[0, 0], shift[1] / self.intrinsics[1, 1], 0.0]
        )
        dynamic = torch.from_numpy(result["motion_label"].astype(np.int64)) == 1
        self.assertTrue(bool(dynamic[in_block].float().mean() > 0.8))
        error = (flow[in_block] - delta).norm(dim=-1)
        self.assertLess(float(error.mean()), 5e-3)
        background = ~in_block
        self.assertLess(float(flow[background].norm(dim=-1).mean()), 1e-3)

    def test_flow_fb_and_depth_edges(self):
        depth0 = torch.ones(self.H, self.W)
        depth0[:, self.W // 2 :] = 0.7
        depth_i = depth0.clone()
        flow_0i = torch.zeros(2, self.H, self.W)
        flow_i0 = torch.zeros(2, self.H, self.W)
        flow_i0[:, 5:10, 5:10] = 7.0
        flow_0i[0, 0, :] = (self.W + 10) - torch.arange(self.W).float()
        result = estimate_sparse_targets(
            depth0,
            depth_i,
            self.intrinsics,
            self.intrinsics,
            torch.eye(4),
            flow_0i,
            flow_i0,
            self.config,
        )
        source = torch.from_numpy(result["source_yx"].astype(np.int64))
        self.assertGreater(source.shape[0], 0)
        rows = source[:, 0]
        cols = source[:, 1]
        self.assertFalse(
            bool(((rows >= 5) & (rows < 10) & (cols >= 5) & (cols < 10)).any())
        )
        self.assertFalse(bool((rows == 0).any()))
        step = self.W // 2
        self.assertFalse(bool((cols == step - 1).any()))
        self.assertGreater(int((cols == step).sum()), 0)
        fb = torch.from_numpy(result["fb_error_px"])
        self.assertTrue(bool((fb <= self.config.fb_tolerance_px + 1e-6).all()))
        self.assertTrue(
            bool((torch.from_numpy(result["flow_m"]).norm(dim=-1) < 1e-4).all())
        )

    def test_sparse_contract_dtypes(self):
        depth0, depth_i, flow_0i, flow_i0 = self._static_inputs()
        result = estimate_sparse_targets(
            depth0,
            depth_i,
            self.intrinsics,
            self.intrinsics,
            self.c2w,
            flow_0i,
            flow_i0,
            self.config,
        )
        self.assertEqual(result["source_yx"].dtype, np.uint16)
        self.assertEqual(result["flow_m"].dtype, np.float32)
        self.assertEqual(result["motion_label"].dtype, np.int8)
        self.assertEqual(result["fb_error_px"].dtype, np.float32)
        source = result["source_yx"].astype(np.int64)
        self.assertLess(int(source[:, 0].max()), BASE_H)
        self.assertIn("coverage", result["stats"])


def write_flow_cache(root, split, row, frame_ids, pairs, dataset_digest="test-digest"):
    """Write a synthetic v2 flow cache for one sequence."""
    sequence_dir = os.path.join(root, split, f"sequence_{row:06d}")
    os.makedirs(sequence_dir, exist_ok=True)
    for target_view, pair in enumerate(pairs, start=1):
        if pair is None:
            continue
        np.savez(
            os.path.join(sequence_dir, f"view_0_to_{target_view}.npz"),
            source_yx=np.asarray(pair["source_yx"], dtype=np.uint16),
            flow_m=np.asarray(pair["flow_m"], dtype=np.float32),
            motion_label=np.asarray(pair["motion_label"], dtype=np.int8),
            fb_error_px=np.asarray(pair["fb_error_px"], dtype=np.float32),
            frame_ids=np.asarray(frame_ids, dtype=np.int64),
            target_view=np.asarray(target_view, dtype=np.int64),
        )
    manifest = {
        "schema": FLOW_SCHEMA,
        "completed": True,
        "dataset_digest": dataset_digest,
        "pose_convention": "stored_extrinsics_are_w2c",
        "teacher": {
            "name": "torchvision_raft_large_C_T_SKHT_V2",
            "sha256": "ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322",
        },
    }
    manifest_path = os.path.join(root, "manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        manifest["schema"] = FLOW_SCHEMA
        manifest["completed"] = True
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream)


def synthetic_pair(count, dynamic=0):
    source_yx = np.stack(
        [np.arange(count) % BASE_H, (np.arange(count) * 7) % BASE_W], axis=1
    )
    flow_m = np.zeros((count, 3), dtype=np.float32)
    labels = np.zeros(count, dtype=np.int8)
    if dynamic:
        labels[:dynamic] = 1
        flow_m[:dynamic, 0] = 0.2
    return {
        "source_yx": source_yx,
        "flow_m": flow_m,
        "motion_label": labels,
        "fb_error_px": np.full(count, 0.3, dtype=np.float32),
    }

DATA_ROOT = os.environ.get("GEOMETRY_DATA_ROOT")
EXPECTED_FIXED4 = {"train": 879, "val": 163, "smoke": 51}
SCENE_DIR = "scenes/scene_000000"


def setUpModule():
    if not DATA_ROOT:
        raise unittest.SkipTest(
            "GEOMETRY_DATA_ROOT is not set; skipping real-data dataset tests."
        )
    if not os.path.isfile(os.path.join(DATA_ROOT, SCENE_DIR, "cameras.npz")):
        raise unittest.SkipTest(f"No cameras.npz under {DATA_ROOT}")


class DatasetContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets = {
            split: GeometryDataset(DATA_ROOT, split, seed=42, epoch=0)
            for split in EXPECTED_FIXED4
        }

    def test_schema_and_fixed4_counts(self):
        for split, expected in EXPECTED_FIXED4.items():
            dataset = self.datasets[split]
            self.assertEqual(len(dataset), expected, split)

        views, targets = collate_geometry([self.datasets["train"][0]])
        self.assertEqual(len(views), 4)
        for view in views:
            self.assertEqual(tuple(view["img"].shape), (1, 3, PADDED_H, PADDED_W))
            self.assertEqual(tuple(view["depth_z"].shape), (1, PADDED_H, PADDED_W, 1))
            self.assertEqual(tuple(view["intrinsics"].shape), (1, 3, 3))
            self.assertEqual(tuple(view["is_metric_scale"].shape), (1,))
            self.assertTrue(bool(view["is_metric_scale"][0]))
            self.assertEqual(view["data_norm_type"], ["dinov2"])
        self.assertEqual(tuple(targets["depth_z_m"].shape), (1, 4, BASE_H, BASE_W, 1))
        self.assertEqual(tuple(targets["valid_mask"].shape), (1, 4, BASE_H, BASE_W))
        self.assertEqual(tuple(targets["holdout_mask"].shape), (1, 4, BASE_H, BASE_W))
        self.assertEqual(tuple(targets["rays"].shape), (1, 4, BASE_H, BASE_W, 3))
        self.assertEqual(tuple(targets["c2w_ref"].shape), (1, 4, 4, 4))
        self.assertEqual(tuple(targets["frame_ids"].shape), (1, 4))

    def test_mm_z_and_padding(self):
        dataset = self.datasets["train"]
        views, targets = collate_geometry([dataset[0]])
        frame_id = int(targets["frame_ids"][0, 0])
        row = dataset.frame_row[frame_id]
        depth_png = cv2.imread(
            os.path.join(
                DATA_ROOT, SCENE_DIR, "depth", f"frame_{frame_id:06d}.png"
            ),
            cv2.IMREAD_UNCHANGED,
        )
        self.assertEqual(depth_png.dtype, np.uint16)
        self.assertEqual(depth_png.shape, (BASE_H, BASE_W))
        valid_png = (depth_png > 0) & (depth_png <= 1300)
        target = targets["depth_z_m"][0, 0, ..., 0].numpy()
        np.testing.assert_allclose(
            target[valid_png], depth_png[valid_png].astype(np.float32) * DEPTH_MM_TO_M,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            targets["valid_mask"][0, 0].numpy(), valid_png
        )
        self.assertTrue(bool(dataset.quality[row]))

        depth_input = views[0]["depth_z"][0, ..., 0]
        self.assertEqual(float(depth_input[BASE_H:, :].abs().sum()), 0.0)
        self.assertEqual(float(depth_input[:, BASE_W:].abs().sum()), 0.0)
        holdout = targets["holdout_mask"][0, 0]
        self.assertGreater(int(holdout.sum()), 0)
        held = depth_input[:BASE_H, :BASE_W][holdout]
        self.assertEqual(float(held.abs().sum()), 0.0)
        kept = targets["valid_mask"][0, 0] & ~holdout
        np.testing.assert_allclose(
            depth_input[:BASE_H, :BASE_W][kept].numpy(),
            target[kept.numpy()],
            rtol=0.0,
            atol=0.0,
        )

        img = views[0]["img"][0]
        right = img[:, :, BASE_W:]
        np.testing.assert_array_equal(
            right.numpy(), img[:, :, BASE_W - 1 : BASE_W].expand(-1, -1, PADDED_W - BASE_W).numpy()
        )
        bottom = img[:, BASE_H:, :]
        np.testing.assert_array_equal(
            bottom.numpy(), img[:, BASE_H - 1 : BASE_H, :].expand(-1, PADDED_H - BASE_H, -1).numpy()
        )

    def test_c2w_reference_identity(self):
        dataset = self.datasets["val"]
        _, targets = collate_geometry([dataset[0]])
        c2w = targets["c2w_ref"][0].numpy()
        np.testing.assert_allclose(c2w[0], np.eye(4, dtype=np.float32), rtol=0.0, atol=1e-5)

        frame_ids = targets["frame_ids"][0].numpy()
        w2c0 = np.eye(4, dtype=np.float64)
        w2c0[:3, :4] = dataset.w2c[dataset.frame_row[int(frame_ids[0])]]
        for i, frame_id in enumerate(frame_ids):
            w2c_i = np.eye(4, dtype=np.float64)
            w2c_i[:3, :4] = dataset.w2c[dataset.frame_row[int(frame_id)]]
            expected = w2c0 @ np.linalg.inv(w2c_i)
            np.testing.assert_allclose(c2w[i], expected, rtol=0.0, atol=1e-5)

    def test_relative_pose_from_stored_w2c(self):
        from any4d.training.flow_teacher import relative_c2w_from_stored

        def rot(angle):
            return np.array(
                [
                    [np.cos(angle), 0.0, np.sin(angle)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(angle), 0.0, np.cos(angle)],
                ]
            )

        c2w0 = np.eye(4)
        c2w0[:3, :3] = rot(0.2)
        c2w0[:3, 3] = [0.1, 0.0, 0.0]
        c2w_i = np.eye(4)
        c2w_i[:3, :3] = rot(-0.35)
        c2w_i[:3, 3] = [0.05, 0.1, 0.2]
        w2c0 = np.linalg.inv(c2w0)
        w2c_i = np.linalg.inv(c2w_i)
        relative = relative_c2w_from_stored(w2c0, w2c_i)
        point_world = np.array([0.3, -0.2, 0.9, 1.0])
        point_cam_i = np.linalg.inv(c2w_i) @ point_world
        point_cam_0 = np.linalg.inv(c2w0) @ point_world
        np.testing.assert_allclose(
            relative @ point_cam_i, point_cam_0, rtol=0.0, atol=1e-9
        )

    def test_no_split_leakage(self):
        frames_by_split = {}
        for split, dataset in self.datasets.items():
            frames = set()
            for record in dataset.sequence_records:
                self.assertEqual(len(record["frame_ids"]), 4)
                for frame_id, row in zip(record["frame_ids"], record["frame_rows"]):
                    self.assertTrue(bool(dataset.quality[row]))
                    self.assertEqual(
                        int(dataset.frame_chunks[row]), int(record["chunk_id"]), split
                    )
                    frames.add(int(frame_id))
            frames_by_split[split] = frames
        self.assertFalse(frames_by_split["train"] & frames_by_split["val"])
        self.assertFalse(frames_by_split["train"] & frames_by_split["smoke"])
        self.assertFalse(frames_by_split["val"] & frames_by_split["smoke"])

    def test_input_depth_holdout(self):
        dataset = GeometryDataset(DATA_ROOT, "train", seed=42, epoch=0)
        _, targets_a = collate_geometry([dataset[0]])
        other = GeometryDataset(DATA_ROOT, "train", seed=42, epoch=0)
        _, targets_b = collate_geometry([other[0]])
        np.testing.assert_array_equal(
            targets_a["holdout_mask"].numpy(), targets_b["holdout_mask"].numpy()
        )
        self.assertTrue(
            bool((targets_a["holdout_mask"] & ~targets_a["valid_mask"]).sum() == 0)
        )
        for index in range(min(5, len(dataset))):
            _, targets = collate_geometry([dataset[index]])
            for view in range(4):
                self.assertGreater(
                    int(targets["holdout_mask"][0, view].sum()), 0, (index, view)
                )

        epoch_one = GeometryDataset(DATA_ROOT, "train", seed=42, epoch=1)
        changed = False
        for index in range(10):
            _, targets_c = collate_geometry([epoch_one[index]])
            _, targets_d = collate_geometry([dataset[index]])
            if not np.array_equal(
                targets_c["holdout_mask"].numpy(), targets_d["holdout_mask"].numpy()
            ):
                changed = True
                break
        self.assertTrue(changed, "epoch must change the train holdout pattern")

        val = self.datasets["val"]
        for index in range(3):
            _, targets = collate_geometry([val[index]])
            for view in range(4):
                self.assertGreater(
                    int(targets["holdout_mask"][0, view].sum()), 0, (index, view)
                )

    def test_holdout_matches_documented_scheme(self):
        dataset = self.datasets["train"]
        record = dataset.sequence_records[0]
        text = f"{dataset.seed}/{dataset.epoch}/scene_000000/{record['row']}/0"
        seed = int.from_bytes(
            hashlib.sha256(text.encode("utf-8")).digest()[:8], "little"
        ) & (2**63 - 1)
        generator = torch.Generator()
        generator.manual_seed(seed)
        blocks = torch.rand((BASE_H // 16, BASE_W // 16), generator=generator) < 0.2
        expected = (
            blocks.repeat_interleave(16, dim=0).repeat_interleave(16, dim=1).numpy()
        )
        np.testing.assert_array_equal(
            dataset._holdout_pattern(record["row"], 0), expected
        )

    def test_dataset_digest_is_reproducible(self):
        digest_a = compute_dataset_digest(DATA_ROOT)
        digest_b = compute_dataset_digest(DATA_ROOT)
        self.assertEqual(digest_a, digest_b)
        self.assertEqual(len(digest_a), 64)
        int(digest_a, 16)


class FlowCacheContractTest(unittest.TestCase):
    DIGEST = "test-digest"

    @classmethod
    def setUpClass(cls):
        cls.base = GeometryDataset(DATA_ROOT, "train", seed=42, epoch=0)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write_cache_for_all(self, overrides):
        empty = [synthetic_pair(0) for _ in range(3)]
        for row, record in enumerate(self.base.sequence_records):
            pairs = overrides.get(row, empty)
            write_flow_cache(
                self.tmp.name,
                "train",
                record["row"],
                record["frame_ids"],
                pairs,
                dataset_digest=self.DIGEST,
            )

    def test_flow_cache_frame_identity(self):
        self._write_cache_for_all(
            {
                0: [
                    synthetic_pair(10, dynamic=2),
                    synthetic_pair(13),
                    synthetic_pair(9, dynamic=1),
                ]
            }
        )
        dataset = GeometryDataset(DATA_ROOT, "train", flow_root=self.tmp.name, dataset_digest=self.DIGEST)
        self.assertEqual(len(dataset), 1)
        views, targets = collate_geometry([dataset[0]])
        for view in views:
            self.assertNotIn("flow_pairs", view)
        pairs = targets["flow_pairs"][0]
        record = dataset.sequence_records[0]
        self.assertEqual(len(pairs), 3)
        for index, pair in enumerate(pairs, start=1):
            self.assertEqual(int(pair["target_view"]), index)
            np.testing.assert_array_equal(
                pair["frame_ids"].numpy(), np.asarray(record["frame_ids"])
            )
            source = pair["source_yx"].numpy()
            self.assertLess(int(source[:, 0].max()), BASE_H)
            self.assertLess(int(source[:, 1].max()), BASE_W)
            self.assertEqual(source.shape[1], 2)
            self.assertEqual(pair["flow_m"].shape[1], 3)
            self.assertEqual(
                pair["flow_m"].shape[0],
                pair["motion_label"].shape[0],
            )

    def test_unknown_flow_is_not_zero_target(self):
        self._write_cache_for_all(
            {
                0: [synthetic_pair(0), synthetic_pair(12, dynamic=2), synthetic_pair(8)],
                1: [synthetic_pair(0) for _ in range(3)],
            }
        )
        dataset = GeometryDataset(DATA_ROOT, "train", flow_root=self.tmp.name, dataset_digest=self.DIGEST)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(
            dataset.flow_filtered["all_empty"], len(self.base) - 1
        )

        _, targets = collate_geometry([dataset[0]])
        empty_pair = targets["flow_pairs"][0][0]
        self.assertEqual(int(empty_pair["flow_m"].shape[0]), 0)
        self.assertEqual(int(empty_pair["source_yx"].shape[0]), 0)
        self.assertFalse(bool(torch.any(empty_pair["flow_m"] != 0)))
        dynamic_pair = targets["flow_pairs"][0][1]
        self.assertGreater(int(dynamic_pair["flow_m"].shape[0]), 0)

        missing = tempfile.TemporaryDirectory()
        self.addCleanup(missing.cleanup)
        record = self.base.sequence_records[0]
        write_flow_cache(
            missing.name,
            "train",
            0,
            record["frame_ids"],
            [None, None, None],
            dataset_digest=self.DIGEST,
        )
        with self.assertRaises(FileNotFoundError):
            GeometryDataset(DATA_ROOT, "train", flow_root=missing.name, dataset_digest=self.DIGEST)

    def test_flow_cache_digest_mismatch_rejected(self):
        self._write_cache_for_all({})
        with self.assertRaisesRegex(ValueError, "different dataset digest"):
            GeometryDataset(
                DATA_ROOT,
                "train",
                flow_root=self.tmp.name,
                dataset_digest="other-digest",
            )

    def test_flow_cache_pose_convention_is_checked(self):
        self._write_cache_for_all({})
        manifest_path = os.path.join(self.tmp.name, "manifest.json")
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        manifest["pose_convention"] = "stored_extrinsics_are_c2w"
        with open(manifest_path, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream)
        with self.assertRaisesRegex(ValueError, "pose convention mismatch"):
            GeometryDataset(
                DATA_ROOT,
                "train",
                flow_root=self.tmp.name,
                dataset_digest=self.DIGEST,
            )

    def test_fractional_flow_coordinates_rejected(self):
        self._write_cache_for_all({})
        record = self.base.sequence_records[0]
        pair_path = os.path.join(
            self.tmp.name,
            "train",
            f"sequence_{record['row']:06d}",
            "view_0_to_1.npz",
        )
        np.savez(
            pair_path,
            source_yx=np.array(
                [[1.25, 1.75], [2.5, 3.5], [4.0, 5.0], [6.0, 7.0]], dtype=np.float32
            ),
            flow_m=np.zeros((4, 3), dtype=np.float32),
            motion_label=np.zeros(4, dtype=np.int8),
            fb_error_px=np.zeros(4, dtype=np.float32),
            frame_ids=np.asarray(record["frame_ids"], dtype=np.int64),
            target_view=np.asarray(1, dtype=np.int64),
        )
        with self.assertRaises(ValueError):
            GeometryDataset(
                DATA_ROOT,
                "train",
                flow_root=self.tmp.name,
                dataset_digest=self.DIGEST,
            )

    def test_duplicate_flow_coordinates_rejected(self):
        duplicate_pair = synthetic_pair(2)
        duplicate_pair["source_yx"] = np.array([[1, 1], [1, 1]], dtype=np.uint16)
        self._write_cache_for_all(
            {0: [duplicate_pair, synthetic_pair(2), synthetic_pair(2)]}
        )
        with self.assertRaises(ValueError):
            GeometryDataset(
                DATA_ROOT,
                "train",
                flow_root=self.tmp.name,
                dataset_digest=self.DIGEST,
            )


if __name__ == "__main__":
    unittest.main()
