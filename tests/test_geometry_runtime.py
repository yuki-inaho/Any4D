"""Tests for the geometry runtime (model init, AMUSE, resume, export)."""

import importlib.util
import json
import os
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from any4d.training.geometry_runtime import (
    AMUSE_OUTPUT_PREFIXES,
    build_amuse,
    ensure_new_directory,
    evaluate_geometry,
    export_inference_checkpoint,
    file_sha256,
    flow_cache_digest,
    info_sharing_block_prefixes,
    legacy_flow_digest,
    load_geometry_state_dict,
    load_resume_checkpoint,
    migrate_checkpoint_flow_digest,
    prepare_model_inputs,
    run_geometry_training,
    save_best_checkpoint,
    save_last_checkpoint,
    set_trainable_geometry_heads,
    TRAINABLE_PREFIXES,
    validate_training_config,
    verify_vendored_amuse,
)


def load_train_cli_module():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "train_geometry.py",
    )
    spec = importlib.util.spec_from_file_location("train_geometry_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeGeometryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.fusion_norm_layer = nn.LayerNorm(4)
        self.scene_flow_dense_head = nn.Linear(4, 4)
        self.dpt_feature_head = nn.Sequential(nn.Conv2d(2, 2, 1), nn.ReLU())
        self.dpt_regressor_head = nn.Sequential(
            nn.Conv2d(2, 2, 1), nn.ReLU(), nn.Conv2d(2, 2, 1)
        )
        self.pose_head = nn.Linear(4, 4)
        self.scale_head = nn.Linear(4, 1)
        self.scene_flow_dpt_feature_head = nn.Linear(4, 4)
        self.scene_flow_dpt_regressor_head = nn.Sequential(
            nn.Conv2d(2, 2, 1), nn.ReLU(), nn.Conv2d(2, 2, 1)
        )

    def forward(self, x):
        return x


def make_view():
    return {
        "img": torch.zeros(1, 3, 8, 12),
        "depth_z": torch.ones(1, 8, 12, 1),
        "intrinsics": torch.eye(3).unsqueeze(0),
        "is_metric_scale": torch.ones(1, dtype=torch.bool),
        "data_norm_type": ["dinov2"],
    }


class GeometryRuntimeTest(unittest.TestCase):
    def test_trainable_allowlist(self):
        model = FakeGeometryModel()
        names = set(set_trainable_geometry_heads(model))
        self.assertTrue(names)
        for name, parameter in model.named_parameters():
            expected = name.startswith(TRAINABLE_PREFIXES)
            self.assertEqual(parameter.requires_grad, expected, name)
        self.assertTrue(model.dpt_feature_head.training)
        self.assertTrue(model.pose_head.training)
        self.assertTrue(model.scale_head.training)
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.scene_flow_dense_head.training)
        self.assertTrue(
            any(name.startswith("dpt_regressor_head.") for name in names)
        )

        with self.assertRaises(ValueError):
            set_trainable_geometry_heads(model, prefixes=("does_not_exist.",))

    def test_model_input_has_no_pose_target(self):
        views = [make_view() for _ in range(4)]
        processed = prepare_model_inputs([dict(view) for view in views])
        for view in processed:
            for key in (
                "camera_poses",
                "camera_pose_quats",
                "camera_pose_trans",
                "c2w_ref",
                "depth_z_m",
                "valid_mask",
                "holdout_mask",
            ):
                self.assertNotIn(key, view)

        contaminated = [make_view() for _ in range(4)]
        contaminated[0]["c2w_ref"] = torch.eye(4)
        with self.assertRaises(ValueError):
            prepare_model_inputs(contaminated)

    def test_initial_load_known_unused_keys(self):
        model = FakeGeometryModel()
        state = dict(model.state_dict())
        state["event_encoder.conv_in.weight"] = torch.zeros(1, 1, 3, 3)
        state["event_encoder.pos_embed"] = torch.zeros(4, 4)
        state["scene_flow_encoder.pos_embed"] = torch.zeros(4, 4)
        state["scene_flow_encoder.post_pe_norm.weight"] = torch.zeros(4)
        report = load_geometry_state_dict(model, state)
        self.assertEqual(report["missing"], [])
        self.assertEqual(len(report["allowed_unexpected"]), 4)

    def test_initial_load_rejects_missing_key(self):
        model = FakeGeometryModel()
        incomplete = {
            key: value
            for key, value in model.state_dict().items()
            if key != "encoder.weight"
        }
        with self.assertRaises(RuntimeError):
            load_geometry_state_dict(model, incomplete)

    def test_initial_load_rejects_unknown_extra_key(self):
        model = FakeGeometryModel()
        with_unknown = dict(model.state_dict())
        with_unknown["bogus.weight"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            load_geometry_state_dict(model, with_unknown)


class FakeAmuseModel(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()

        class ConvBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2 = nn.Sequential(
                    nn.Conv2d(2, 2, 1), nn.ReLU(), nn.Conv2d(2, 2, 1)
                )
                self.fc_t = nn.Linear(4, 3)
                self.fc_rot = nn.Linear(4, 4)

        self.dpt_regressor_head = ConvBlock()
        self.pose_head = ConvBlock()
        self.scale_head = nn.Module()
        self.scale_head.output_proj = nn.Linear(4, 1)
        self.dpt_feature_head = nn.Linear(4, 4)
        self.scene_flow_dpt_regressor_head = ConvBlock()
        self.scene_flow_dpt_feature_head = nn.Linear(4, 4)
        self.info_sharing = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.to(dtype)


class AmuseWiringTest(unittest.TestCase):
    def setUp(self):
        self.model = FakeAmuseModel(dtype=torch.float64)
        self.names = [name for name, _ in self.model.named_parameters()]

    def test_vendored_amuse_source_hash(self):
        self.assertEqual(
            verify_vendored_amuse(),
            "84fd3fbbc99e1718cf1c821ceff3369439f48e6fbd8ecc3a2b83afa5d82eea1f",
        )

    def test_amuse_parameter_coverage(self):
        optimizer, manifest = build_amuse(
            self.model, self.names, warmup_steps=2
        )
        covered = []
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                covered.append(id(parameter))
        self.assertEqual(len(covered), len(set(covered)))
        self.assertEqual(len(covered), len(self.names))
        self.assertEqual(len(manifest["groups"]), 2)

        manifest_names = [
            entry["name"]
            for group in manifest["groups"]
            for entry in group["params"]
        ]
        self.assertEqual(sorted(manifest_names), sorted(self.names))

        muon_names = {
            entry["name"]
            for group in manifest["groups"]
            if group["update_type"] == "muon"
            for entry in group["params"]
        }
        aux_names = set(manifest_names) - muon_names
        self.assertIn("dpt_feature_head.weight", muon_names)
        self.assertIn("dpt_regressor_head.conv2.0.weight", muon_names)
        self.assertIn("scene_flow_dpt_feature_head.weight", muon_names)
        self.assertIn("scene_flow_dpt_regressor_head.conv2.0.weight", muon_names)
        self.assertIn("dpt_regressor_head.conv2.2.weight", aux_names)
        self.assertIn("scene_flow_dpt_regressor_head.conv2.2.weight", aux_names)
        self.assertIn("pose_head.fc_t.bias", aux_names)
        self.assertIn("pose_head.fc_rot.weight", aux_names)
        self.assertIn("scale_head.output_proj.weight", aux_names)
        self.assertTrue(AMUSE_OUTPUT_PREFIXES)

    def test_info_sharing_block_prefixes(self):
        model = FakeAmuseModel()
        self.assertEqual(
            info_sharing_block_prefixes(model, 2),
            ("info_sharing.1.", "info_sharing.2."),
        )
        with self.assertRaises(ValueError):
            info_sharing_block_prefixes(model, 4)

    def test_amuse_low_lr_groups(self):
        model = FakeAmuseModel(dtype=torch.float64)
        names = set_trainable_geometry_heads(
            model, prefixes=TRAINABLE_PREFIXES + ("info_sharing.2.",)
        )
        optimizer, manifest = build_amuse(
            model,
            names,
            warmup_steps=2,
            low_lr_prefixes=("info_sharing.2.",),
            low_lr_muon_lr=1e-5,
            low_lr_aux_lr=1e-6,
        )
        covered = [id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(covered), len(set(covered)))
        self.assertEqual(len(covered), len(names))
        low_muon = [
            g
            for g in manifest["groups"]
            if g["update_type"] == "muon" and abs(g["lr"] - 1e-5) < 1e-12
        ]
        low_aux = [
            g
            for g in manifest["groups"]
            if g["update_type"] == "adamw" and abs(g["lr"] - 1e-6) < 1e-12
        ]
        self.assertEqual(len(low_muon), 1)
        self.assertEqual(len(low_aux), 1)
        self.assertTrue(
            any(p["name"] == "info_sharing.2.weight" for p in low_muon[0]["params"])
        )
        self.assertTrue(
            any(p["name"] == "info_sharing.2.bias" for p in low_aux[0]["params"])
        )

    def test_amuse_x_y_roundtrip(self):
        optimizer, _ = build_amuse(self.model, self.names, warmup_steps=2)
        optimizer.train()
        for step in range(2):
            torch.manual_seed(step)
            for _, parameter in self.model.named_parameters():
                parameter.grad = torch.randn_like(parameter)
            optimizer.step()
        y_after_step = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }
        optimizer.eval()
        x_now = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }
        optimizer.train()
        for name, parameter in self.model.named_parameters():
            torch.testing.assert_close(
                parameter, y_after_step[name], rtol=1e-6, atol=1e-9, msg=name
            )
        changed = any(
            not torch.allclose(x_now[name], y_after_step[name]) for name in self.names
        )
        self.assertTrue(changed)

    def test_amuse_second_update_finite(self):
        optimizer, _ = build_amuse(self.model, self.names, warmup_steps=2)
        optimizer.train()
        for step in range(2):
            torch.manual_seed(step)
            for _, parameter in self.model.named_parameters():
                parameter.grad = torch.randn_like(parameter)
            optimizer.step()
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.isfinite(parameter).all(), name)
        for group in optimizer.param_groups:
            self.assertEqual(group["k"], 2)
            for tensor in group["params"]:
                self.assertTrue(bool(torch.isfinite(tensor).all()))


def make_contract(**overrides):
    contract = {
        "base_sha256": "base-hash",
        "dataset_digest": "dataset-digest",
        "flow_digest": "flow-digest",
        "objective_id": "any4d_factorized_pseudo_sf_v2",
        "split": "train",
        "mask_scheme": "block16_p0.2_seed42",
        "resolution": [480, 640],
        "views": 4,
        "precision": "bf16",
        "max_updates": 10,
        "epochs": 3,
    }
    contract.update(overrides)
    return contract


class CheckpointRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = FakeAmuseModel(dtype=torch.float64)
        self.names = set_trainable_geometry_heads(self.model)
        self.optimizer, self.manifest = build_amuse(
            self.model, self.names, warmup_steps=2
        )
        self.optimizer.train()
        self.contract = make_contract()

    def _path(self, name):
        return os.path.join(self.tmp.name, name)

    def _step(self, seed):
        torch.manual_seed(seed)
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                parameter.grad = torch.randn_like(parameter)
        self.optimizer.step()

    def _save_last(self, path, global_step=2):
        save_last_checkpoint(
            path,
            model=self.model,
            trainable_names=self.names,
            optimizer=self.optimizer,
            contract=self.contract,
            manifest=self.manifest,
            resolved_config={"height": 480, "width": 640},
            global_step=global_step,
            epoch=0,
            cursor=global_step,
            shuffle_permutation=[0, 1, 2],
            best_metric=None,
        )

    def test_resume_next_step_equivalence(self):
        self._step(0)
        self._step(1)
        self._save_last(self._path("last.pt"))
        self._step(2)
        expected = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }

        resumed_model = FakeAmuseModel(dtype=torch.float64)
        resumed_model.load_state_dict(self.model.state_dict())
        resumed_names = set_trainable_geometry_heads(resumed_model)
        for name, parameter in resumed_model.named_parameters():
            if parameter.requires_grad:
                parameter.data.copy_(torch.randn_like(parameter))
        resumed_optimizer, resumed_manifest = build_amuse(
            resumed_model, resumed_names, warmup_steps=2
        )
        metadata = load_resume_checkpoint(
            self._path("last.pt"),
            model=resumed_model,
            optimizer=resumed_optimizer,
            manifest=resumed_manifest,
            contract=self.contract,
            trainable_names=resumed_names,
        )
        self.assertEqual(metadata["global_step"], 2)
        self.assertTrue(resumed_optimizer.train_mode)
        torch.manual_seed(2)
        for name, parameter in resumed_model.named_parameters():
            if parameter.requires_grad:
                parameter.grad = torch.randn_like(parameter)
        resumed_optimizer.step()
        for name, parameter in resumed_model.named_parameters():
            torch.testing.assert_close(
                parameter, expected[name], rtol=1e-6, atol=1e-9, msg=name
            )

    def test_resume_rejects_contract_drift(self):
        self._step(0)
        self._save_last(self._path("last.pt"))
        drifted = make_contract(dataset_digest="other-digest")
        with self.assertRaises(ValueError):
            load_resume_checkpoint(
                self._path("last.pt"),
                model=self.model,
                optimizer=self.optimizer,
                manifest=self.manifest,
                contract=drifted,
                trainable_names=self.names,
            )
        legacy = make_contract(objective_id="any4d_metric_z_v1")
        with self.assertRaises(ValueError):
            load_resume_checkpoint(
                self._path("last.pt"),
                model=self.model,
                optimizer=self.optimizer,
                manifest=self.manifest,
                contract=legacy,
                trainable_names=self.names,
            )
        with self.assertRaises(ValueError):
            load_resume_checkpoint(
                self._path("last.pt"),
                model=self.model,
                optimizer=self.optimizer,
                manifest=self.manifest,
                contract=self.contract,
                trainable_names=list(self.names) + ["bogus.weight"],
            )

    def test_export_strict_roundtrip(self):
        base = FakeAmuseModel(dtype=torch.float64)
        base.load_state_dict(self.model.state_dict())
        base_path = self._path("base.pt")
        torch.save({"model": base.state_dict()}, base_path)
        self.contract["base_sha256"] = file_sha256(base_path)
        self._step(0)
        self._step(1)
        self.optimizer.eval()
        best_path = self._path("best.pt")
        save_best_checkpoint(
            best_path,
            model=self.model,
            trainable_names=self.names,
            contract=self.contract,
            manifest=self.manifest,
            resolved_config={"height": 480, "width": 640},
            metric=0.123,
            global_step=2,
        )
        x_expected = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }
        self.optimizer.train()
        output = self._path("export.pt")
        metadata = export_inference_checkpoint(
            base_path,
            best_path,
            output,
            model_factory=lambda: FakeAmuseModel(dtype=torch.float64),
        )
        self.assertTrue(os.path.isfile(output))
        self.assertEqual(metadata["best_metric"], 0.123)
        payload = torch.load(output, map_location="cpu", weights_only=False)
        fresh = FakeAmuseModel(dtype=torch.float64)
        fresh.load_state_dict(payload["model"], strict=True)
        for name, parameter in fresh.named_parameters():
            torch.testing.assert_close(
                parameter, x_expected[name], rtol=0.0, atol=0.0, msg=name
            )

        changed = any(
            not torch.equal(parameter, dict(base.named_parameters())[name])
            for name, parameter in self.model.named_parameters()
        )
        self.assertTrue(changed)

    def test_existing_output_is_not_overwritten(self):
        existing_dir = self._path("run")
        os.makedirs(existing_dir)
        with self.assertRaises(FileExistsError):
            ensure_new_directory(existing_dir)
        ensure_new_directory(self._path("fresh-run"))

        base_path = self._path("base.pt")
        torch.save({"model": FakeAmuseModel(dtype=torch.float64).state_dict()}, base_path)
        self.contract["base_sha256"] = file_sha256(base_path)
        self._step(0)
        self.optimizer.eval()
        best_path = self._path("best.pt")
        save_best_checkpoint(
            best_path,
            model=self.model,
            trainable_names=self.names,
            contract=self.contract,
            manifest=self.manifest,
            resolved_config={},
            metric=0.1,
            global_step=1,
        )
        self.optimizer.train()
        output = self._path("export.pt")
        export_inference_checkpoint(
            base_path,
            best_path,
            output,
            model_factory=lambda: FakeAmuseModel(dtype=torch.float64),
        )
        with self.assertRaises(FileExistsError):
            export_inference_checkpoint(
                base_path,
                best_path,
                output,
                model_factory=lambda: FakeAmuseModel(dtype=torch.float64),
            )

    def test_export_rejects_missing_metadata(self):
        base_path = self._path("base.pt")
        torch.save({"model": FakeAmuseModel(dtype=torch.float64).state_dict()}, base_path)
        self.contract["base_sha256"] = file_sha256(base_path)
        self._step(0)
        self.optimizer.eval()
        best_path = self._path("best.pt")
        save_best_checkpoint(
            best_path,
            model=self.model,
            trainable_names=self.names,
            contract=self.contract,
            manifest=self.manifest,
            resolved_config={},
            metric=0.1,
            global_step=1,
        )
        self.optimizer.train()
        payload = torch.load(best_path, map_location="cpu", weights_only=False)

        without_objective = dict(payload)
        without_objective.pop("objective_id")
        stripped_objective = self._path("best_no_objective.pt")
        torch.save(without_objective, stripped_objective)
        with self.assertRaisesRegex(ValueError, "objective_id"):
            export_inference_checkpoint(
                base_path,
                stripped_objective,
                self._path("export_no_objective.pt"),
                model_factory=lambda: FakeAmuseModel(dtype=torch.float64),
            )

        without_base = dict(payload)
        without_base.pop("base_sha256")
        stripped_base = self._path("best_no_base.pt")
        torch.save(without_base, stripped_base)
        with self.assertRaisesRegex(ValueError, "base_sha256"):
            export_inference_checkpoint(
                base_path,
                stripped_base,
                self._path("export_no_base.pt"),
                model_factory=lambda: FakeAmuseModel(dtype=torch.float64),
            )

    def test_export_rejects_wrong_base(self):
        base_path = self._path("base.pt")
        torch.save({"model": FakeAmuseModel(dtype=torch.float64).state_dict()}, base_path)
        contract = make_contract(base_sha256=file_sha256(base_path))
        self._step(0)
        self.optimizer.eval()
        best_path = self._path("best.pt")
        save_best_checkpoint(
            best_path,
            model=self.model,
            trainable_names=self.names,
            contract=contract,
            manifest=self.manifest,
            resolved_config={},
            metric=0.1,
            global_step=1,
        )
        self.optimizer.train()
        other_base = self._path("other_base.pt")
        torch.save(
            {"model": FakeAmuseModel(dtype=torch.float64).state_dict()}, other_base
        )
        with self.assertRaises(ValueError):
            export_inference_checkpoint(
                other_base,
                best_path,
                self._path("export.pt"),
                model_factory=lambda: FakeAmuseModel(dtype=torch.float64),
            )


class TinyGeometryDataset(torch.utils.data.Dataset):
    """Small synthetic dataset matching the real collate contract."""

    H, W, V = 8, 12, 4

    def __init__(self, length=2):
        self.length = length
        generator = torch.Generator().manual_seed(7)
        torch.manual_seed(7)
        self.depth = torch.rand(length, self.V, self.H, self.W, 1) * 1.5 + 0.2
        self.holdout = torch.rand(length, self.V, self.H, self.W) < 0.5
        rays = torch.randn(length, self.V, self.H, self.W, 3, generator=generator)
        rays = rays / rays.norm(dim=-1, keepdim=True)
        rays[..., 2] = rays[..., 2].abs()
        self.rays = rays / rays.norm(dim=-1, keepdim=True)
        self.c2w = torch.eye(4).repeat(length, self.V, 1, 1)
        self.c2w[:, 1, :3, 3] = 0.1
        self.c2w[:, 2, 0, 3] = 0.2
        self.c2w[:, 3, :3, :3] = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        self.flow_stats = {"dynamic_points": 3}
        self.manifest = {"settings_hash": "test-flow-digest"}

    def __len__(self):
        return self.length

    def _flow_pairs(self):
        pairs = []
        for target_view in (1, 2, 3):
            pairs.append(
                {
                    "source_yx": torch.tensor(
                        [[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.long
                    ),
                    "flow_m": torch.tensor(
                        [
                            [0.002, 0.0, 0.0],
                            [0.11, 0.01, 0.0],
                            [0.05, 0.0, 0.0],
                            [-0.002, 0.001, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                    "motion_label": torch.tensor([0, 1, -1, 0], dtype=torch.int8),
                    "fb_error_px": torch.full((4,), 0.3),
                    "frame_ids": torch.arange(4),
                    "target_view": torch.tensor(target_view),
                }
            )
        return pairs

    def __getitem__(self, index):
        views = []
        for _ in range(self.V):
            views.append(
                {
                    "img": torch.randn(3, self.H, self.W),
                    "depth_z": self.depth[index, 0].clone(),
                    "intrinsics": torch.eye(3),
                    "is_metric_scale": torch.tensor(True),
                    "data_norm_type": ["dinov2"],
                }
            )
        targets = {
            "depth_z_m": self.depth[index],
            "valid_mask": torch.ones(self.V, self.H, self.W, dtype=torch.bool),
            "holdout_mask": self.holdout[index],
            "rays": self.rays[index],
            "c2w_ref": self.c2w[index],
            "frame_ids": torch.arange(4),
            "sequence_id": torch.tensor(index),
            "chunk_id": torch.tensor(0),
            "flow_pairs": self._flow_pairs(),
        }
        return views, targets


class FakeTrainableModel(FakeAmuseModel):
    def forward(self, views):
        batch, _, height, width = views[0]["img"].shape
        parameter = self.dpt_feature_head.weight.mean()
        scale_value = 1.0 + 0.01 * parameter
        scale = torch.ones(batch, 1) * scale_value
        predictions = []
        for _ in views:
            rays = torch.zeros(batch, height, width, 3)
            rays[..., 2] = 1.0
            depth = torch.full((batch, height, width, 1), 0.5) * (
                1.0 + 0.02 * parameter
            )
            points = depth * rays
            predictions.append(
                {
                    "depth_along_ray": depth,
                    "ray_directions": rays,
                    "cam_trans": torch.zeros(batch, 3) + 0.001 * parameter,
                    "cam_quats": torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(batch, 1),
                    "pts3d": points,
                    "scene_flow": torch.zeros(batch, height, width, 3)
                    + 0.0005 * parameter,
                    "metric_scaling_factor": scale,
                }
            )
        return predictions


class TrainingLoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = FakeTrainableModel(dtype=torch.float32)
        self.names = set_trainable_geometry_heads(self.model)
        self.train_dataset = TinyGeometryDataset(length=3)
        self.eval_dataset = TinyGeometryDataset(length=2)
        self.base_path = os.path.join(self.tmp.name, "base.pt")
        torch.save({"model": self.model.state_dict()}, self.base_path)

    def _config(self, output):
        return {
            "data_root": self.tmp.name,
            "base_checkpoint": self.base_path,
            "output": output,
            "train_split": "train",
            "eval_split": "val",
            "max_updates": 2,
            "eval_limit": 2,
            "epochs": 2,
            "warmup_steps": 1,
            "height": 480,
            "width": 640,
            "views": 4,
            "batch_size": 1,
            "precision": "bf16",
            "optimizer": "amuse",
            "seed": 42,
            "resume": None,
            "flow_root": self.tmp.name,
            "flow_digest": "test-flow-digest",
        }

    def test_train_loop_two_updates(self):
        output = os.path.join(self.tmp.name, "run")
        summary = run_geometry_training(
            self._config(output),
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            device="cpu",
            dataset_digest="test-digest",
            base_sha256="test-base",
        )
        self.assertTrue(os.path.isfile(os.path.join(output, "metrics.jsonl")))
        self.assertTrue(os.path.isfile(os.path.join(output, "last.pt")))
        self.assertTrue(os.path.isfile(os.path.join(output, "best.pt")))
        self.assertTrue(os.path.isfile(os.path.join(output, "summary.json")))
        self.assertEqual(summary["global_step"], 2)
        self.assertTrue(summary["val/objective"] > 0)

        with open(os.path.join(output, "metrics.jsonl"), encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        step_rows = [row for row in rows if row.get("event") == "step"]
        self.assertEqual(len(step_rows), 2)
        for row in step_rows:
            self.assertTrue(torch.isfinite(torch.tensor(row["loss"])))

    def test_train_loop_with_unfreeze_block(self):
        output = os.path.join(self.tmp.name, "run-unfreeze")
        config = self._config(output)
        config["unfreeze_blocks"] = 1
        config["max_updates"] = 2
        summary = run_geometry_training(
            config,
            model=FakeTrainableModel(dtype=torch.float32),
            train_dataset=TinyGeometryDataset(length=2),
            eval_dataset=TinyGeometryDataset(length=1),
            device="cpu",
            dataset_digest="test-digest",
            base_sha256="test-base",
        )
        self.assertGreater(summary["trainable_count"], 0)
        with open(
            os.path.join(output, "trainable_manifest.json"), encoding="utf-8"
        ) as stream:
            manifest = json.load(stream)["manifest"]
        names = [
            entry["name"]
            for group in manifest["groups"]
            for entry in group["params"]
        ]
        self.assertTrue(any(name.startswith("info_sharing.") for name in names))
        levels = {round(group["lr"], 8) for group in manifest["groups"]}
        self.assertIn(round(1e-4 / 3, 8), levels)

    def test_trainer_propagates_epoch_to_dataset(self):
        output = os.path.join(self.tmp.name, "run-epochs")
        dataset = RecordingDataset(length=2)
        config = self._config(output)
        config["epochs"] = 2
        config["max_updates"] = None
        config["eval_limit"] = 1
        run_geometry_training(
            config,
            model=FakeTrainableModel(dtype=torch.float32),
            train_dataset=dataset,
            eval_dataset=TinyGeometryDataset(length=1),
            device="cpu",
            dataset_digest="test-digest",
            base_sha256="test-base",
        )
        self.assertEqual(dataset.epoch_values, [0, 1])

    def test_fixed_shape_rejects_silent_resize(self):
        config = self._config(os.path.join(self.tmp.name, "run2"))
        config["height"] = 240
        with self.assertRaises(ValueError):
            validate_training_config(config)
        config["height"] = 480
        config["views"] = 8
        with self.assertRaises(ValueError):
            validate_training_config(config)
        config["views"] = 4
        del config["flow_root"]
        with self.assertRaises(ValueError):
            validate_training_config(config)

    def test_train_cli_contract(self):
        cli = load_train_cli_module()
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "train",
                "--data-root",
                "/tmp/data",
                "--base-checkpoint",
                "/tmp/base.pt",
                "--output",
                "/tmp/out",
                "--flow-root",
                "/tmp/flow",
                "--train-split",
                "smoke",
                "--eval-split",
                "val",
                "--max-updates",
                "2",
                "--eval-limit",
                "2",
                "--epochs",
                "1",
                "--warmup-steps",
                "1",
                "--height",
                "480",
                "--width",
                "640",
                "--views",
                "4",
                "--batch-size",
                "1",
                "--precision",
                "bf16",
                "--optimizer",
                "amuse",
                "--muon-lr",
                "3e-5",
                "--aux-lr",
                "3e-6",
                "--unfreeze-blocks",
                "2",
                "--unfrozen-muon-lr",
                "1e-5",
                "--unfrozen-aux-lr",
                "1e-6",
                "--objective",
                "any4d_factorized_pseudo_sf_v2",
                "--seed",
                "42",
            ]
        )
        config = cli.args_to_config(args)
        validate_training_config(config)
        self.assertEqual(config["train_split"], "smoke")
        self.assertAlmostEqual(config["muon_lr"], 3e-5)
        self.assertAlmostEqual(config["aux_lr"], 3e-6)
        self.assertEqual(config["unfreeze_blocks"], 2)
        self.assertAlmostEqual(config["unfrozen_muon_lr"], 1e-5)
        self.assertAlmostEqual(config["unfrozen_aux_lr"], 1e-6)

        bad = parser.parse_args(
            [
                "train",
                "--data-root",
                "/tmp/data",
                "--base-checkpoint",
                "/tmp/base.pt",
                "--output",
                "/tmp/out",
                "--flow-root",
                "/tmp/flow",
                "--objective",
                "any4d_metric_z_v1",
            ]
        )
        with self.assertRaises(ValueError):
            cli.args_to_config(bad)

        check = parser.parse_args(["check-data", "--data-root", "/tmp/data"])
        self.assertEqual(check.command, "check-data")
        prepare = parser.parse_args(
            [
                "prepare-flow",
                "--data-root",
                "/tmp/data",
                "--output",
                "/tmp/flow",
                "--teacher",
                "raft_large_C_T_SKHT_V2",
                "--allow-weight-download",
            ]
        )
        self.assertEqual(prepare.command, "prepare-flow")
        self.assertEqual(prepare.output_root, "/tmp/flow")
        self.assertTrue(prepare.allow_weight_download)
        self.assertFalse(prepare.resume)
        with self.assertRaises(SystemExit):
            parser.parse_args(["train", "--data-root"])

        existing = os.path.join(self.tmp.name, "cli-run")
        os.makedirs(existing)
        config["output"] = existing
        with self.assertRaises(FileExistsError):
            cli.run_train(config)


class EvalCliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = FakeTrainableModel(dtype=torch.float32)
        self.names = set_trainable_geometry_heads(self.model)
        self.dataset = TinyGeometryDataset(length=2)
        self.base_path = os.path.join(self.tmp.name, "base.pt")
        torch.save({"model": self.model.state_dict()}, self.base_path)
        self.optimizer, self.manifest = build_amuse(
            self.model, self.names, warmup_steps=1
        )
        self.optimizer.train()
        torch.manual_seed(0)
        for _, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                parameter.grad = torch.randn_like(parameter)
        self.optimizer.step()
        self.optimizer.eval()
        self.metric = evaluate_geometry(self.model, self.dataset, device="cpu")[
            "val/objective"
        ]
        self.contract = make_contract(
            base_sha256="test-base", dataset_digest="test-digest"
        )
        self.best_path = os.path.join(self.tmp.name, "best.pt")
        save_best_checkpoint(
            self.best_path,
            model=self.model,
            trainable_names=self.names,
            contract=self.contract,
            manifest=self.manifest,
            resolved_config={},
            metric=self.metric,
            global_step=1,
        )
        self.optimizer.train()

    def _eval(self):
        cli = load_train_cli_module()
        config = {
            "data_root": self.tmp.name,
            "base_checkpoint": self.base_path,
            "checkpoint": self.best_path,
            "split": "val",
            "compare_base": False,
            "output": os.path.join(self.tmp.name, "final_evaluation.json"),
            "config_dir": None,
        }
        return cli.run_eval(
            config,
            model=FakeTrainableModel(dtype=torch.float32),
            dataset=self.dataset,
            device="cpu",
            base_sha256="test-base",
            dataset_digest="test-digest",
        )

    def test_eval_fixed_holdout(self):
        result = self._eval()
        self.assertEqual(result["samples"], len(self.dataset))
        expected = int(
            torch.stack(
                [self.dataset[index][1]["holdout_mask"] for index in range(len(self.dataset))]
            ).sum()
        )
        self.assertEqual(result["best"]["heldout_depth/count"], expected)

    def test_metric_sum_count_reduction(self):
        result = self._eval()
        sums = result["best"]["loss_sums"]
        expected = sum(
            sums[f"{term}_sum"] / sums[f"{term}_count"]
            for term in ("ray", "quat", "trans", "depth", "point", "scale", "flow")
        )
        self.assertAlmostEqual(
            expected, result["best"]["val/objective"], places=10
        )

    def test_best_metric_reloads_identically(self):
        result = self._eval()
        self.assertTrue(result.get("metric_reproduced"))
        second = self._eval()
        self.assertAlmostEqual(
            result["best"]["val/objective"],
            second["best"]["val/objective"],
            places=10,
        )


class RecordingDataset(TinyGeometryDataset):
    def __init__(self, length=2):
        super().__init__(length)
        self._epoch = 0
        self.epoch_values = []

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        self._epoch = value
        self.epoch_values.append(value)


class FlowDigestTest(unittest.TestCase):
    def test_flow_cache_digest_tracks_npz_content(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = tmp.name
        os.makedirs(os.path.join(root, "train"))
        np.savez(
            os.path.join(root, "train", "pair.npz"),
            flow_m=np.zeros((2, 3), dtype=np.float32),
        )
        with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as stream:
            json.dump({"a": 1}, stream)
        with open(
            os.path.join(root, "flow_quality.json"), "w", encoding="utf-8"
        ) as stream:
            json.dump({"b": 2}, stream)
        first = flow_cache_digest(root)
        np.savez(
            os.path.join(root, "train", "pair.npz"),
            flow_m=np.ones((2, 3), dtype=np.float32),
        )
        second = flow_cache_digest(root)
        self.assertNotEqual(first, second)
        self.assertEqual(second, flow_cache_digest(root))


class FlowDigestMigrationTest(unittest.TestCase):
    def _cache(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = tmp.name
        os.makedirs(os.path.join(root, "train"))
        np.savez(
            os.path.join(root, "train", "pair.npz"),
            flow_m=np.zeros((2, 3), dtype=np.float32),
        )
        with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "completed": True,
                    "dataset_digest": "ds",
                    "pose_convention": "stored_extrinsics_are_w2c",
                    "teacher": {"name": "t", "sha256": "s"},
                },
                stream,
            )
        with open(
            os.path.join(root, "flow_quality.json"), "w", encoding="utf-8"
        ) as stream:
            json.dump({"train": {}}, stream)
        return root

    def test_migrate_updates_digest_and_contract(self):
        root = self._cache()
        legacy = legacy_flow_digest(root)
        checkpoint_path = os.path.join(root, "last.pt")
        torch.save(
            {
                "schema_version": 1,
                "mode": "train_y",
                "flow_digest": legacy,
                "dataset_digest": "ds",
                "resolved_config": {"contract": {"flow_digest": legacy}},
            },
            checkpoint_path,
        )
        new_digest = migrate_checkpoint_flow_digest(checkpoint_path, root)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["flow_digest"], new_digest)
        self.assertEqual(
            payload["resolved_config"]["contract"]["flow_digest"], new_digest
        )
        self.assertNotEqual(new_digest, legacy)
        self.assertTrue(os.path.isfile(checkpoint_path + ".pre_migration"))

    def test_migrate_rejects_mismatched_digest(self):
        root = self._cache()
        checkpoint_path = os.path.join(root, "last.pt")
        torch.save(
            {
                "schema_version": 1,
                "mode": "train_y",
                "flow_digest": "wrong-digest",
                "dataset_digest": "ds",
            },
            checkpoint_path,
        )
        with self.assertRaisesRegex(ValueError, "refusing to migrate"):
            migrate_checkpoint_flow_digest(checkpoint_path, root)


class FlowOffsetModel(FakeAmuseModel):
    def forward(self, views):
        batch, _, height, width = views[0]["img"].shape
        parameter = self.dpt_feature_head.weight.mean()
        scale = torch.ones(batch, 1) * (1.0 + 0.0 * parameter)
        predictions = []
        for _ in views:
            rays = torch.zeros(batch, height, width, 3)
            rays[..., 2] = 1.0
            depth = torch.full((batch, height, width, 1), 0.5)
            predictions.append(
                {
                    "depth_along_ray": depth,
                    "ray_directions": rays,
                    "cam_trans": torch.zeros(batch, 3),
                    "cam_quats": torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(batch, 1),
                    "pts3d": depth * rays,
                    "scene_flow": torch.full((batch, height, width, 3), 0.01),
                    "metric_scaling_factor": scale,
                }
            )
        return predictions


class FlowAggregationTest(unittest.TestCase):
    def test_evaluate_flow_metric_not_double_counted(self):
        dataset = TinyGeometryDataset(length=2)
        delta = 0.01
        expected_sum = 0.0
        expected_count = 0
        for index in range(len(dataset)):
            for pair in dataset[index][1]["flow_pairs"]:
                error = (torch.full((4, 3), delta) - pair["flow_m"]).norm(dim=-1)
                keep = pair["motion_label"] != -1
                expected_sum += float(error[keep].sum())
                expected_count += int(keep.sum())
        result = evaluate_geometry(
            FlowOffsetModel(), dataset, "cpu", limit=None, use_amp=False
        )
        self.assertEqual(result["flow/count"], expected_count)
        self.assertAlmostEqual(
            float(result["flow/epe_m"]),
            expected_sum / expected_count,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
