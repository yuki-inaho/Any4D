"""Tests for the v2 factorized geometry + scene-flow objective."""

import unittest

import torch

from any4d.training.geometry_loss import (
    f_log_scalar,
    f_log_vector,
    geometry_loss,
    geometry_metrics,
    OBJECTIVE_ID,
)
from any4d.utils.geometry import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)

H, W, B, V = 6, 8, 1, 4
SCALE_EPS = 1e-6


def make_flow_pair(seed, target_view, count=6):
    generator = torch.Generator().manual_seed(seed + target_view)
    coordinates = torch.randperm(H * W, generator=generator)[:count]
    source_yx = torch.stack([coordinates // W, coordinates % W], dim=1)
    labels = torch.tensor([0, 0, 0, 1, 1, -1], dtype=torch.int8)[:count]
    flow = torch.zeros(count, 3)
    flow[labels == 0] = 0.005
    flow[labels == 1] = torch.randn(int((labels == 1).sum()), 3, generator=generator) * 0.02 + 0.1
    return {
        "source_yx": source_yx.long(),
        "flow_m": flow.float(),
        "motion_label": labels,
        "fb_error_px": torch.full((count,), 0.4),
        "frame_ids": torch.arange(4),
        "target_view": torch.tensor(target_view),
    }


def make_targets(seed=0):
    generator = torch.Generator().manual_seed(seed)
    depth = torch.rand(B, V, H, W, 1, generator=generator) * 1.9 + 0.1
    valid = torch.ones(B, V, H, W, dtype=torch.bool)
    holdout = torch.rand(B, V, H, W, generator=generator) < 0.5
    holdout[:, :, 0, 0] = True
    rays = torch.randn(B, V, H, W, 3, generator=generator)
    rays = rays / rays.norm(dim=-1, keepdim=True)
    rays[..., 2] = rays[..., 2].abs()
    rays = rays / rays.norm(dim=-1, keepdim=True)
    c2w = torch.eye(4).repeat(B, V, 1, 1)
    for view in range(1, V):
        quats = torch.randn(B, 4, generator=generator)
        quats = quats / quats.norm(dim=-1, keepdim=True)
        c2w[:, view, :3, :3] = quaternion_to_rotation_matrix(quats)
        c2w[:, view, :3, 3] = torch.randn(B, 3, generator=generator) * 0.3
    flow_pairs = [[make_flow_pair(seed, view) for view in (1, 2, 3)]]
    return {
        "depth_z_m": depth,
        "valid_mask": valid,
        "holdout_mask": holdout,
        "rays": rays,
        "c2w_ref": c2w,
        "frame_ids": torch.arange(V).unsqueeze(0).repeat(B, 1),
        "sequence_id": torch.zeros(B, dtype=torch.long),
        "flow_pairs": flow_pairs,
    }


def gt_pointmaps(targets):
    depth = targets["depth_z_m"][..., 0]
    rays = targets["rays"]
    r_z = rays[..., 2].clamp_min(1e-6)
    camera = depth[..., None] * rays / r_z[..., None]
    rotation = targets["c2w_ref"][..., :3, :3]
    translation = targets["c2w_ref"][..., :3, 3]
    world = torch.einsum("bvhwc,bvdc->bvhwd", camera, rotation)
    return world + translation[:, :, None, None, :]


def gt_scene_scale(targets):
    points = gt_pointmaps(targets)
    valid = targets["valid_mask"].float()
    return (points.norm(dim=-1) * valid).sum(dim=(1, 2, 3)) / valid.sum(dim=(1, 2, 3))


def make_predictions(targets, scale_multiplier=1.0, s_tensor=None):
    depth = targets["depth_z_m"][..., 0]
    rays = targets["rays"]
    r_z = rays[..., 2].clamp_min(1e-6)
    points = gt_pointmaps(targets)
    scale_gt = gt_scene_scale(targets)
    normalized_points = points / scale_gt[:, None, None, None, None]
    normalized_along = (depth / r_z) / scale_gt[:, None, None, None]
    normalized_trans = targets["c2w_ref"][..., :3, 3] / scale_gt[:, None, None]
    quats = rotation_matrix_to_quaternion(targets["c2w_ref"][..., :3, :3])
    if s_tensor is None:
        s_tensor = (scale_gt * scale_multiplier).view(B, 1)
    else:
        s_tensor = s_tensor.view(B, 1)
    predictions = []
    for view in range(V):
        normalized_flow = torch.zeros(B, H, W, 3)
        for pair in targets["flow_pairs"][0]:
            if int(pair["target_view"]) != view:
                continue
            source = pair["source_yx"]
            normalized_flow[:, source[:, 0], source[:, 1]] = (
                pair["flow_m"] / scale_gt.view(B, 1, 1)
            )
        predictions.append(
            {
                "depth_along_ray": normalized_along[:, view, ..., None]
                * s_tensor.view(B, 1, 1, 1),
                "ray_directions": rays[:, view],
                "cam_trans": normalized_trans[:, view] * s_tensor,
                "cam_quats": quats[:, view],
                "pts3d": normalized_points[:, view]
                * s_tensor.view(B, 1, 1, 1),
                "scene_flow": normalized_flow * s_tensor.view(B, 1, 1, 1),
                "metric_scaling_factor": s_tensor,
            }
        )
    return predictions


def mean_of(out, term):
    count = int(out[f"{term}_count"])
    return float(out[f"{term}_sum"]) / count if count else 0.0


class FactorizedLossTest(unittest.TestCase):
    def test_perfect_geometry_zero(self):
        targets = make_targets()
        out = geometry_loss(make_predictions(targets), targets)
        self.assertLess(float(out["loss"]), 1e-5)
        for term in ("ray", "quat", "trans", "depth", "point", "scale", "flow"):
            self.assertLess(abs(mean_of(out, term)), 1e-6, term)
        expected_flow = sum(
            int((pair["motion_label"] != -1).sum())
            for pair in targets["flow_pairs"][0]
        )
        self.assertEqual(int(out["flow_count"]), expected_flow)
        self.assertEqual(int(out["scale_count"]), B)
        self.assertEqual(int(out["quat_count"]), B * V)

    def test_shared_scene_scale(self):
        targets = make_targets()
        out_scaled = geometry_loss(make_predictions(targets), targets)
        self.assertLess(float(out_scaled["loss"]), 1e-5)
        wrong = geometry_loss(make_predictions(targets, scale_multiplier=1.5), targets)
        self.assertGreater(mean_of(wrong, "scale"), 1e-4)
        for term in ("ray", "quat", "trans", "depth", "point", "flow"):
            self.assertLess(abs(mean_of(wrong, term)), 1e-4, term)

    def test_pred0_anchor(self):
        targets = make_targets()
        predictions = make_predictions(targets)
        baseline = float(geometry_loss(predictions, targets)["loss"])
        predictions[0]["cam_trans"] = predictions[0]["cam_trans"].clone()
        predictions[0]["cam_trans"][0, 0] += 0.05
        perturbed = float(geometry_loss(predictions, targets)["loss"])
        self.assertGreater(perturbed, baseline + 1e-4)

    def _leaf(self, tensor):
        return tensor.detach().clone().requires_grad_(True)

    def _perturbed_predictions(self, targets, scale_factor=1.5, offset=0.003):
        scale_gt = gt_scene_scale(targets).detach()
        s = (scale_gt * scale_factor).clone().requires_grad_(True)
        predictions = make_predictions(targets, s_tensor=s)
        with torch.no_grad():
            for prediction in predictions:
                for key in (
                    "depth_along_ray",
                    "ray_directions",
                    "cam_trans",
                    "cam_quats",
                    "pts3d",
                    "scene_flow",
                ):
                    prediction[key] = prediction[key].detach().clone().requires_grad_(True)
                    prediction[key].add_(offset)
        return s, predictions

    def test_scale_gradient_isolation(self):
        targets = make_targets()
        s, predictions = self._perturbed_predictions(targets)
        out = geometry_loss(predictions, targets)
        out["scale_sum"].backward()
        self.assertIsNotNone(s.grad)
        self.assertGreater(float(s.grad.abs().sum()), 0.0)
        for key in ("depth_along_ray", "pts3d", "scene_flow", "cam_trans"):
            grad = predictions[0][key].grad
            if grad is not None:
                self.assertLess(float(grad.abs().sum()), 1e-4, key)

        s_shape, predictions = self._perturbed_predictions(targets)
        out = geometry_loss(predictions, targets)
        out["depth_sum"].backward()
        self.assertIsNotNone(s_shape.grad)
        self.assertLess(float(s_shape.grad.abs().sum()), 1e-4)

    def test_scene_flow_head_gradients(self):
        targets = make_targets()
        _, predictions = self._perturbed_predictions(targets, scale_factor=1.0)
        out = geometry_loss(predictions, targets)
        out["flow_sum"].backward()
        for view in (1, 2, 3):
            grad = predictions[view]["scene_flow"].grad
            self.assertIsNotNone(grad, view)
            self.assertTrue(torch.isfinite(grad).all(), view)
            self.assertGreater(float(grad.abs().sum()), 0.0, view)

    def test_log_vector_zero_limit(self):
        self.assertEqual(float(f_log_vector(torch.zeros(3)).norm()), 0.0)
        self.assertEqual(float(f_log_scalar(torch.tensor(0.0))), 0.0)
        small = torch.tensor([1e-8, 0.0, 0.0], requires_grad=True)
        target = torch.tensor([1e-3, 0.0, 0.0])
        error = (f_log_vector(small) - f_log_vector(target)).norm()
        error.backward()
        self.assertTrue(torch.isfinite(small.grad).all())
        self.assertGreater(float(small.grad.abs().sum()), 0.0)
        value = float(f_log_vector(torch.tensor([1e-6, 0.0, 0.0])).norm())
        self.assertGreater(value, 0.0)
        self.assertAlmostEqual(value, 1e-6, delta=1e-8)

    def test_ray_z_not_range(self):
        targets = make_targets()
        predictions = make_predictions(targets)
        for view in range(V):
            predictions[view]["depth_along_ray"] = targets["depth_z_m"][
                :, view, ..., :1
            ].clone()
        out = geometry_loss(predictions, targets)
        self.assertGreater(mean_of(out, "depth"), 1e-3)

    def test_xyzw_sign_invariance(self):
        targets = make_targets()
        predictions = make_predictions(targets)
        baseline = geometry_loss(predictions, targets)
        flipped = make_predictions(targets)
        for prediction in flipped:
            prediction["cam_quats"] = -prediction["cam_quats"]
        other = geometry_loss(flipped, targets)
        for term in ("ray", "quat", "trans", "depth", "point", "scale", "flow"):
            self.assertAlmostEqual(mean_of(baseline, term), mean_of(other, term), places=6)

    def test_invalid_targets_masked_before_arithmetic(self):
        targets = make_targets()
        predictions = make_predictions(targets)
        targets["depth_z_m"][targets["holdout_mask"] == 0] = 0.0
        mask = targets["valid_mask"] & targets["holdout_mask"]
        for view in range(V):
            free = (~mask[:, view]).nonzero()[0]
            predictions[view]["depth_along_ray"] = predictions[view][
                "depth_along_ray"
            ].clone()
            predictions[view]["depth_along_ray"][
                0, int(free[1]), int(free[2]), 0
            ] = float("nan")
        out = geometry_loss(predictions, targets)
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_empty_mask_fails(self):
        targets = make_targets()
        targets["holdout_mask"] = torch.zeros_like(targets["holdout_mask"])
        with self.assertRaises(ValueError):
            geometry_loss(make_predictions(targets), targets)

    def test_geometry_gradients(self):
        targets = make_targets()
        s, predictions = self._perturbed_predictions(targets, scale_factor=1.2)
        out = geometry_loss(predictions, targets)
        out["loss"].backward()
        for view, prediction in enumerate(predictions):
            for key in (
                "depth_along_ray",
                "ray_directions",
                "cam_trans",
                "cam_quats",
                "pts3d",
                "scene_flow",
            ):
                if key == "scene_flow" and view == 0:
                    continue
                grad = prediction[key].grad
                self.assertIsNotNone(grad, (view, key))
                self.assertTrue(torch.isfinite(grad).all(), (view, key))
        self.assertIsNotNone(s.grad)
        self.assertGreater(float(s.grad.abs().sum()), 0.0)
        self.assertGreater(float(predictions[0]["ray_directions"].grad.abs().sum()), 0.0)
        self.assertGreater(float(predictions[0]["pts3d"].grad.abs().sum()), 0.0)

    def test_empty_flow_pairs_graph_connected(self):
        targets = make_targets()
        empty = {
            "source_yx": torch.zeros(0, 2, dtype=torch.long),
            "flow_m": torch.zeros(0, 3),
            "motion_label": torch.zeros(0, dtype=torch.int8),
            "fb_error_px": torch.zeros(0),
            "frame_ids": torch.arange(4),
            "target_view": torch.tensor(0),
        }
        targets["flow_pairs"] = [[dict(empty) for _ in range(3)]]
        predictions = make_predictions(targets)
        for prediction in predictions:
            prediction["pts3d"] = self._leaf(prediction["pts3d"])
        out = geometry_loss(predictions, targets)
        self.assertEqual(int(out["flow_count"]), 0)
        self.assertTrue(bool(out["flow_empty"]))
        out["loss"].backward()
        grad = predictions[0]["pts3d"].grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())

    def test_metrics_and_flow_breakdown(self):
        targets = make_targets()
        metrics = geometry_metrics(make_predictions(targets), targets)
        self.assertAlmostEqual(float(metrics["heldout_depth/mae_m"]), 0.0, delta=1e-5)
        self.assertAlmostEqual(float(metrics["pseudo_pose/rotation_deg"]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["ray/angular_deg"]), 0.0, delta=0.2)
        labels = torch.cat(
            [pair["motion_label"] for pair in targets["flow_pairs"][0]]
        )
        self.assertEqual(
            int(metrics["flow/static_count"]), int((labels == 0).sum())
        )
        self.assertEqual(
            int(metrics["flow/dynamic_count"]), int((labels == 1).sum())
        )
        self.assertEqual(
            int(metrics["flow/ambiguous_count"]), int((labels == -1).sum())
        )
        self.assertEqual(int(metrics["flow/count"]), int((labels != -1).sum()))
        self.assertAlmostEqual(float(metrics["flow/epe_m"]), 0.0, places=6)

    def test_pose_metric_does_not_rebase_by_pred0(self):
        targets = make_targets()
        predictions = make_predictions(targets)
        offset = torch.tensor([0.5, 0.0, 0.0])
        for prediction in predictions:
            prediction["cam_trans"] = prediction["cam_trans"] + offset
        metrics = geometry_metrics(predictions, targets)
        self.assertAlmostEqual(
            float(metrics["pseudo_pose/translation_l2_m"]), 0.5, places=5
        )

    def test_pose_metric_includes_first_camera(self):
        targets = make_targets()
        predictions = make_predictions(targets)
        offset = torch.tensor([0.5, 0.0, 0.0])
        prediction = predictions[0]
        prediction["cam_trans"] = prediction["cam_trans"] + offset
        metrics = geometry_metrics(predictions, targets)
        self.assertAlmostEqual(
            float(metrics["pseudo_pose/translation_l2_m"]), 0.125, places=5
        )
        self.assertEqual(int(metrics["pseudo_pose/count"]), V)

    def test_metrics_exclude_ambiguous_flow(self):
        targets = make_targets()
        predictions = make_predictions(targets)
        for pair in targets["flow_pairs"][0]:
            mask = pair["motion_label"] == -1
            pair["flow_m"] = pair["flow_m"].clone()
            pair["flow_m"][mask] = 100.0
        metrics = geometry_metrics(predictions, targets)
        self.assertAlmostEqual(float(metrics["flow/epe_m"]), 0.0, places=5)
        labels = torch.cat(
            [pair["motion_label"] for pair in targets["flow_pairs"][0]]
        )
        self.assertEqual(int(metrics["flow/count"]), int((labels != -1).sum()))
        self.assertGreater(int(metrics["flow/ambiguous_count"]), 0)

    def test_pointmap_rotation_is_not_transposed(self):
        targets = make_targets()
        wrong = make_predictions(targets)
        depth = targets["depth_z_m"][..., 0]
        rays = targets["rays"]
        r_z = rays[..., 2].clamp_min(1e-6)
        camera = depth[..., None] * rays / r_z[..., None]
        rotation = targets["c2w_ref"][..., :3, :3]
        translation = targets["c2w_ref"][..., :3, 3]
        transposed = torch.einsum("bvhwc,bvcd->bvhwd", camera, rotation)
        transposed = transposed + translation[:, :, None, None, :]
        for view in range(V):
            wrong[view]["pts3d"] = transposed[:, view]
        out = geometry_loss(wrong, targets)
        self.assertGreater(mean_of(out, "point"), 1e-4)

    def test_objective_id_is_v2(self):
        self.assertEqual(OBJECTIVE_ID, "any4d_factorized_pseudo_sf_v2")


if __name__ == "__main__":
    unittest.main()
