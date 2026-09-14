"""Factorized geometry + scene-flow objective (v2).

Implements section 5.3 of the geometry handoff workdoc: ray, quaternion,
translation, ray-depth, pointmap, metric-scale and pseudo scene-flow terms
over a shared per-clip metric scale. This is an explicit implementation of
the paper's factored representation, not a reproduction of the unreleased
training loss.
"""

import math

import torch

from any4d.utils.geometry import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)

OBJECTIVE_ID = "any4d_factorized_pseudo_sf_v2"
STATIC_FLOW_WEIGHT = 1.0
DYNAMIC_FLOW_WEIGHT = 10.0
SCALE_FLOOR = 1e-6
EPS = 1e-6


def f_log_scalar(x):
    """log(1 + x) for non-negative scalars."""
    return torch.log1p(x)


def f_log_vector(v, small=1e-4):
    """v * log(1 + ||v||) / ||v|| with a stable small-norm expansion."""
    norm = v.norm(dim=-1, keepdim=True)
    safe = norm.clamp_min(small)
    coefficient = torch.where(
        norm < small,
        1.0 - norm / 2.0 + norm * norm / 3.0,
        torch.log1p(safe) / safe,
    )
    return v * coefficient


def _check_finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"non-finite {name}")


def _pose_matrices(predictions):
    matrices = []
    for view, prediction in enumerate(predictions):
        quats = prediction["cam_quats"]
        trans = prediction["cam_trans"]
        rotation = quaternion_to_rotation_matrix(quats)
        matrix = torch.zeros(
            quats.shape[0], 4, 4, dtype=rotation.dtype, device=rotation.device
        )
        matrix[:, :3, :3] = rotation
        matrix[:, :3, 3] = trans
        matrix[:, 3, 3] = 1.0
        matrices.append(matrix)
    return torch.stack(matrices, dim=1)


def _relative_poses(predictions):
    matrices = _pose_matrices(predictions)
    return torch.linalg.inv(matrices[:, :1]) @ matrices


def geometry_loss(predictions, targets, eps=EPS):
    """Return the v2 objective plus per-term sums and counts."""
    valid = targets["valid_mask"]
    holdout = targets["holdout_mask"]
    rays_gt = targets["rays"]
    c2w = targets["c2w_ref"]
    batch, views, height, width = valid.shape
    if len(predictions) != views:
        raise ValueError(f"expected {views} view predictions, got {len(predictions)}")

    scale = predictions[0]["metric_scaling_factor"]
    for view, prediction in enumerate(predictions):
        if not torch.equal(prediction["metric_scaling_factor"], scale):
            raise ValueError(f"metric_scaling_factor differs at view {view}")
        for key in ("depth_along_ray", "ray_directions", "cam_trans", "cam_quats", "pts3d", "scene_flow"):
            if key not in prediction:
                raise KeyError(f"prediction view {view} misses '{key}'")
    if scale.ndim != 2 or scale.shape != (batch, 1):
        raise ValueError(f"metric_scaling_factor must be (B,1), got {tuple(scale.shape)}")
    _check_finite(scale, "metric scaling factor")
    if (scale <= 0).any():
        raise RuntimeError("non-positive metric scaling factor")
    scale_image = scale.view(batch, 1, 1, 1)
    scale_vec = scale.view(batch, 1)

    z_gt = targets["depth_z_m"][..., 0]
    r_z = rays_gt[..., 2].clamp_min(eps)
    depth_gt = z_gt / r_z
    camera_points = z_gt[..., None] * rays_gt / r_z[..., None]
    rotation_gt = c2w[..., :3, :3]
    translation_gt = c2w[..., :3, 3]
    points_gt = torch.einsum(
        "bvhwc,bvdc->bvhwd", camera_points, rotation_gt
    ) + translation_gt[:, :, None, None, :]

    valid_float = valid.float()
    valid_count = valid_float.sum(dim=(1, 2, 3))
    if (valid_count == 0).any():
        raise RuntimeError("empty GT valid set for the shared scene scale")

    predictions_scaled = []
    for view, prediction in enumerate(predictions):
        depth_along = prediction["depth_along_ray"][:, :height, :width] / scale_image
        rays = prediction["ray_directions"][:, :height, :width]
        rays = rays / (rays.norm(dim=-1, keepdim=True) + eps)
        points = prediction["pts3d"][:, :height, :width] / scale_image
        predictions_scaled.append(
            {
                "depth": depth_along,
                "rays": rays,
                "points": points,
                "trans": prediction["cam_trans"] / scale_vec,
                "quats": prediction["cam_quats"],
                "flow": prediction["scene_flow"][:, :height, :width] / scale_image,
            }
        )

    z_scale = (points_gt.norm(dim=-1) * valid_float).sum(dim=(1, 2, 3)) / valid_count
    z_hat = (
        torch.stack(
            [entry["points"] for entry in predictions_scaled], dim=1
        ).norm(dim=-1)
        * valid_float
    ).sum(dim=(1, 2, 3)) / valid_count
    _check_finite(z_scale, "GT scene scale")
    _check_finite(z_hat, "predicted scene scale")
    if (z_scale <= SCALE_FLOOR).any() or (z_hat <= SCALE_FLOOR).any():
        raise RuntimeError("scene scale collapsed below the floor")

    sums = {
        key: torch.zeros((), dtype=torch.float64, device=valid.device)
        for key in ("ray", "quat", "trans", "depth", "point", "scale", "flow")
    }
    counts = {key: 0 for key in sums}
    holdout_mask = valid & holdout
    if not holdout_mask.any():
        raise ValueError("empty holdout mask")

    for view in range(views):
        entry = predictions_scaled[view]
        ray_error = (entry["rays"] - rays_gt[:, view]).norm(dim=-1)
        sums["ray"] = sums["ray"] + ray_error.sum().double()
        counts["ray"] += ray_error.numel()

        quats_gt = rotation_matrix_to_quaternion(rotation_gt[:, view])
        delta = entry["quats"] - quats_gt
        quat_error = torch.minimum(delta.norm(dim=-1), (entry["quats"] + quats_gt).norm(dim=-1))
        sums["quat"] = sums["quat"] + quat_error.sum().double()
        counts["quat"] += entry["quats"].shape[0]

        trans_error = (
            entry["trans"] / z_hat.unsqueeze(-1)
            - translation_gt[:, view] / z_scale.unsqueeze(-1)
        ).norm(dim=-1)
        sums["trans"] = sums["trans"] + trans_error.sum().double()
        counts["trans"] += trans_error.shape[0]

        mask = holdout_mask[:, view]
        count = int(mask.sum())
        if count == 0:
            continue
        depth_pred = entry["depth"][..., 0][mask] / z_hat.unsqueeze(-1).expand(batch, height, width)[mask]
        depth_ref = depth_gt[:, view][mask] / z_scale.unsqueeze(-1).expand(batch, height, width)[mask]
        _check_finite(depth_pred, f"scaled predicted ray depth (view {view})")
        if (depth_pred <= 0).any() or (depth_ref <= 0).any():
            raise RuntimeError("non-positive ray depth inside the holdout mask")
        depth_error = (f_log_scalar(depth_pred) - f_log_scalar(depth_ref)).abs()
        sums["depth"] = sums["depth"] + depth_error.sum().double()
        counts["depth"] += count

        points_pred = entry["points"][mask] / z_hat.unsqueeze(-1).expand(batch, height, width, 3)[mask]
        points_ref = points_gt[:, view][mask] / z_scale.unsqueeze(-1).expand(batch, height, width, 3)[mask]
        point_error = (
            f_log_vector(points_pred) - f_log_vector(points_ref)
        ).norm(dim=-1)
        sums["point"] = sums["point"] + point_error.sum().double()
        counts["point"] += count

    scale_error = (
        f_log_scalar(scale_vec.squeeze(-1) * z_hat.detach())
        - f_log_scalar(z_scale)
    ).abs()
    sums["scale"] = sums["scale"] + scale_error.sum().double()
    counts["scale"] += batch

    flow_empty = False
    empty_pairs = 0
    flow_pairs = targets.get("flow_pairs")
    if flow_pairs is None:
        raise KeyError("targets miss 'flow_pairs'")
    for batch_index in range(batch):
        for pair in flow_pairs[batch_index]:
            source = pair["source_yx"].long()
            count = int(source.shape[0])
            if count == 0:
                empty_pairs += 1
                continue
            view = int(pair["target_view"])
            if not 1 <= view < views:
                raise ValueError(f"target_view {view} outside 1..{views - 1}")
            labels = pair["motion_label"]
            keep = labels != -1
            kept = int(keep.sum())
            if kept == 0:
                empty_pairs += 1
                continue
            gathered = predictions_scaled[view]["flow"][
                batch_index, source[:, 0], source[:, 1]
            ]
            flow_gt = pair["flow_m"]
            error = (
                f_log_vector(gathered[keep] / z_hat[batch_index])
                - f_log_vector(flow_gt[keep] / z_scale[batch_index])
            ).norm(dim=-1)
            weights = torch.where(
                labels[keep] == 1,
                torch.full_like(error, DYNAMIC_FLOW_WEIGHT),
                torch.full_like(error, STATIC_FLOW_WEIGHT),
            )
            sums["flow"] = sums["flow"] + (weights * error).sum().double()
            counts["flow"] += kept

    if counts["flow"] == 0:
        flow_empty = True
        graph_zero = (
            predictions[0]["pts3d"].sum() + predictions[0]["depth_along_ray"].sum()
        ) * 0.0
        flow_mean = graph_zero.double()
    else:
        flow_mean = sums["flow"] / counts["flow"]

    for term, count in counts.items():
        if count == 0 and term != "flow":
            raise RuntimeError(f"empty {term} loss set")

    loss = (
        sums["ray"] / counts["ray"]
        + sums["quat"] / counts["quat"]
        + sums["trans"] / counts["trans"]
        + sums["depth"] / counts["depth"]
        + sums["point"] / counts["point"]
        + sums["scale"] / counts["scale"]
        + flow_mean
    )
    _check_finite(loss, "objective")
    return {
        "loss": loss,
        "objective_id": OBJECTIVE_ID,
        **{f"{term}_sum": sums[term] for term in sums},
        **{f"{term}_count": counts[term] for term in sums},
        "flow_empty": flow_empty,
        "flow_empty_pairs": empty_pairs,
        "shared_scale_gt": z_scale.detach(),
        "shared_scale_pred": z_hat.detach(),
    }


def geometry_metrics(predictions, targets):
    """Held-out geometry, pose, ray and pseudo-flow metrics."""
    valid = targets["valid_mask"]
    holdout = targets["holdout_mask"]
    rays_gt = targets["rays"]
    c2w = targets["c2w_ref"]
    batch, views, height, width = valid.shape

    z_gt = targets["depth_z_m"][..., 0]
    depth_abs_sum = 0.0
    depth_sq_sum = 0.0
    depth_rel_sum = 0.0
    depth_count = 0
    valid_count = int(valid.sum())
    ray_angle_sum = 0.0
    ray_count = 0
    for view in range(views):
        prediction = predictions[view]
        rays = prediction["ray_directions"][:, :height, :width]
        rays = rays / (rays.norm(dim=-1, keepdim=True) + EPS)
        depth_pred = (
            prediction["depth_along_ray"][:, :height, :width, 0]
            * rays[..., 2]
        )
        mask = valid[:, view] & holdout[:, view]
        count = int(mask.sum())
        if count == 0:
            raise ValueError(f"empty holdout mask at view {view}")
        target_depth = z_gt[:, view][mask].double()
        predicted_depth = depth_pred[mask].double()
        error = (predicted_depth - target_depth).abs()
        depth_abs_sum += float(error.sum())
        depth_sq_sum += float((error**2).sum())
        depth_rel_sum += float((error / target_depth).sum())
        depth_count += count

        cosine = (rays * rays_gt[:, view]).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.acos(cosine) * (180.0 / math.pi)
        ray_angle_sum += float(angle.sum())
        ray_count += angle.numel()

    predicted_poses = _pose_matrices(predictions)
    trans_l2_sum = 0.0
    rot_deg_sum = 0.0
    pose_count = 0
    for view in range(views):
        trans_error = predicted_poses[:, view, :3, 3] - c2w[:, view, :3, 3]
        trans_l2_sum += float(trans_error.norm(dim=-1).sum())
        rotation_pred = predicted_poses[:, view, :3, :3]
        rotation_target = c2w[:, view, :3, :3]
        trace = (rotation_pred.transpose(1, 2) @ rotation_target).diagonal(
            dim1=-2, dim2=-1
        ).sum(dim=-1)
        cosine = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
        rot_deg_sum += float((torch.acos(cosine) * (180.0 / math.pi)).sum())
        pose_count += batch

    flow = {
        "flow/epe_sum": 0.0,
        "flow/count": 0,
        "flow/static_epe_sum": 0.0,
        "flow/static_count": 0,
        "flow/dynamic_epe_sum": 0.0,
        "flow/dynamic_count": 0,
        "flow/ambiguous_count": 0,
    }
    for batch_index in range(batch):
        for pair in targets["flow_pairs"][batch_index]:
            source = pair["source_yx"].long()
            count = int(source.shape[0])
            if count == 0:
                continue
            view = int(pair["target_view"])
            predicted = predictions[view]["scene_flow"][
                batch_index, source[:, 0], source[:, 1]
            ]
            error = (predicted - pair["flow_m"]).norm(dim=-1)
            labels = pair["motion_label"]
            keep = labels != -1
            flow["flow/epe_sum"] += float(error[keep].sum())
            flow["flow/count"] += int(keep.sum())
            static = labels == 0
            dynamic = labels == 1
            flow["flow/static_epe_sum"] += float(error[static].sum())
            flow["flow/static_count"] += int(static.sum())
            flow["flow/dynamic_epe_sum"] += float(error[dynamic].sum())
            flow["flow/dynamic_count"] += int(dynamic.sum())
            flow["flow/ambiguous_count"] += int((labels == -1).sum())

    mae = depth_abs_sum / depth_count
    metrics = {
        "heldout_depth/mae_m": mae,
        "heldout_depth/mae_sum": depth_abs_sum,
        "heldout_depth/rmse_m": math.sqrt(depth_sq_sum / depth_count),
        "heldout_depth/rmse_sq_sum": depth_sq_sum,
        "heldout_depth/absrel": depth_rel_sum / depth_count,
        "heldout_depth/absrel_sum": depth_rel_sum,
        "heldout_depth/count": depth_count,
        "heldout_depth/valid_count": valid_count,
        "heldout_depth/coverage": depth_count / valid_count,
        "pseudo_pose/translation_l2_m": trans_l2_sum / pose_count,
        "pseudo_pose/translation_l2_sum": trans_l2_sum,
        "pseudo_pose/rotation_deg": rot_deg_sum / pose_count,
        "pseudo_pose/rotation_deg_sum": rot_deg_sum,
        "pseudo_pose/count": pose_count,
        "ray/angular_deg": ray_angle_sum / ray_count,
        "ray/angular_sum": ray_angle_sum,
        "ray/count": ray_count,
        **flow,
    }
    if flow["flow/count"] == 0:
        raise ValueError("evaluation has no pseudo-flow targets")
    metrics["flow/epe_m"] = metrics["flow/epe_sum"] / metrics["flow/count"]
    metrics["flow/static_epe_m"] = (
        metrics["flow/static_epe_sum"] / metrics["flow/static_count"]
        if metrics["flow/static_count"]
        else None
    )
    metrics["flow/dynamic_epe_m"] = (
        metrics["flow/dynamic_epe_sum"] / metrics["flow/dynamic_count"]
        if metrics["flow/dynamic_count"]
        else None
    )
    return metrics
