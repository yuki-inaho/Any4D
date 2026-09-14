"""Runtime helpers for Any4D geometry fine-tuning.

Covers the head-only model initialization from the official checkpoint,
the AMUSE optimizer wiring, checkpoint save/resume/export and the shared
evaluation loop described in the geometry handoff workdoc.
"""

import hashlib
import json
import os
import random
import time

import numpy as np
import torch

from any4d.models import init_model
from any4d.training.geometry_data import (
    collate_geometry,
    compute_dataset_digest,
    GeometryDataset,
    PADDED_H,
    PADDED_W,
)
from any4d.training.geometry_loss import geometry_loss, geometry_metrics, OBJECTIVE_ID
from any4d.training.vendor.amuse import AMUSE
from any4d.utils.inference import (
    preprocess_input_views_for_inference,
    validate_input_views_for_inference,
)
from any4d.utils.rgbd import init_hydra_config

TRAINABLE_PREFIXES = (
    "dpt_feature_head.",
    "dpt_regressor_head.",
    "pose_head.",
    "scale_head.",
    "scene_flow_dpt_feature_head.",
    "scene_flow_dpt_regressor_head.",
)

AMUSE_OUTPUT_PREFIXES = (
    "dpt_regressor_head.conv2.2.",
    "scene_flow_dpt_regressor_head.conv2.2.",
    "pose_head.fc_t.",
    "pose_head.fc_rot.",
    "scale_head.output_proj.",
)

VENDORED_AMUSE_SHA256 = (
    "84fd3fbbc99e1718cf1c821ceff3369439f48e6fbd8ecc3a2b83afa5d82eea1f"
)

# Known artifacts of the training snapshot that produced the official
# checkpoint: an unused event encoder and scene-flow encoder positional
# embeddings removed from the released model code/config. Anything else
# unexpected is a revision mismatch and must stop the load.
ALLOWED_UNEXPECTED_KEY_PREFIXES = (
    "event_encoder.",
    "scene_flow_encoder.pos_embed",
    "scene_flow_encoder.post_pe_norm",
)

FORBIDDEN_MODEL_INPUT_KEYS = {
    "camera_poses",
    "camera_pose_quats",
    "camera_pose_trans",
    "c2w_ref",
    "depth_z_m",
    "valid_mask",
    "holdout_mask",
}


def set_trainable_geometry_heads(model, prefixes=TRAINABLE_PREFIXES):
    """Freeze everything, then re-enable only the allowed geometry heads."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    parameter_names = dict(model.named_parameters())
    trainable = []
    for prefix in prefixes:
        matched = [name for name in parameter_names if name.startswith(prefix)]
        if not matched:
            raise ValueError(f"trainable prefix matched no parameter: {prefix}")
        trainable.extend(matched)
    trainable = sorted(set(trainable))
    if not trainable:
        raise ValueError("no trainable parameters matched the allowlist")

    for name in trainable:
        parameter_names[name].requires_grad_(True)

    model.eval()
    tops = tuple(prefix.rstrip(".") for prefix in prefixes)
    for name, module in model.named_modules():
        if name == "":
            continue
        if name in tops or any(name.startswith(f"{top}.") for top in tops):
            module.train()
    return trainable


def load_geometry_state_dict(model, state_dict):
    """Load the official checkpoint with an explicit unexpected-key allowlist."""
    result = model.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    allowed = [
        key
        for key in unexpected
        if any(key.startswith(prefix) for prefix in ALLOWED_UNEXPECTED_KEY_PREFIXES)
    ]
    unknown = [key for key in unexpected if key not in allowed]
    if missing:
        raise RuntimeError(
            f"checkpoint is missing {len(missing)} keys, first: {missing[:5]}"
        )
    if unknown:
        raise RuntimeError(
            f"checkpoint has {len(unknown)} unexpected keys outside the known "
            f"revision allowlist, first: {unknown[:5]}"
        )
    return {"missing": missing, "allowed_unexpected": allowed}


def prepare_model_inputs(views):
    """Validate and preprocess views without leaking teacher targets."""
    validate_input_views_for_inference(views)
    processed = preprocess_input_views_for_inference(views)
    for index, view in enumerate(processed):
        forbidden = sorted(set(view) & FORBIDDEN_MODEL_INPUT_KEYS)
        if forbidden:
            raise ValueError(
                f"view {index} carries teacher-only keys into the model input: {forbidden}"
            )
    return processed


def parameter_hash(model, names=None):
    """Deterministic digest of the selected parameters' values and shapes."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if names is not None and name not in names:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_base_state_dict(base_checkpoint):
    if not os.path.isfile(base_checkpoint):
        raise FileNotFoundError(f"missing base checkpoint: {base_checkpoint}")
    checkpoint = torch.load(
        base_checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    if "model" not in checkpoint:
        raise ValueError(f"checkpoint has no 'model' entry: {base_checkpoint}")
    return checkpoint["model"]


def build_geometry_model(
    config_dir,
    base_checkpoint,
    device="cpu",
    machine="local",
    task="rgbd",
):
    """Build Any4D from the repo config and load the official geometry heads."""
    config = init_hydra_config(
        config_dir,
        "train",
        overrides=[
            f"machine={machine}",
            "model=any4d",
            "model.encoder.uses_torch_hub=false",
            f"model/task={task}",
        ],
    )
    model = init_model(config.model.model_str, config.model.model_config)
    report = load_geometry_state_dict(model, load_base_state_dict(base_checkpoint))
    trainable = set_trainable_geometry_heads(model)
    model.to(device)
    return model, report, trainable


def verify_vendored_amuse():
    """Verify that the vendored AMUSE source matches the pinned revision."""
    path = os.path.join(os.path.dirname(__file__), "vendor", "amuse.py")
    with open(path, "rb") as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    if digest != VENDORED_AMUSE_SHA256:
        raise RuntimeError(
            f"vendored AMUSE source SHA256 mismatch: {digest} != {VENDORED_AMUSE_SHA256}"
        )
    return digest


def build_amuse(
    model,
    trainable_names,
    warmup_steps,
    muon_lr=1e-4,
    muon_weight_decay=0.01,
    muon_momentum=0.95,
    aux_lr=1e-5,
    aux_weight_decay=0.0,
    aux_beta2=0.999,
    aux_eps=1e-10,
    beta1=0.9,
    rho=1.0,
    r=0.0,
    weight_lr_power=2.0,
):
    """Build the pinned AMUSE optimizer over the allowlisted geometry heads.

    Returns ``(optimizer, manifest)`` where the manifest records the final
    post-construction parameter order, which AMUSE may reorder inside the
    Muon group.
    """
    verify_vendored_amuse()
    if int(warmup_steps) <= 0:
        raise ValueError("AMUSE requires warmup_steps > 0")

    parameters = dict(model.named_parameters())
    muon = []
    aux = []
    seen = set()
    for name in trainable_names:
        if name in seen:
            raise ValueError(f"duplicate trainable parameter name: {name}")
        seen.add(name)
        if name not in parameters:
            raise ValueError(f"trainable parameter not found in model: {name}")
        parameter = parameters[name]
        if not parameter.requires_grad:
            raise ValueError(f"trainable parameter is frozen: {name}")
        if name.startswith(AMUSE_OUTPUT_PREFIXES) or parameter.ndim == 1:
            aux.append(parameter)
        elif parameter.ndim in (2, 4):
            muon.append(parameter)
        else:
            raise ValueError(
                f"unexpected parameter rank {parameter.ndim} for {name}"
            )
    if not muon or not aux:
        raise ValueError("both AMUSE parameter groups must be non-empty")
    if len(muon) + len(aux) != len(trainable_names):
        raise ValueError("AMUSE group coverage mismatch")

    param_groups = [
        {
            "params": muon,
            "use_muon": True,
            "lr": muon_lr,
            "weight_decay": muon_weight_decay,
            "momentum": muon_momentum,
            "aux_update_type": "adamw",
        },
        {
            "params": aux,
            "use_muon": False,
            "update_type": "adamw",
            "lr": aux_lr,
            "weight_decay": aux_weight_decay,
            "beta2": aux_beta2,
            "eps": aux_eps,
        },
    ]
    optimizer = AMUSE(
        param_groups,
        beta1=beta1,
        weight_lr_power=weight_lr_power,
        warmup_steps=int(warmup_steps),
        rho=rho,
        r=r,
    )
    name_by_id = {id(parameters[name]): name for name in trainable_names}
    manifest = {"warmup_steps": int(warmup_steps), "groups": []}
    for index, group in enumerate(optimizer.param_groups):
        entry = {
            "index": index,
            "update_type": group["update_type"],
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "params": [],
        }
        if group["use_muon"]:
            entry["momentum"] = group["momentum"]
            entry["aux_update_type"] = group["aux_update_type"]
        else:
            entry["beta2"] = group["beta2"]
            entry["eps"] = group["eps"]
        for parameter in group["params"]:
            name = name_by_id[id(parameter)]
            entry["params"].append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "numel": parameter.numel(),
                }
            )
        manifest["groups"].append(entry)
    return optimizer, manifest


SCHEMA_VERSION = 1
CONTRACT_KEYS = (
    "base_sha256",
    "dataset_digest",
    "flow_digest",
    "objective_id",
    "split",
    "mask_scheme",
    "resolution",
    "views",
    "precision",
    "max_updates",
    "epochs",
)


def ensure_new_directory(path):
    """Create a new run directory, refusing to reuse an existing path."""
    if os.path.exists(path):
        raise FileExistsError(f"output path already exists: {path}")
    os.makedirs(path)
    return path


def _atomic_torch_save(payload, path):
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"output directory does not exist: {directory}")
    temporary = f"{path}.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _manifest_names(manifest):
    return [
        entry["name"]
        for group in manifest["groups"]
        for entry in group["params"]
    ]


def collect_head_delta(model, trainable_names):
    """Snapshot the trainable parameters (and head buffers) by state-dict key."""
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    trainable_ids = {id(parameters[name]) for name in trainable_names}
    tops = tuple(sorted({name.split(".")[0] for name in trainable_names}))
    delta = {}
    buffer_prefixes = tuple(f"{top}." for top in tops)
    for key, tensor in model.state_dict(keep_vars=True).items():
        if key in parameters and id(parameters[key]) in trainable_ids or key in buffers and key.startswith(buffer_prefixes):
            delta[key] = tensor.detach().clone()
    covered = {
        id(parameters[key])
        for key in delta
        if key in parameters
    }
    missing = [name for name in trainable_names if id(parameters[name]) not in covered]
    if missing:
        raise RuntimeError(f"head delta misses trainable parameters: {missing[:5]}")
    return delta


def apply_head_delta(model, delta):
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for key, value in delta.items():
        target = parameters.get(key, buffers.get(key))
        if target is None:
            raise ValueError(f"head delta key not found in model: {key}")
        if tuple(target.shape) != tuple(value.shape):
            raise ValueError(
                f"head delta shape mismatch for {key}: "
                f"{tuple(target.shape)} != {tuple(value.shape)}"
            )
        with torch.no_grad():
            target.copy_(value)


def _amuse_constructor_state(optimizer):
    return {
        "beta1_init": optimizer.beta1_init,
        "rho": optimizer.rho,
        "r": optimizer.r,
        "weight_lr_power": optimizer.weight_lr_power,
        "warmup_steps": optimizer.warmup_steps,
        "weight_decay_at_y": optimizer.weight_decay_at_y,
    }


def _restore_amuse_constructor_state(optimizer, state):
    for key, value in state.items():
        setattr(optimizer, key, value)


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _validate_contract(stored, current):
    mismatches = [
        key
        for key in CONTRACT_KEYS
        if stored.get(key) != current.get(key)
    ]
    if mismatches:
        raise ValueError(
            "resume contract mismatch for keys: "
            + ", ".join(
                f"{key} ({stored.get(key)!r} != {current.get(key)!r})"
                for key in mismatches
            )
        )


def save_last_checkpoint(
    path,
    *,
    model,
    trainable_names,
    optimizer,
    contract,
    manifest,
    resolved_config,
    global_step,
    epoch,
    cursor,
    shuffle_permutation,
    best_metric,
):
    """Save the resumable y-state checkpoint atomically."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "train_y",
        "head_delta": collect_head_delta(model, trainable_names),
        "amuse_state": optimizer.state_dict(),
        "amuse_constructor": _amuse_constructor_state(optimizer),
        "train_mode": bool(optimizer.train_mode),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "cursor": int(cursor),
        "shuffle_permutation": list(shuffle_permutation),
        "rng": _capture_rng_state(),
        "group_manifest": manifest,
        "resolved_config": resolved_config,
        "best_metric": best_metric,
    }
    payload.update({key: contract[key] for key in CONTRACT_KEYS if key in contract})
    _atomic_torch_save(payload, path)
    return payload


def save_best_checkpoint(
    path,
    *,
    model,
    trainable_names,
    contract,
    manifest,
    resolved_config,
    metric,
    global_step,
):
    """Save the evaluation x-state checkpoint (call while optimizer is in eval)."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "eval_x",
        "head_delta": collect_head_delta(model, trainable_names),
        "group_manifest": manifest,
        "resolved_config": resolved_config,
        "best_metric": metric,
        "global_step": int(global_step),
    }
    payload.update({key: contract[key] for key in CONTRACT_KEYS if key in contract})
    _atomic_torch_save(payload, path)
    return payload


def load_resume_checkpoint(
    path,
    *,
    model,
    optimizer,
    manifest,
    contract,
    trainable_names,
):
    """Restore a train_y checkpoint after validating the run contract."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {payload.get('schema_version')}")
    if payload.get("mode") != "train_y":
        raise ValueError(f"resume requires a train_y checkpoint, got {payload.get('mode')!r}")
    _validate_contract(payload, contract)

    current_names = set(_manifest_names(manifest))
    stored_names = set(_manifest_names(payload["group_manifest"]))
    if _manifest_names(payload["group_manifest"]) != _manifest_names(manifest):
        raise ValueError("resume group manifest order mismatch")
    if stored_names != current_names:
        raise ValueError("resume group manifest names mismatch")
    if set(trainable_names) != current_names:
        raise ValueError("resume trainable parameter set mismatch")

    apply_head_delta(model, payload["head_delta"])
    _restore_amuse_constructor_state(optimizer, payload["amuse_constructor"])
    optimizer.load_state_dict(payload["amuse_state"])
    optimizer.train_mode = bool(payload["train_mode"])
    _restore_rng_state(payload["rng"])
    return {
        "global_step": payload["global_step"],
        "epoch": payload["epoch"],
        "cursor": payload["cursor"],
        "shuffle_permutation": payload["shuffle_permutation"],
        "best_metric": payload["best_metric"],
        "train_mode": payload["train_mode"],
    }


def export_inference_checkpoint(
    base_checkpoint,
    checkpoint_path,
    output_path,
    model_factory,
):
    """Merge base + x delta into a standalone inference state dict."""
    if os.path.exists(output_path):
        raise FileExistsError(f"export output already exists: {output_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("export requires a v2 checkpoint schema")
    if payload.get("mode") != "eval_x":
        raise ValueError(
            f"export requires an eval_x checkpoint, got {payload.get('mode')!r}"
        )
    base_sha = file_sha256(base_checkpoint)
    if payload.get("base_sha256") != base_sha:
        raise ValueError(
            "checkpoint base_sha256 is missing or does not match the supplied "
            "base checkpoint"
        )
    if payload.get("objective_id") != OBJECTIVE_ID:
        raise ValueError(
            "checkpoint objective_id is missing or mismatched: "
            f"{payload.get('objective_id')!r}"
        )
    model = model_factory()
    load_geometry_state_dict(model, load_base_state_dict(base_checkpoint))
    apply_head_delta(model, payload["head_delta"])
    full_state = model.state_dict()

    verifier = model_factory()
    verifier.load_state_dict(full_state, strict=True)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_checkpoint": os.path.basename(checkpoint_path),
        "best_metric": payload.get("best_metric"),
        "global_step": payload.get("global_step"),
        "base_sha256": payload.get("base_sha256"),
    }
    _atomic_torch_save({"model": full_state, "geometry_finetune": metadata}, output_path)
    return metadata


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_config(config):
    required = (
        "data_root",
        "base_checkpoint",
        "output",
        "train_split",
        "eval_split",
        "epochs",
        "warmup_steps",
        "height",
        "width",
        "views",
        "batch_size",
        "precision",
        "optimizer",
        "seed",
        "flow_root",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"training config misses keys: {missing}")
    if (config["height"], config["width"]) != (480, 640):
        raise ValueError(
            "geometry training is fixed to 480x640; refusing "
            f"{config['height']}x{config['width']}"
        )
    if config["views"] != 4:
        raise ValueError(f"geometry training is fixed to 4 views, got {config['views']}")
    if config["batch_size"] != 1:
        raise ValueError(f"geometry training is fixed to batch size 1, got {config['batch_size']}")
    if config["precision"] != "bf16":
        raise ValueError(f"geometry training requires bf16 autocast, got {config['precision']}")
    if config["optimizer"] != "amuse":
        raise ValueError(f"geometry training requires the AMUSE optimizer, got {config['optimizer']}")
    if int(config["warmup_steps"]) <= 0:
        raise ValueError("warmup_steps must be positive")
    if int(config["epochs"]) < 1:
        raise ValueError("epochs must be at least 1")
    for key in ("max_updates", "eval_limit"):
        value = config.get(key)
        if value is not None and int(value) < 1:
            raise ValueError(f"{key} must be positive when provided")


def set_head_train_mode(model):
    tops = tuple(prefix.rstrip(".") for prefix in TRAINABLE_PREFIXES)
    for name, module in model.named_modules():
        if name and (name in tops or any(name.startswith(f"{top}.") for top in tops)):
            module.train()


def _move_batch(batch_views, targets, device):
    views = [
        {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in view.items()
        }
        for view in batch_views
    ]
    moved_targets = {}
    for key, value in targets.items():
        if key == "flow_pairs":
            moved_targets[key] = [
                [
                    {
                        pair_key: pair_value.to(device)
                        if torch.is_tensor(pair_value)
                        else pair_value
                        for pair_key, pair_value in pair.items()
                    }
                    for pair in sample
                ]
                for sample in value
            ]
        else:
            moved_targets[key] = value.to(device) if torch.is_tensor(value) else value
    return views, moved_targets


def _forward_predictions(model, views, use_amp):
    if use_amp:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(views)
    return model(views)


def flow_cache_digest(flow_root):
    """Content digest over every file of the pseudo-flow cache.

    Covers the manifest, the quality report and all sparse NPZ teacher
    files so that changed teacher values invalidate resume comparisons.
    """
    root = os.path.abspath(flow_root)
    entries = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            path = os.path.join(directory, filename)
            if not path.endswith(".npz") and filename not in (
                "manifest.json",
                "flow_quality.json",
            ):
                continue
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            with open(path, "rb") as stream:
                file_digest = hashlib.sha256(stream.read()).hexdigest()
            entries.append((relative, file_digest))
    if not entries:
        raise FileNotFoundError(f"flow cache is empty: {flow_root}")
    entries.sort()
    digest = hashlib.sha256()
    for relative, file_digest in entries:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def evaluate_geometry(model, dataset, device, limit=None, use_amp=False):
    """Deterministic v2 evaluation over a dataset with dataset-level sums."""
    model.eval()
    terms = ("ray", "quat", "trans", "depth", "point", "scale", "flow")
    totals = {f"{term}_sum": 0.0 for term in terms}
    totals.update({f"{term}_count": 0 for term in terms})
    totals.update(
        {
            "mae_sum": 0.0,
            "rmse_sq_sum": 0.0,
            "absrel_sum": 0.0,
            "valid_count": 0,
            "trans_l2_sum": 0.0,
            "rot_deg_sum": 0.0,
            "pose_count": 0,
            "ray_angle_sum": 0.0,
            "ray_angle_count": 0,
            "flow_epe_sum": 0.0,
            "flow_epe_count": 0,
            "flow_static_sum": 0.0,
            "flow_static_count": 0,
            "flow_dynamic_sum": 0.0,
            "flow_dynamic_count": 0,
        }
    )
    count = len(dataset) if limit is None else min(int(limit), len(dataset))
    with torch.no_grad():
        for index in range(count):
            views, targets = collate_geometry([dataset[index]])
            views, targets = _move_batch(views, targets, device)
            processed = prepare_model_inputs(views)
            predictions = _forward_predictions(model, processed, use_amp)
            loss_out = geometry_loss(predictions, targets)
            for term in terms:
                totals[f"{term}_sum"] += float(loss_out[f"{term}_sum"].detach())
                totals[f"{term}_count"] += int(loss_out[f"{term}_count"])
            metrics = geometry_metrics(predictions, targets)
            totals["mae_sum"] += metrics["heldout_depth/mae_sum"]
            totals["rmse_sq_sum"] += metrics["heldout_depth/rmse_sq_sum"]
            totals["absrel_sum"] += metrics["heldout_depth/absrel_sum"]
            totals["valid_count"] += metrics["heldout_depth/valid_count"]
            totals["trans_l2_sum"] += metrics["pseudo_pose/translation_l2_sum"]
            totals["rot_deg_sum"] += metrics["pseudo_pose/rotation_deg_sum"]
            totals["pose_count"] += metrics["pseudo_pose/count"]
            totals["ray_angle_sum"] += metrics["ray/angular_sum"]
            totals["ray_angle_count"] += metrics["ray/count"]
            totals["flow_epe_sum"] += metrics["flow/epe_sum"]
            totals["flow_epe_count"] += metrics["flow/count"]
            totals["flow_static_sum"] += metrics["flow/static_epe_sum"]
            totals["flow_static_count"] += metrics["flow/static_count"]
            totals["flow_dynamic_sum"] += metrics["flow/dynamic_epe_sum"]
            totals["flow_dynamic_count"] += metrics["flow/dynamic_count"]

    if totals["flow_count"] == 0 or totals["flow_epe_count"] == 0:
        raise ValueError("evaluation has no pseudo-flow targets")
    for term in terms:
        if totals[f"{term}_count"] == 0:
            raise ValueError(f"evaluation has an empty {term} loss set")
    objective = sum(
        totals[f"{term}_sum"] / totals[f"{term}_count"] for term in terms
    )
    result = {
        "val/objective": objective,
        "heldout_depth/mae_m": totals["mae_sum"] / totals["depth_count"],
        "heldout_depth/rmse_m": (totals["rmse_sq_sum"] / totals["depth_count"]) ** 0.5,
        "heldout_depth/absrel": totals["absrel_sum"] / totals["depth_count"],
        "heldout_depth/count": totals["depth_count"],
        "heldout_depth/valid_count": totals["valid_count"],
        "heldout_depth/coverage": totals["depth_count"] / totals["valid_count"],
        "pseudo_pose/translation_l2_m": totals["trans_l2_sum"] / totals["pose_count"],
        "pseudo_pose/rotation_deg": totals["rot_deg_sum"] / totals["pose_count"],
        "pseudo_pose/count": totals["pose_count"],
        "ray/angular_deg": totals["ray_angle_sum"] / totals["ray_angle_count"],
        "ray/count": totals["ray_angle_count"],
        "flow/epe_m": totals["flow_epe_sum"] / totals["flow_epe_count"],
        "flow/count": totals["flow_epe_count"],
        "flow/static_epe_m": (
            totals["flow_static_sum"] / totals["flow_static_count"]
            if totals["flow_static_count"]
            else None
        ),
        "flow/static_count": totals["flow_static_count"],
        "flow/dynamic_epe_m": (
            totals["flow_dynamic_sum"] / totals["flow_dynamic_count"]
            if totals["flow_dynamic_count"]
            else None
        ),
        "flow/dynamic_count": totals["flow_dynamic_count"],
        "objective_id": OBJECTIVE_ID,
        "loss_sums": {key: value for key, value in totals.items()},
    }
    return result


def _epoch_permutation(seed, epoch, length):
    generator = torch.Generator()
    generator.manual_seed(int(seed) * 100003 + int(epoch))
    return torch.randperm(length, generator=generator).tolist()


def run_geometry_training(
    config,
    model=None,
    train_dataset=None,
    eval_dataset=None,
    device=None,
    dataset_digest=None,
    base_sha256=None,
):
    """Head-only geometry training loop with AMUSE and resumable checkpoints."""
    validate_training_config(config)
    output = config["output"]
    resume = config.get("resume")
    if resume is None and os.path.exists(output):
        raise FileExistsError(f"output run directory already exists: {output}")
    if resume is not None and not os.path.isfile(resume):
        raise FileNotFoundError(f"resume checkpoint not found: {resume}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "CUDA device does not support bf16; refusing to switch precision"
        )
    use_amp = device == "cuda"

    if base_sha256 is None:
        base_sha256 = file_sha256(config["base_checkpoint"])
    if dataset_digest is None:
        dataset_digest = compute_dataset_digest(config["data_root"])

    mask_scheme = f"torch-block16-p0.2-seed{config['seed']}-val42"
    if train_dataset is None:
        train_dataset = GeometryDataset(
            config["data_root"],
            config["train_split"],
            seed=config["seed"],
            epoch=0,
            flow_root=config["flow_root"],
            dataset_digest=dataset_digest,
        )
    if eval_dataset is None:
        eval_dataset = GeometryDataset(
            config["data_root"],
            config["eval_split"],
            seed=42,
            epoch=0,
            flow_root=config["flow_root"],
            dataset_digest=dataset_digest,
        )
    flow_digest = config.get("flow_digest")
    if flow_digest is None:
        flow_digest = flow_cache_digest(config["flow_root"])
    contract = {
        "base_sha256": base_sha256,
        "dataset_digest": dataset_digest,
        "flow_digest": flow_digest,
        "objective_id": OBJECTIVE_ID,
        "split": config["train_split"],
        "mask_scheme": mask_scheme,
        "resolution": [config["height"], config["width"]],
        "views": config["views"],
        "precision": config["precision"],
        "max_updates": config.get("max_updates"),
        "epochs": config["epochs"],
    }
    resolved_config = {
        "config": config,
        "contract": contract,
        "padded_resolution": [PADDED_H, PADDED_W],
        "device": device,
    }
    for split_name, dataset in (("train", train_dataset), ("eval", eval_dataset)):
        stats = getattr(dataset, "flow_stats", None)
        if not stats or stats["dynamic_points"] == 0:
            raise RuntimeError(
                f"{split_name} split has no usable pseudo-dynamic flow targets"
            )

    if model is None:
        model, _, trainable_names = build_geometry_model(
            config["config_dir"], config["base_checkpoint"], device=device
        )
    else:
        trainable_names = set_trainable_geometry_heads(model)
        model.to(device)
    optimizer, manifest = build_amuse(
        model, trainable_names, warmup_steps=config["warmup_steps"]
    )
    set_head_train_mode(model)
    named_parameters = dict(model.named_parameters())
    trainable_parameters = [named_parameters[name] for name in trainable_names]
    trainable_set = set(trainable_names)
    frozen_names = {
        name for name in named_parameters if name not in trainable_set
    }
    frozen_hash_start = parameter_hash(model, names=frozen_names)
    trainable_hash_start = parameter_hash(model, names=trainable_set)

    os.makedirs(output, exist_ok=resume is not None)
    metrics_path = os.path.join(output, "metrics.jsonl")
    with open(os.path.join(output, "resolved_config.json"), "w", encoding="utf-8") as stream:
        json.dump(resolved_config, stream, indent=2)
    with open(os.path.join(output, "trainable_manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(
            {"manifest": manifest, "trainable_count": len(trainable_names)},
            stream,
            indent=2,
        )

    start_epoch = 0
    cursor = 0
    global_step = 0
    best_metric = None
    initial_metric = None
    last_metrics = None
    peak_allocated = 0
    peak_reserved = 0
    stop = False

    if resume is not None:
        metadata = load_resume_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            manifest=manifest,
            contract=contract,
            trainable_names=trainable_names,
        )
        start_epoch = metadata["epoch"]
        cursor = metadata["cursor"]
        global_step = metadata["global_step"]
        best_metric = metadata["best_metric"]

    def snapshot_memory():
        nonlocal peak_allocated, peak_reserved
        if device == "cuda":
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())

    def append_metrics(row):
        with open(metrics_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")

    def evaluate_and_save(epoch, step, epoch_cursor, permutation):
        nonlocal best_metric, last_metrics
        optimizer.eval()
        model.eval()
        metrics = evaluate_geometry(
            model, eval_dataset, device, limit=config.get("eval_limit"), use_amp=use_amp
        )
        improved = best_metric is None or metrics["val/objective"] < best_metric
        if improved:
            best_metric = metrics["val/objective"]
            save_best_checkpoint(
                os.path.join(output, "best.pt"),
                model=model,
                trainable_names=trainable_names,
                contract=contract,
                manifest=manifest,
                resolved_config=resolved_config,
                metric=best_metric,
                global_step=step,
            )
        optimizer.train()
        save_last_checkpoint(
            os.path.join(output, "last.pt"),
            model=model,
            trainable_names=trainable_names,
            optimizer=optimizer,
            contract=contract,
            manifest=manifest,
            resolved_config=resolved_config,
            global_step=step,
            epoch=epoch,
            cursor=epoch_cursor,
            shuffle_permutation=permutation,
            best_metric=best_metric,
        )
        set_head_train_mode(model)
        last_metrics = metrics
        append_metrics(
            {
                "event": "eval",
                "epoch": epoch,
                "global_step": step,
                "val/objective": metrics["val/objective"],
                "heldout_depth/mae_m": metrics["heldout_depth/mae_m"],
                "heldout_depth/rmse_m": metrics["heldout_depth/rmse_m"],
                "heldout_depth/absrel": metrics["heldout_depth/absrel"],
                "pseudo_pose/translation_l2_m": metrics["pseudo_pose/translation_l2_m"],
                "pseudo_pose/rotation_deg": metrics["pseudo_pose/rotation_deg"],
                "ray/angular_deg": metrics["ray/angular_deg"],
                "flow/epe_m": metrics["flow/epe_m"],
                "flow/static_epe_m": metrics["flow/static_epe_m"],
                "flow/dynamic_epe_m": metrics["flow/dynamic_epe_m"],
                "flow/count": metrics["flow/count"],
                "best_metric": best_metric,
                "improved": bool(improved),
            }
        )
        return metrics

    if resume is None:
        initial = evaluate_and_save(0, 0, 0, [])
        initial_metric = initial["val/objective"]
    else:
        initial_metric = best_metric

    for epoch in range(start_epoch, int(config["epochs"])):
        if stop:
            break
        if hasattr(train_dataset, "epoch"):
            train_dataset.epoch = epoch
        permutation = _epoch_permutation(config["seed"], epoch, len(train_dataset))
        start_position = cursor if epoch == start_epoch else 0
        optimizer.train()
        set_head_train_mode(model)
        for position in range(start_position, len(permutation)):
            if config.get("max_updates") is not None and global_step >= int(config["max_updates"]):
                stop = True
                break
            index = permutation[position]
            batch_views, targets = collate_geometry([train_dataset[index]])
            views, targets = _move_batch(batch_views, targets, device)
            processed = prepare_model_inputs(views)
            optimizer.zero_grad(set_to_none=True)
            started = time.time()
            predictions = _forward_predictions(model, processed, use_amp)
            loss_out = geometry_loss(predictions, targets)
            loss = loss_out["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at step {global_step} (sequence {int(targets['sequence_id'])})"
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient norm at step {global_step}")
            optimizer.step()
            global_step += 1
            cursor = position + 1
            snapshot_memory()
            append_metrics(
                {
                    "event": "step",
                    "global_step": global_step,
                    "epoch": epoch,
                    "sequence_id": int(targets["sequence_id"]),
                    "loss": float(loss.detach()),
                    "ray_sum": float(loss_out["ray_sum"].detach()),
                    "ray_count": int(loss_out["ray_count"]),
                    "quat_sum": float(loss_out["quat_sum"].detach()),
                    "quat_count": int(loss_out["quat_count"]),
                    "trans_sum": float(loss_out["trans_sum"].detach()),
                    "trans_count": int(loss_out["trans_count"]),
                    "depth_sum": float(loss_out["depth_sum"].detach()),
                    "depth_count": int(loss_out["depth_count"]),
                    "point_sum": float(loss_out["point_sum"].detach()),
                    "point_count": int(loss_out["point_count"]),
                    "scale_sum": float(loss_out["scale_sum"].detach()),
                    "scale_count": int(loss_out["scale_count"]),
                    "flow_sum": float(loss_out["flow_sum"].detach()),
                    "flow_count": int(loss_out["flow_count"]),
                    "valid_depth_count": int(targets["valid_mask"].sum()),
                    "grad_norm": float(grad_norm),
                    "elapsed_s": time.time() - started,
                    "allocated_mib": torch.cuda.max_memory_allocated() / 2**20
                    if device == "cuda"
                    else 0.0,
                    "reserved_mib": torch.cuda.max_memory_reserved() / 2**20
                    if device == "cuda"
                    else 0.0,
                    "height": int(targets["depth_z_m"].shape[-3]),
                    "width": int(targets["depth_z_m"].shape[-2]),
                    "views": int(targets["depth_z_m"].shape[1]),
                    "batch": int(targets["depth_z_m"].shape[0]),
                }
            )
        evaluate_and_save(epoch, global_step, cursor, permutation)
        cursor = 0
        if stop:
            break

    summary = {
        "initial/objective": initial_metric,
        "val/objective": None if last_metrics is None else last_metrics["val/objective"],
        "best/objective": best_metric,
        "improved": bool(
            initial_metric is not None
            and best_metric is not None
            and best_metric < initial_metric
        ),
        "global_step": global_step,
        "epochs": int(config["epochs"]),
        "base_sha256": base_sha256,
        "dataset_digest": dataset_digest,
        "mask_scheme": mask_scheme,
        "trainable_count": len(trainable_names),
        "frozen_parameter_hash_start": frozen_hash_start,
        "frozen_parameter_hash_end": parameter_hash(model, names=frozen_names),
        "trainable_parameter_hash_start": trainable_hash_start,
        "trainable_parameter_hash_end": parameter_hash(model, names=trainable_set),
        "peak_allocated_mib": peak_allocated / 2**20 if device == "cuda" else 0.0,
        "peak_reserved_mib": peak_reserved / 2**20 if device == "cuda" else 0.0,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %Z%z"),
    }
    summary["frozen_parameters_unchanged"] = (
        summary["frozen_parameter_hash_start"] == summary["frozen_parameter_hash_end"]
    )
    summary["trainable_parameters_updated"] = (
        summary["trainable_parameter_hash_start"]
        != summary["trainable_parameter_hash_end"]
    )
    with open(os.path.join(output, "summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return summary
