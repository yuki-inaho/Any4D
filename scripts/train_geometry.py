#!/usr/bin/env python
"""Single CLI for head-only geometry fine-tuning of Any4D.

Subcommands:
    check-data  Read-only audit of the staged RGB-D dataset contract.
    train       Head-only geometry training with AMUSE (resumable).
    eval        Compare base and best checkpoints on the fixed holdout.
    export      Merge base + best into a standalone inference checkpoint.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cv2
import numpy as np
import torch

from any4d.training.geometry_data import (
    compute_dataset_digest,
    GeometryDataset,
    MAX_DEPTH_MM,
)
from any4d.training.geometry_loss import OBJECTIVE_ID
from any4d.training.geometry_runtime import (
    apply_head_delta,
    build_geometry_model,
    evaluate_geometry,
    export_inference_checkpoint,
    file_sha256,
    run_geometry_training,
    SCHEMA_VERSION,
    validate_training_config,
)

DEFAULT_CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
EXPECTED_FRAMES = 1607
EXPECTED_SEQUENCES = 1408
EXPECTED_FIXED4 = {"train": 879, "val": 163, "smoke": 51}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="train_geometry.py",
        description="Head-only Any4D geometry fine-tuning on the RGB-D handoff data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-data", help="audit the dataset contract")
    check.add_argument("--data-root", required=True)

    prepare = subparsers.add_parser(
        "prepare-flow", help="generate the pseudo scene-flow cache"
    )
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--output-root", "--output", dest="output_root", required=True)
    prepare.add_argument("--teacher", default="raft_large_C_T_SKHT_V2")
    prepare.add_argument("--allow-weight-download", action="store_true")
    prepare.add_argument("--splits", default="train,val,smoke")
    prepare.add_argument("--max-pairs", type=int, default=None)
    prepare.add_argument("--resume", action="store_true")

    migrate = subparsers.add_parser(
        "migrate-flow-digest",
        help="migrate a stored legacy flow_digest to the content digest",
    )
    migrate.add_argument("--checkpoint", required=True)
    migrate.add_argument("--flow-root", required=True)
    migrate.add_argument("--output", default=None)

    train = subparsers.add_parser("train", help="run head-only geometry training")
    train.add_argument("--data-root", required=True)
    train.add_argument("--base-checkpoint", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--flow-root", required=True)
    train.add_argument("--objective", default="any4d_factorized_pseudo_sf_v2")
    train.add_argument("--train-split", default="train")
    train.add_argument("--eval-split", default="val")
    train.add_argument("--max-updates", type=int, default=None)
    train.add_argument("--eval-limit", type=int, default=None)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--warmup-steps", type=int, default=100)
    train.add_argument("--height", type=int, default=480)
    train.add_argument("--width", type=int, default=640)
    train.add_argument("--views", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--precision", default="bf16")
    train.add_argument("--optimizer", default="amuse")
    train.add_argument("--muon-lr", type=float, default=1e-4)
    train.add_argument("--aux-lr", type=float, default=1e-5)
    train.add_argument("--unfreeze-blocks", type=int, default=0)
    train.add_argument("--unfrozen-muon-lr", type=float, default=None)
    train.add_argument("--unfrozen-aux-lr", type=float, default=None)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--resume", default=None)
    train.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)

    evaluate = subparsers.add_parser("eval", help="evaluate base vs best")
    evaluate.add_argument("--data-root", required=True)
    evaluate.add_argument("--base-checkpoint", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--flow-root", required=True)
    evaluate.add_argument("--split", default="val")
    evaluate.add_argument("--compare-base", action="store_true")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)

    export = subparsers.add_parser("export", help="export an inference checkpoint")
    export.add_argument("--base-checkpoint", required=True)
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    return parser


def args_to_config(args):
    if args.objective != OBJECTIVE_ID:
        raise ValueError(
            f"unsupported objective {args.objective!r}; expected {OBJECTIVE_ID!r}"
        )
    return {
        "data_root": args.data_root,
        "base_checkpoint": args.base_checkpoint,
        "output": args.output,
        "flow_root": args.flow_root,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "max_updates": args.max_updates,
        "eval_limit": args.eval_limit,
        "epochs": args.epochs,
        "warmup_steps": args.warmup_steps,
        "height": args.height,
        "width": args.width,
        "views": args.views,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "optimizer": args.optimizer,
        "muon_lr": args.muon_lr,
        "aux_lr": args.aux_lr,
        "unfreeze_blocks": args.unfreeze_blocks,
        "unfrozen_muon_lr": args.unfrozen_muon_lr,
        "unfrozen_aux_lr": args.unfrozen_aux_lr,
        "seed": args.seed,
        "resume": args.resume,
        "config_dir": args.config_dir,
    }


def run_migrate_flow_digest(config):
    from any4d.training.geometry_runtime import migrate_checkpoint_flow_digest

    digest = migrate_checkpoint_flow_digest(
        config["checkpoint"], config["flow_root"], config["output"]
    )
    return {"flow_digest": digest, "checkpoint": config["output"] or config["checkpoint"]}


def run_check_data(data_root):
    """Read-only audit; prints a JSON report and returns it."""
    scene_dir = os.path.join(data_root, "scenes", "scene_000000")
    cameras = np.load(os.path.join(scene_dir, "cameras.npz"), allow_pickle=False)
    sequences = np.load(os.path.join(scene_dir, "sequences.npz"), allow_pickle=False)

    frames = len(cameras["frame_ids"])
    rgb_files = len(os.listdir(os.path.join(scene_dir, "rgb")))
    depth_files = len(os.listdir(os.path.join(scene_dir, "depth")))
    lengths = sequences["lengths"]
    fixed4 = {
        split: int(((sequences["split_ids"] == split_id) & (lengths == 4)).sum())
        for split, split_id in (("train", 0), ("val", 1), ("smoke", 2))
    }
    frame_sets = {}
    chunk_crossings = 0
    for split in EXPECTED_FIXED4:
        dataset = GeometryDataset(data_root, split, seed=42, epoch=0)
        ids = set()
        for record in dataset.sequence_records:
            for frame_id, row in zip(record["frame_ids"], record["frame_rows"]):
                ids.add(frame_id)
                if int(dataset.frame_chunks[row]) != int(record["chunk_id"]):
                    chunk_crossings += 1
        frame_sets[split] = ids
    overlaps = {
        f"{a}-{b}": len(frame_sets[a] & frame_sets[b])
        for a, b in (("train", "val"), ("train", "smoke"), ("val", "smoke"))
    }

    intrinsics = cameras["intrinsics"]
    extrinsics = cameras["extrinsics_w2c"]
    rotations = extrinsics[:, :3, :3]
    orthonormal = np.einsum("bij,bkj->bik", rotations, rotations)
    k_last_row_ok = bool(
        np.allclose(intrinsics[:, 2, :], np.array([0.0, 0.0, 1.0], dtype=np.float32))
    )
    sample_ids = np.linspace(0, frames - 1, 8).astype(int).tolist()
    depth_checks = []
    for frame_id in sample_ids:
        image = cv2.imread(
            os.path.join(scene_dir, "depth", f"frame_{frame_id:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        valid = (image > 0) & (image <= MAX_DEPTH_MM)
        depth_checks.append(
            {
                "frame_id": int(frame_id),
                "dtype": str(image.dtype),
                "shape": list(image.shape),
                "max_mm": int(image.max()),
                "valid_fraction": float(valid.mean()),
            }
        )
    val_dataset = GeometryDataset(data_root, "val", seed=42, epoch=0)
    val_holdout_min = None
    for index in range(len(val_dataset)):
        _, targets = val_dataset[index]
        count = int(targets["holdout_mask"].sum())
        val_holdout_min = count if val_holdout_min is None else min(val_holdout_min, count)

    checks = {
        "frames_is_expected": frames == EXPECTED_FRAMES,
        "sequences_is_expected": len(sequences["sequences"]) == EXPECTED_SEQUENCES,
        "rgb_depth_file_counts": rgb_files == frames and depth_files == frames,
        "fixed4_counts": fixed4 == EXPECTED_FIXED4,
        "split_frame_overlap": overlaps,
        "split_frame_overlap_zero": all(value == 0 for value in overlaps.values()),
        "chunk_crossings": chunk_crossings,
        "intrinsics_finite": bool(np.isfinite(intrinsics).all()),
        "intrinsics_max_abs_diff": float(np.abs(intrinsics - intrinsics[0]).max()),
        "intrinsics_last_row_ok": k_last_row_ok,
        "extrinsics_finite": bool(np.isfinite(extrinsics).all()),
        "rotation_orthonormal_max_err": float(np.abs(orthonormal - np.eye(3)).max()),
        "quality_flags_all_true": bool(cameras["quality_flags"].all()),
        "depth_samples": depth_checks,
        "val_holdout_min_valid_count": val_holdout_min,
        "val_holdout_all_positive": bool(val_holdout_min is not None and val_holdout_min > 0),
        "dataset_digest": compute_dataset_digest(data_root),
    }
    boolean_keys = (
        "frames_is_expected",
        "sequences_is_expected",
        "rgb_depth_file_counts",
        "fixed4_counts",
        "split_frame_overlap_zero",
        "intrinsics_finite",
        "intrinsics_last_row_ok",
        "extrinsics_finite",
        "quality_flags_all_true",
        "val_holdout_all_positive",
    )
    report = {
        "command": "check-data",
        "data_root_basename": os.path.basename(os.path.normpath(data_root)),
        "counts": {
            "frames": frames,
            "sequences": len(sequences["sequences"]),
            "rgb_files": rgb_files,
            "depth_files": depth_files,
            "fixed4": fixed4,
        },
        "checks": checks,
        "passed": all(bool(checks[key]) for key in boolean_keys)
        and chunk_crossings == 0
        and checks["rotation_orthonormal_max_err"] < 1e-5
        and checks["intrinsics_max_abs_diff"] < 1e-5,
    }
    print(json.dumps(report, indent=2))
    return report


def run_train(config):
    validate_training_config(config)
    return run_geometry_training(config)


def run_prepare_flow(config):
    from any4d.training.flow_teacher import generate_flow_cache, TEACHER_NAME

    if config["teacher"] != TEACHER_NAME:
        raise ValueError(
            f"unsupported teacher {config['teacher']!r}; expected {TEACHER_NAME!r}"
        )
    splits = tuple(
        split.strip() for split in config["splits"].split(",") if split.strip()
    )
    if not splits:
        raise ValueError("--splits must select at least one split")
    return generate_flow_cache(
        config["data_root"],
        config["output_root"],
        splits=splits,
        resume=config["resume"],
        max_pairs=config["max_pairs"],
        allow_weight_download=config["allow_weight_download"],
    )


def run_eval(config, model=None, dataset=None, device=None, base_sha256=None, dataset_digest=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        config["checkpoint"], map_location="cpu", weights_only=False
    )
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema for eval")
    if checkpoint.get("mode") != "eval_x":
        raise ValueError(f"eval requires an eval_x checkpoint, got {checkpoint.get('mode')!r}")
    if base_sha256 is None:
        base_sha256 = file_sha256(config["base_checkpoint"])
    if checkpoint.get("base_sha256") not in (None, base_sha256):
        raise ValueError("checkpoint base_sha256 does not match --base-checkpoint")
    if dataset_digest is None:
        dataset_digest = compute_dataset_digest(config["data_root"])
    stored_digest = checkpoint.get("dataset_digest")
    if stored_digest is not None and stored_digest != dataset_digest:
        raise ValueError("checkpoint dataset_digest does not match --data-root")
    if dataset is None:
        dataset = GeometryDataset(
            config["data_root"],
            config["split"],
            seed=42,
            epoch=0,
            flow_root=config["flow_root"],
            dataset_digest=dataset_digest,
        )

    if model is None:
        model, _, _ = build_geometry_model(
            config["config_dir"], config["base_checkpoint"], device=device
        )
    use_amp = device == "cuda"
    result = {
        "command": "eval",
        "split": config["split"],
        "samples": len(dataset),
        "base_sha256": base_sha256,
        "dataset_digest": dataset_digest,
        "saved_best_metric": checkpoint.get("best_metric"),
    }
    if config.get("compare_base"):
        result["base"] = evaluate_geometry(model, dataset, device, use_amp=use_amp)
    apply_head_delta(model, checkpoint["head_delta"])
    result["best"] = evaluate_geometry(model, dataset, device, use_amp=use_amp)
    saved = checkpoint.get("best_metric")
    if saved is not None:
        tolerance = 1e-5 + 1e-4 * abs(saved)
        if abs(result["best"]["val/objective"] - saved) > tolerance:
            raise RuntimeError(
                "best checkpoint metric reproduction failed: "
                f"{result['best']['val/objective']} vs saved {saved}"
            )
        result["metric_reproduced"] = True
    if result.get("base") is not None:
        result["improved"] = bool(
            result["best"]["val/objective"] < result["base"]["val/objective"]
        )
    with open(config["output"], "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "loss_sums"}, indent=2))
    return result


def run_export(config):
    def model_factory():
        model, _, _ = build_geometry_model(
            config["config_dir"], config["base_checkpoint"], device="cpu"
        )
        return model

    metadata = export_inference_checkpoint(
        config["base_checkpoint"],
        config["checkpoint"],
        config["output"],
        model_factory,
    )
    print(json.dumps(metadata, indent=2))
    return metadata


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "check-data":
            return 0 if run_check_data(args.data_root)["passed"] else 1
        if args.command == "prepare-flow":
            manifest = run_prepare_flow(
                {
                    "data_root": args.data_root,
                    "output_root": args.output_root,
                    "teacher": args.teacher,
                    "allow_weight_download": args.allow_weight_download,
                    "splits": args.splits,
                    "max_pairs": args.max_pairs,
                    "resume": args.resume,
                }
            )
            print(json.dumps(manifest, indent=2))
            return 0
        if args.command == "train":
            run_train(args_to_config(args))
            return 0
        if args.command == "migrate-flow-digest":
            report = run_migrate_flow_digest(
                {
                    "checkpoint": args.checkpoint,
                    "flow_root": args.flow_root,
                    "output": args.output,
                }
            )
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "eval":
            run_eval(
                {
                    "data_root": args.data_root,
                    "base_checkpoint": args.base_checkpoint,
                    "checkpoint": args.checkpoint,
                    "flow_root": args.flow_root,
                    "split": args.split,
                    "compare_base": args.compare_base,
                    "output": args.output,
                    "config_dir": args.config_dir,
                }
            )
            return 0
        if args.command == "export":
            run_export(
                {
                    "base_checkpoint": args.base_checkpoint,
                    "checkpoint": args.checkpoint,
                    "output": args.output,
                    "config_dir": args.config_dir,
                }
            )
            return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary converts any failure to exit 1
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
