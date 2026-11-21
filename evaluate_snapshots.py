"""Evaluate FoldedACT snapshots and log test metrics by training step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from policy.evaluate_act import BatchNormalizer, evaluate_policy_on_dataset, save_json
from policy.splits_and_configs import DEFAULT_DATASET_ROOT, SPLITS
from policy.stea_act_dataset import DEFAULT_CACHE_DIR, STEAACTDataset
from policy2.folded_act import FoldedACTPolicy


def snapshot_checkpoints(run_dir: Path) -> list[Path]:
    checkpoints = run_dir / "checkpoints"
    candidates = [*sorted(checkpoints.glob("step_*")), checkpoints / "last"]
    return [path for path in candidates if (path / "checkpoint_meta.json").is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--wandb-project", default="stea-act")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    args = parser.parse_args()

    checkpoints = snapshot_checkpoints(args.run_dir)
    if not checkpoints:
        parser.error(f"No completed numbered or last checkpoints in {args.run_dir}")

    config = json.loads((args.run_dir / "run_config.json").read_text())
    camera_keys = tuple(config["camera_keys"])
    dataset = STEAACTDataset(
        SPLITS["test"], camera_keys, chunk_size=config["chunk_size"],
        dof_mode=config["dof_mode"], gripper_mode=config["gripper_mode"],
        gripper_shift=config["gripper_shift"], gripper_open_threshold=config["gripper_open_threshold"],
        dataset_root=args.dataset_root, cache_dir=args.cache_dir,
    )
    coefficient = config["temporal_ensemble_coeff"]
    max_history = config["temporal_ensemble_max_history"]
    torch.set_float32_matmul_precision("high")

    from contextlib import nullcontext

    tracking = nullcontext()
    if args.wandb_mode != "disabled":
        import wandb

        tracking = wandb.init(
            project=args.wandb_project, name=f"{args.run_dir.name}_snapshot_test",
            group="foldedact_snapshot_test", job_type="snapshot_test", dir=str(args.run_dir),
            config=config, mode=args.wandb_mode,
        )

    with tracking as wandb_run:
        for checkpoint in checkpoints:
            step = json.loads((checkpoint / "checkpoint_meta.json").read_text())["step"]
            policy = FoldedACTPolicy.from_pretrained(str(checkpoint)).to(args.device)
            stats = torch.load(checkpoint / "train_stats.pt", map_location="cpu", weights_only=True)
            metrics = evaluate_policy_on_dataset(
                policy, dataset, BatchNormalizer(stats, camera_keys, args.device),
                batch_size=config["batch_size"], record_attention=False,
                temporal_ensemble_coeff=coefficient, temporal_ensemble_max_history=max_history,
            )
            del policy
            save_json({
                "split": "test", "checkpoint_step": step, "coefficient": coefficient,
                "max_history": max_history, "metrics": metrics,
            }, args.run_dir / "snapshot_test" / f"step_{step:06d}.json")
            closing = metrics["by_phase"]["closing_transition"]
            if wandb_run is not None:
                wandb_run.log({
                    "test/closing_te_pos_l1_mm": closing["pos_l1_mm"],
                    "test/closing_step0_pos_l1_mm": closing["step0_metrics"]["pos_l1_mm"],
                    "test/closing_te_pos_l2_mm": closing["pos_l2_mm"],
                    "test/closing_gripper_acc_pct": closing["gripper_acc_pct"],
                    "test/overall_te_pos_l2_mm": metrics["temporal_ensemble_metrics"]["pos_l2_mm"],
                }, step=step)
            print(f"Step {step}: test Closing Pos L1 = {closing['pos_l1_mm']:.2f} mm", flush=True)


if __name__ == "__main__":
    main()