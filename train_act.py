"""Train and evaluate a single ACT policy under a configurable camera-view subset on STEA."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

# Ensure repo root is importable when executed as `python policy/train_act.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

from policy.evaluate_act import BatchNormalizer, evaluate_policy_on_dataset, save_json
from policy.splits_and_configs import (
    CAMERA_PRESETS,
    DEFAULT_DATASET_ROOT,
    SPLITS,
    validate_splits,
)
from policy.stea_act_dataset import (
    DEFAULT_CACHE_DIR,
    STEAACTDataset,
    compute_train_stats,
    load_reconstructed_trajectories,
)
from policy2.folded_act import FoldedACT, FoldedACTConfig, FoldedACTPolicy


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_act_config(
    camera_keys: tuple[str, ...],
    image_shapes: dict[str, tuple[int, int, int]],
    state_dim: int,
    action_dim: int,
    *,
    chunk_size: int = 50,
    lr: float = 5e-5,
    lr_backbone: float = 1e-5,
    weight_decay: float = 1e-4,
    kl_weight: float = 10.0,
    dropout: float = 0.1,
    device: str = "cuda",
    spatial_folding: bool = True,
    swin_after_tokenization: bool = True,
    camera_overlay: bool = False,
    early_camera_overlay: bool = True,
    sep_swin_for_views: bool | None = None,
) -> FoldedACTConfig:
    input_features: dict[str, PolicyFeature] = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    for ck in camera_keys:
        input_features[ck] = PolicyFeature(type=FeatureType.VISUAL, shape=image_shapes[ck])

    output_features: dict[str, PolicyFeature] = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }

    return FoldedACTConfig(
        device=device,
        spatial_folding=spatial_folding,
        swin_after_tokenization=swin_after_tokenization,
        camera_overlay=camera_overlay,
        early_camera_overlay=early_camera_overlay,
        sep_swin_for_views=swin_after_tokenization if sep_swin_for_views is None else sep_swin_for_views,
        n_obs_steps=1,
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        n_encoder_layers=4,
        n_decoder_layers=1,
        use_vae=True,
        latent_dim=32,
        n_vae_encoder_layers=4,
        kl_weight=kl_weight,
        dropout=dropout,
        optimizer_lr=lr,
        optimizer_lr_backbone=lr_backbone,
        optimizer_weight_decay=weight_decay,
    )


def build_warmup_cosine_lr_lambda(total_steps: int, warmup_steps: int, min_lr_ratio: float = 0.1):
    warmup_steps = min(warmup_steps, max(total_steps - 1, 1))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return _lr_lambda


def infinite_dataloader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


@contextmanager
def unwrapped_policy_model(policy: FoldedACTPolicy, raw_model: FoldedACT):
    current_model = policy.model
    policy.model = raw_model
    try:
        yield policy
    finally:
        policy.model = current_model


def save_checkpoint_bundle(
    ckpt_dir: Path,
    policy: FoldedACTPolicy,
    stats: dict[str, dict[str, torch.Tensor]],
    metadata: dict[str, Any],
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    orig_model = policy.model
    if hasattr(orig_model, "_orig_mod"):
        policy.model = cast(FoldedACT, orig_model._orig_mod)
    try:
        policy.save_pretrained(ckpt_dir)
    finally:
        policy.model = orig_model
    torch.save(stats, ckpt_dir / "train_stats.pt")
    save_json(metadata, ckpt_dir / "checkpoint_meta.json")


def init_wandb_run(
    args: argparse.Namespace,
    out_dir: Path,
    run_meta: dict[str, Any],
) -> Any | None:
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    if wandb is None:
        raise ImportError(
            "wandb is required when W&B logging is enabled. Install wandb or pass --no-wandb."
        )
    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "name": args.wandb_run_name or out_dir.name,
        "dir": str(out_dir),
        "config": run_meta,
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    if args.wandb_group:
        init_kwargs["group"] = args.wandb_group
    if args.wandb_mode:
        init_kwargs["mode"] = args.wandb_mode
    return wandb.init(**init_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ACT with configurable camera views on STEA_pick_success")
    parser.add_argument(
        "--camera-preset",
        type=str,
        default="2cams_top_left",
        choices=list(CAMERA_PRESETS.keys()),
        help="Camera subset preset from policy/STEA_dataset_distribution_and_act_analysis.md",
    )
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--dof-mode", type=str, default="8d", choices=["8d", "16d"])
    parser.add_argument("--gripper-mode", type=str, default="binary", choices=["binary", "continuous"])
    parser.add_argument("--gripper-shift", type=int, default=1)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.70)
    parser.add_argument("--image-norm", type=str, default="imagenet", choices=["imagenet", "train_stats"])
    parser.add_argument(
        "--spatial-folding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fold diagonal camera token pairs (enabled by default)",
    )
    parser.add_argument(
        "--swin-after-tokenization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply a shifted-window Swin block before overlay and folding (enabled by default)",
    )
    parser.add_argument(
        "--camera-overlay",
        action="store_true",
        help="Fuse adjacent camera pairs (2cams_top_left or all_4cams); uses 6-ch conv1 when --early-camera-overlay is enabled, or sums post-Swin tokens otherwise",
    )
    parser.add_argument(
        "--early-camera-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --camera-overlay is set, channel-stack camera pairs into a widened 6-ch ResNet18 conv1 (use --no-early-camera-overlay for post-Swin token sum)",
    )
    parser.add_argument(
        "--sep-swin-for-views",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use an independent Swin block per visual stream before folding (default when Swin is enabled)",
    )
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no-amp", action="store_true", help="Disable bf16 mixed precision")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile policy.model with torch.compile to speed up training (enabled by default on CUDA)",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile mode when --compile is enabled",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log training losses, validation losses, and best-checkpoint test losses to W&B",
    )
    parser.add_argument("--wandb-project", type=str, default="stea-act")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
    )
    args = parser.parse_args()
    if args.camera_overlay:
        if len(CAMERA_PRESETS[args.camera_preset]) not in (2, 4):
            parser.error("--camera-overlay requires 2cams_top_left or all_4cams")
        if not args.early_camera_overlay and not args.swin_after_tokenization:
            parser.error("--camera-overlay with --no-early-camera-overlay requires --swin-after-tokenization")
    if args.sep_swin_for_views and not args.swin_after_tokenization:
        parser.error("--sep-swin-for-views requires --swin-after-tokenization")
    if args.sep_swin_for_views is None:
        args.sep_swin_for_views = args.swin_after_tokenization
    return args


def main() -> None:
    args = parse_args()
    validate_splits()
    seed_everything(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    camera_keys = CAMERA_PRESETS[args.camera_preset]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "outputs" / "STEA_pick_success" / f"act_{args.camera_preset}_{args.dof_mode}_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== STEA ACT Camera-View Training: {args.camera_preset} ===")
    print(f"  Device       : {device}")
    print(f"  Cameras ({len(camera_keys)})  : {list(camera_keys)}")
    print(f"  DoF / Gripper: {args.dof_mode} / {args.gripper_mode} (shift={args.gripper_shift}, thr={args.gripper_open_threshold})")
    print(f"  Chunk / Batch: chunk_size={args.chunk_size}, batch_size={args.batch_size}, steps={args.steps}")
    print(f"  Output Dir   : {out_dir}")

    trajectories = load_reconstructed_trajectories(
        dataset_root=args.dataset_root,
        dof_mode=args.dof_mode,
        gripper_mode=args.gripper_mode,
        gripper_shift=args.gripper_shift,
        gripper_open_threshold=args.gripper_open_threshold,
    )

    ds_kwargs = dict(
        camera_keys=camera_keys,
        chunk_size=args.chunk_size,
        dof_mode=args.dof_mode,
        gripper_mode=args.gripper_mode,
        gripper_shift=args.gripper_shift,
        gripper_open_threshold=args.gripper_open_threshold,
        dataset_root=args.dataset_root,
        cache_dir=args.cache_dir,
        trajectories=trajectories,
    )
    train_ds = STEAACTDataset(SPLITS["train"], **ds_kwargs)
    val_ds = STEAACTDataset(SPLITS["val"], **ds_kwargs)
    test_ds = STEAACTDataset(SPLITS["test"], **ds_kwargs)

    print(
        f"  Splits       : train={len(train_ds)} frames ({len(train_ds.episodes)} eps), "
        f"val={len(val_ds)} frames ({len(val_ds.episodes)} eps), "
        f"test={len(test_ds)} frames ({len(test_ds.episodes)} eps)"
    )

    stats = compute_train_stats(train_ds, image_norm=args.image_norm)
    normalizer = BatchNormalizer(stats=stats, camera_keys=camera_keys, device=device)

    act_cfg = build_act_config(
        camera_keys=camera_keys,
        image_shapes=train_ds.image_shapes,
        state_dim=train_ds.subspace.state_dim,
        action_dim=train_ds.subspace.action_dim,
        chunk_size=args.chunk_size,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        kl_weight=args.kl_weight,
        device=str(device),
        spatial_folding=args.spatial_folding,
        swin_after_tokenization=args.swin_after_tokenization,
        camera_overlay=args.camera_overlay,
        early_camera_overlay=args.early_camera_overlay,
        sep_swin_for_views=args.sep_swin_for_views,
    )

    policy = FoldedACTPolicy(act_cfg).to(device)
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Parameters   : total={total_params:,}, trainable={trainable_params:,}")

    optim_groups = policy.get_optim_params()
    # Explicitly attach base lr to non-backbone group so LambdaLR scales both groups cleanly
    optim_groups[0]["lr"] = args.lr
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=build_warmup_cosine_lr_lambda(args.steps, args.warmup_steps)
    )

    raw_model = policy.model
    use_compile = args.compile and device.type == "cuda" and hasattr(torch, "compile")
    if use_compile:
        print(f"  torch.compile: enabled (mode={args.compile_mode})")
        policy.model = cast(FoldedACT, torch.compile(raw_model, mode=args.compile_mode))
    else:
        print("  torch.compile: disabled")

    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        generator=gen,
    )
    train_iter = infinite_dataloader(train_loader)

    use_amp = (not args.no_amp) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    visual_tokens = 0
    stride = 16 if act_cfg.replace_final_stride_with_dilation else 32
    for camera_key in camera_keys[::2] if act_cfg.camera_overlay else camera_keys:
        _, height, width = train_ds.image_shapes[camera_key]
        feature_rows = (height + stride - 1) // stride
        feature_cols = (width + stride - 1) // stride
        if act_cfg.spatial_folding:
            visual_tokens += 2 * ((feature_rows + 1) // 2) * ((feature_cols + 1) // 2)
        else:
            visual_tokens += feature_rows * feature_cols
    run_meta = {
        "camera_preset": args.camera_preset,
        "camera_keys": list(camera_keys),
        "spatial_folding": act_cfg.spatial_folding,
        "swin_after_tokenization": act_cfg.swin_after_tokenization,
        "camera_overlay": act_cfg.camera_overlay,
        "early_camera_overlay": act_cfg.early_camera_overlay,
        "sep_swin_for_views": act_cfg.sep_swin_for_views,
        "visual_tokens_per_step": visual_tokens,
        "dof_mode": args.dof_mode,
        "gripper_mode": args.gripper_mode,
        "gripper_shift": args.gripper_shift,
        "gripper_open_threshold": args.gripper_open_threshold,
        "image_norm": args.image_norm,
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "lr": args.lr,
        "lr_backbone": args.lr_backbone,
        "weight_decay": args.weight_decay,
        "kl_weight": args.kl_weight,
        "seed": args.seed,
        "use_bf16_amp": use_amp,
        "use_torch_compile": use_compile,
        "compile_mode": args.compile_mode if use_compile else None,
        "train_frames": len(train_ds),
        "val_frames": len(val_ds),
        "test_frames": len(test_ds),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "started_at": ts,
    }
    save_json(run_meta, out_dir / "run_config.json")
    wandb_run = init_wandb_run(args, out_dir, run_meta)

    history_path = out_dir / "train_history.jsonl"
    best_val_score = float("inf")
    best_val_step = 0

    t_start = time.time()
    running_loss = 0.0
    running_l1 = 0.0
    running_kld = 0.0
    running_count = 0

    try:
        policy.train()
        for step in range(1, args.steps + 1):
            raw_batch = next(train_iter)
            norm_batch = normalizer.normalize_batch(raw_batch, include_action=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, loss_dict = policy.forward(norm_batch)

            loss.backward()
            grad_norm = clip_grad_norm_(policy.parameters(), args.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            running_l1 += float(loss_dict["l1_loss"])
            running_kld += float(loss_dict.get("kld_loss", 0.0))
            running_count += 1

            if step % args.log_freq == 0 or step == 1:
                elapsed = time.time() - t_start
                steps_per_s = step / max(elapsed, 1e-6)
                avg_loss = running_loss / running_count
                avg_l1 = running_l1 / running_count
                avg_kld = running_kld / running_count
                cur_lr = optimizer.param_groups[0]["lr"]
                log_rec = {
                    "type": "train",
                    "step": step,
                    "loss": round(avg_loss, 5),
                    "l1_loss": round(avg_l1, 5),
                    "kld_loss": round(avg_kld, 5),
                    "grad_norm": round(float(grad_norm), 4),
                    "lr": cur_lr,
                    "steps_per_sec": round(steps_per_s, 2),
                    "elapsed_sec": round(elapsed, 1),
                }
                with open(history_path, "a", encoding="utf-8") as hf:
                    hf.write(json.dumps(log_rec) + "\n")
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": avg_loss,
                            "train/l1_loss": avg_l1,
                            "train/kld_loss": avg_kld,
                            "train/grad_norm": float(grad_norm),
                            "train/lr": cur_lr,
                            "train/steps_per_sec": steps_per_s,
                        },
                        step=step,
                    )
                print(
                    f"[{args.camera_preset} | step {step:5d}/{args.steps}] "
                    f"loss={avg_loss:.4f} (l1={avg_l1:.4f}, kld={avg_kld:.4f}) "
                    f"lr={cur_lr:.2e} speed={steps_per_s:.1f} it/s",
                    flush=True,
                )
                running_loss = 0.0
                running_l1 = 0.0
                running_kld = 0.0
                running_count = 0

            if step % args.eval_freq == 0 or step == args.steps:
                with unwrapped_policy_model(policy, raw_model):
                    val_res = evaluate_policy_on_dataset(
                        policy,
                        val_ds,
                        normalizer,
                        batch_size=args.batch_size,
                        num_workers=min(args.num_workers, 2),
                        record_attention=False,
                    )
                val_chunk_m = val_res["chunk_horizon_metrics"]
                val_score = val_chunk_m["norm_l1_active8d"]
                te_m = val_res["temporal_ensemble_metrics"]
                val_rec = {
                    "type": "val",
                    "step": step,
                    "val_norm_l1": round(val_score, 5),
                    "val_te_pos_l2_mm": round(te_m["pos_l2_mm"], 3),
                    "val_te_rot_deg": round(te_m["rot_geodesic_deg"], 3),
                    "val_te_gripper_acc_pct": round(te_m["gripper_acc_pct"], 2),
                    "val_grasp_mae_frames": round(te_m["grasp_timing_mae_frames"], 2),
                }
                with open(history_path, "a", encoding="utf-8") as hf:
                    hf.write(json.dumps(val_rec) + "\n")

                is_best = val_score < best_val_score
                if is_best:
                    best_val_score = val_score
                    best_val_step = step
                    save_checkpoint_bundle(
                        out_dir / "checkpoints" / "best_val",
                        policy,
                        stats,
                        {"step": step, "val_norm_l1": val_score, **val_rec},
                    )

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "val/norm_l1": val_score,
                            "val/best_norm_l1": best_val_score,
                            "val/chunk_pos_l1_mm": val_chunk_m["pos_l1_mm"],
                            "val/chunk_pos_l2_mm": val_chunk_m["pos_l2_mm"],
                            "val/chunk_rot_l1": val_chunk_m["rot_l1"],
                            "val/chunk_rot_deg": val_chunk_m["rot_geodesic_deg"],
                            "val/chunk_gripper_l1": val_chunk_m["gripper_l1"],
                            "val/te_pos_l2_mm": te_m["pos_l2_mm"],
                            "val/te_rot_deg": te_m["rot_geodesic_deg"],
                            "val/te_gripper_acc_pct": te_m["gripper_acc_pct"],
                            "val/te_gripper_f1_pct": te_m["gripper_f1_pct"],
                            "val/grasp_mae_frames": te_m["grasp_timing_mae_frames"],
                        },
                        step=step,
                    )

                tag = " *BEST*" if is_best else ""
                print(
                    f"  -> [VAL @ step {step:5d}] norm_l1={val_score:.4f} | "
                    f"TE pos_l2={te_m['pos_l2_mm']:.2f}mm, rot={te_m['rot_geodesic_deg']:.2f}deg, "
                    f"grip_acc={te_m['gripper_acc_pct']:.1f}%, grasp_mae={te_m['grasp_timing_mae_frames']:.1f}f{tag}",
                    flush=True,
                )
                policy.train()

        total_train_sec = time.time() - t_start
        policy.model = raw_model
        save_checkpoint_bundle(
            out_dir / "checkpoints" / "last",
            policy,
            stats,
            {"step": args.steps, "total_train_sec": total_train_sec},
        )

        # Evaluate LAST checkpoint on val & test with cross-attention attribution
        print(f"[{args.camera_preset}] Evaluating final ('last' @ step {args.steps}) checkpoint on Val & Test ...")
        last_val_eval = evaluate_policy_on_dataset(
            policy, val_ds, normalizer, batch_size=args.batch_size, num_workers=2, record_attention=True
        )
        last_test_eval = evaluate_policy_on_dataset(
            policy, test_ds, normalizer, batch_size=args.batch_size, num_workers=2, record_attention=True
        )

        # Reload BEST_VAL checkpoint and evaluate on val & test with cross-attention attribution
        best_ckpt_dir = out_dir / "checkpoints" / "best_val"
        print(f"[{args.camera_preset}] Evaluating best_val (@ step {best_val_step}) checkpoint on Val & Test ...")
        best_policy = FoldedACTPolicy.from_pretrained(str(best_ckpt_dir)).to(device)
        best_val_eval = evaluate_policy_on_dataset(
            best_policy, val_ds, normalizer, batch_size=args.batch_size, num_workers=2, record_attention=True
        )
        best_test_eval = evaluate_policy_on_dataset(
            best_policy, test_ds, normalizer, batch_size=args.batch_size, num_workers=2, record_attention=True
        )

        final_summary = {
            "run_config": {
                **run_meta,
                "total_train_sec": round(total_train_sec, 2),
                "best_val_step": best_val_step,
                "best_val_norm_l1": best_val_score,
            },
            "best_val_checkpoint": {
                "step": best_val_step,
                "val": best_val_eval,
                "test": best_test_eval,
            },
            "last_checkpoint": {
                "step": args.steps,
                "val": last_val_eval,
                "test": last_test_eval,
            },
        }
        save_json(final_summary, out_dir / "eval_results.json")

        bt = best_test_eval["temporal_ensemble_metrics"]
        bc = best_test_eval["chunk_horizon_metrics"]
        if wandb_run is not None:
            best_test_metrics = {
                "test_best/norm_l1": bc["norm_l1_active8d"],
                "test_best/chunk_pos_l1_mm": bc["pos_l1_mm"],
                "test_best/chunk_pos_l2_mm": bc["pos_l2_mm"],
                "test_best/chunk_rot_l1": bc["rot_l1"],
                "test_best/chunk_rot_deg": bc["rot_geodesic_deg"],
                "test_best/chunk_gripper_l1": bc["gripper_l1"],
                "test_best/chunk_gripper_acc_pct": bc["gripper_acc_pct"],
                "test_best/chunk_gripper_f1_pct": bc["gripper_f1_pct"],
                "test_best/te_pos_l2_mm": bt["pos_l2_mm"],
                "test_best/te_rot_deg": bt["rot_geodesic_deg"],
                "test_best/te_gripper_acc_pct": bt["gripper_acc_pct"],
                "test_best/te_gripper_f1_pct": bt["gripper_f1_pct"],
                "test_best/grasp_mae_frames": bt["grasp_timing_mae_frames"],
            }
            wandb_run.log(best_test_metrics, step=args.steps)
            wandb_run.summary.update(
                {
                    "best_val_step": best_val_step,
                    "best_val_norm_l1": best_val_score,
                    "total_train_sec": round(total_train_sec, 2),
                    **best_test_metrics,
                }
            )

        print(
            f"=== DONE {args.camera_preset} in {total_train_sec / 60.0:.2f} min (best_step={best_val_step}) ===\n"
            f"  [TEST best_val] norm_l1={bc['norm_l1_active8d']:.4f} | "
            f"TE pos_l2={bt['pos_l2_mm']:.2f} mm | TE rot={bt['rot_geodesic_deg']:.2f} deg | "
            f"TE grip_acc={bt['gripper_acc_pct']:.2f}% (F1={bt['gripper_f1_pct']:.2f}%) | "
            f"grasp_mae={bt['grasp_timing_mae_frames']:.2f} frames",
            flush=True,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
