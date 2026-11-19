"""Offline evaluation and cross-attention attribution for STEA ACT camera-view ablations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler

from policy.splits_and_configs import CAMERA_SHORT_NAMES, EPISODE_BAG_COUNT, SubspaceSpec
from policy.stea_act_dataset import STEAACTDataset


class BatchNormalizer:
    """Fast GPU normalizer/unnormalizer matching LeRobot's MEAN_STD NormalizerProcessorStep."""

    def __init__(
        self,
        stats: dict[str, dict[str, Tensor]],
        camera_keys: tuple[str, ...],
        device: torch.device | str,
        eps: float = 1e-8,
    ) -> None:
        self.camera_keys = camera_keys
        self.device = torch.device(device)
        self.eps = eps
        self.stats_gpu: dict[str, dict[str, Tensor]] = {}
        for key, sub in stats.items():
            self.stats_gpu[key] = {
                s_name: t.to(device=self.device, dtype=torch.float32) for s_name, t in sub.items()
            }

    def normalize_batch(self, batch: dict[str, Tensor], include_action: bool = True) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for k, v in batch.items():
            if isinstance(v, Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v

        st = self.stats_gpu["observation.state"]
        out["observation.state"] = (out["observation.state"] - st["mean"]) / (st["std"] + self.eps)

        for ck in self.camera_keys:
            cs = self.stats_gpu[ck]
            img = out[ck]
            if img.dtype == torch.uint8:
                img = img.to(dtype=torch.float32).div_(255.0)
            out[ck] = (img - cs["mean"]) / (cs["std"] + self.eps)

        if include_action and "action" in out:
            act_s = self.stats_gpu["action"]
            out["action"] = (out["action"] - act_s["mean"]) / (act_s["std"] + self.eps)

        return out

    def unnormalize_action(self, norm_action: Tensor) -> Tensor:
        act_s = self.stats_gpu["action"]
        return norm_action * (act_s["std"] + self.eps) + act_s["mean"]


class CrossAttentionRecorder:
    """Forward hook capturing decoder-to-encoder cross-attention weights in ACT."""

    def __init__(self, policy: ACTPolicy) -> None:
        self.weights: list[Tensor] = []
        self._handle = policy.model.decoder.layers[-1].multihead_attn.register_forward_hook(self._hook)

    def _hook(self, module: torch.nn.Module, inputs: tuple, output: tuple) -> None:
        # nn.MultiheadAttention returns (attn_output, attn_output_weights) where weights are (B, Q, K)
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            self.weights.append(output[1].detach().cpu())

    def pop_latest(self) -> Tensor | None:
        if not self.weights:
            return None
        latest = self.weights[-1]
        self.weights.clear()
        return latest

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def quaternion_geodesic_deg(pred_q: np.ndarray, gt_q: np.ndarray) -> np.ndarray:
    """Compute SO(3) geodesic angle error in degrees between quaternions (..., 4)."""
    pred_norm = pred_q / np.maximum(np.linalg.norm(pred_q, axis=-1, keepdims=True), 1e-8)
    gt_norm = gt_q / np.maximum(np.linalg.norm(gt_q, axis=-1, keepdims=True), 1e-8)
    dot = np.clip(np.abs(np.sum(pred_norm * gt_norm, axis=-1)), 0.0, 1.0)
    return (2.0 * np.arccos(dot)) * (180.0 / math.pi)


def binary_f1(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    tp = float(np.sum((pred_bin == 1) & (gt_bin == 1)))
    fp = float(np.sum((pred_bin == 1) & (gt_bin == 0)))
    fn = float(np.sum((pred_bin == 0) & (gt_bin == 1)))
    denom = 2.0 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 1.0


def summarize_step_errors(
    pred_act: np.ndarray,
    gt_act: np.ndarray,
    subspace: SubspaceSpec,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute physical arm position (mm), orientation (deg), and gripper accuracy/F1 metrics."""
    if valid_mask is not None:
        pred_act = pred_act[valid_mask]
        gt_act = gt_act[valid_mask]

    pred_pos = pred_act[:, subspace.right_pos_slice]
    gt_pos = gt_act[:, subspace.right_pos_slice]

    pred_quat = pred_act[:, subspace.right_quat_slice]
    gt_quat = gt_act[:, subspace.right_quat_slice]

    pred_grip = pred_act[:, subspace.right_gripper_idx]
    gt_grip = gt_act[:, subspace.right_gripper_idx]

    pos_l1_mm = float(np.mean(np.abs(pred_pos - gt_pos)) * 1000.0)
    pos_l2_mm = float(np.mean(np.linalg.norm(pred_pos - gt_pos, axis=-1)) * 1000.0)
    rot_l1 = float(np.mean(np.abs(pred_quat - gt_quat)))
    rot_deg = float(np.mean(quaternion_geodesic_deg(pred_quat, gt_quat)))

    pred_bin = (pred_grip >= 0.5).astype(np.int32)
    gt_bin = (gt_grip >= 0.5).astype(np.int32)
    grip_acc = float(np.mean(pred_bin == gt_bin) * 100.0)
    grip_f1 = float(binary_f1(pred_bin, gt_bin) * 100.0)
    grip_l1 = float(np.mean(np.abs(pred_grip - gt_grip)))

    return {
        "num_steps": int(pred_act.shape[0]),
        "pos_l1_mm": pos_l1_mm,
        "pos_l2_mm": pos_l2_mm,
        "rot_l1": rot_l1,
        "rot_geodesic_deg": rot_deg,
        "gripper_acc_pct": grip_acc,
        "gripper_f1_pct": grip_f1,
        "gripper_l1": grip_l1,
    }


def classify_phase_from_raw_gripper_joint(joint_rad: np.ndarray) -> np.ndarray:
    """Classify each frame into 0=approach (>0.70), 1=closing ([0.05, 0.70]), 2=lift_closed (<0.05)."""
    phase = np.full(joint_rad.shape[0], 1, dtype=np.int32)
    phase[joint_rad > 0.70] = 0
    phase[joint_rad < 0.05] = 2
    return phase


@torch.no_grad()
def evaluate_policy_on_dataset(
    policy: ACTPolicy,
    dataset: STEAACTDataset,
    normalizer: BatchNormalizer,
    *,
    batch_size: int = 64,
    num_workers: int = 2,
    temporal_ensemble_coeff: float = 0.01,
    record_attention: bool = True,
) -> dict[str, Any]:
    """Run full chunk, step-0, temporal-ensembled rollout, stratified clutter, and attention evaluation."""
    policy.eval()
    subspace = dataset.subspace
    camera_keys = dataset.camera_keys

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    attn_recorder = CrossAttentionRecorder(policy) if record_attention else None

    all_pred_chunks: list[np.ndarray] = []
    all_gt_chunks: list[np.ndarray] = []
    all_valid_masks: list[np.ndarray] = []
    all_norm_l1_per_sample: list[np.ndarray] = []
    all_ep_indices: list[np.ndarray] = []
    all_frame_indices: list[np.ndarray] = []
    all_abs_indices: list[np.ndarray] = []
    all_attn_step0: list[np.ndarray] = []
    all_attn_chunk_mean: list[np.ndarray] = []

    right_active_dims = (
        list(range(8)) if subspace.mode == "8d" else [7, 8, 9, 10, 11, 12, 13, 15]
    )

    try:
        for raw_batch in loader:
            norm_batch = normalizer.normalize_batch(raw_batch, include_action=True)
            norm_pred_chunk = policy.predict_action_chunk(norm_batch)  # (B, S, D)

            # Standardized L1 on active right-arm 8 dims (apples-to-apples across 8d and 16d)
            valid = ~norm_batch["action_is_pad"]  # (B, S)
            diff_active = torch.abs(
                norm_pred_chunk[:, :, right_active_dims] - norm_batch["action"][:, :, right_active_dims]
            ).mean(dim=-1)  # (B, S)
            sample_norm_l1 = (diff_active * valid).sum(dim=-1) / torch.clamp(valid.sum(dim=-1), min=1)

            unnorm_pred = normalizer.unnormalize_action(norm_pred_chunk).cpu().numpy()
            gt_chunk = raw_batch["action"].numpy()
            valid_np = (~raw_batch["action_is_pad"]).numpy()

            all_pred_chunks.append(unnorm_pred)
            all_gt_chunks.append(gt_chunk)
            all_valid_masks.append(valid_np)
            all_norm_l1_per_sample.append(sample_norm_l1.cpu().numpy())
            all_ep_indices.append(raw_batch["episode_index"].numpy())
            all_frame_indices.append(raw_batch["frame_index"].numpy())
            all_abs_indices.append(raw_batch["index"].numpy())

            if attn_recorder is not None:
                attn_w = attn_recorder.pop_latest()  # (B, chunk_size, encoder_seq_len)
                if attn_w is not None:
                    all_attn_step0.append(attn_w[:, 0, :].numpy())
                    all_attn_chunk_mean.append(attn_w.mean(dim=1).numpy())
    finally:
        if attn_recorder is not None:
            attn_recorder.close()

    pred_chunks = np.concatenate(all_pred_chunks, axis=0)      # (N, S, D)
    gt_chunks = np.concatenate(all_gt_chunks, axis=0)          # (N, S, D)
    valid_masks = np.concatenate(all_valid_masks, axis=0)      # (N, S)
    norm_l1_samples = np.concatenate(all_norm_l1_per_sample, axis=0)  # (N,)
    ep_indices = np.concatenate(all_ep_indices, axis=0)        # (N,)
    abs_indices = np.concatenate(all_abs_indices, axis=0)      # (N,)

    # 1. Full non-padded chunk metrics
    chunk_metrics = summarize_step_errors(
        pred_chunks.reshape(-1, subspace.action_dim),
        gt_chunks.reshape(-1, subspace.action_dim),
        subspace,
        valid_mask=valid_masks.reshape(-1),
    )
    chunk_metrics["norm_l1_active8d"] = float(np.mean(norm_l1_samples))

    # 2. Step-0 immediate prediction metrics
    step0_pred = pred_chunks[:, 0, :]
    step0_gt = gt_chunks[:, 0, :]
    step0_metrics = summarize_step_errors(step0_pred, step0_gt, subspace)

    # 3. Sequential Temporal-Ensembled episode rollout metrics + grasp timing accuracy
    te_preds = np.zeros_like(step0_pred)
    grasp_timing_errors: list[float] = []
    grasp_trigger_hits = 0

    for ep in dataset.episodes:
        ep_mask = np.flatnonzero(ep_indices == ep)
        # Dataset is ordered sequentially per episode
        ensembler = ACTTemporalEnsembler(temporal_ensemble_coeff, dataset.chunk_size)
        ep_chunks_t = torch.from_numpy(pred_chunks[ep_mask])  # (T_ep, S, D)
        ep_te_out: list[np.ndarray] = []
        for t in range(ep_chunks_t.shape[0]):
            act_t = ensembler.update(ep_chunks_t[t : t + 1])  # (1, D)
            ep_te_out.append(act_t[0].numpy())
        ep_te_arr = np.stack(ep_te_out, axis=0)
        te_preds[ep_mask] = ep_te_arr

        # Grasp timing alignment (first frame where gripper command >= 0.5)
        gt_g = step0_gt[ep_mask, subspace.right_gripper_idx] >= 0.5
        pr_g = ep_te_arr[:, subspace.right_gripper_idx] >= 0.5
        gt_close_idxs = np.flatnonzero(gt_g)
        pr_close_idxs = np.flatnonzero(pr_g)
        if len(gt_close_idxs) > 0 and len(pr_close_idxs) > 0:
            grasp_trigger_hits += 1
            grasp_timing_errors.append(float(abs(int(pr_close_idxs[0]) - int(gt_close_idxs[0]))))

    te_metrics = summarize_step_errors(te_preds, step0_gt, subspace)
    te_metrics["grasp_trigger_rate_pct"] = float(100.0 * grasp_trigger_hits / max(len(dataset.episodes), 1))
    te_metrics["grasp_timing_mae_frames"] = (
        float(np.mean(grasp_timing_errors)) if grasp_timing_errors else float("nan")
    )
    te_metrics["grasp_timing_mae_sec"] = (
        float(np.mean(grasp_timing_errors) / 10.0) if grasp_timing_errors else float("nan")
    )

    # 4. Stratified metrics by scene clutter (bag_count = 1..5)
    by_bag_count: dict[str, Any] = {}
    sample_bag_counts = np.array([EPISODE_BAG_COUNT[int(e)] for e in ep_indices], dtype=np.int32)
    for bc in range(1, 6):
        bc_mask = sample_bag_counts == bc
        if not np.any(bc_mask):
            continue
        bc_chunk = summarize_step_errors(
            pred_chunks[bc_mask].reshape(-1, subspace.action_dim),
            gt_chunks[bc_mask].reshape(-1, subspace.action_dim),
            subspace,
            valid_mask=valid_masks[bc_mask].reshape(-1),
        )
        bc_te = summarize_step_errors(te_preds[bc_mask], step0_gt[bc_mask], subspace)
        by_bag_count[str(bc)] = {
            "num_episodes": int(len(set(ep_indices[bc_mask].tolist()))),
            "num_frames": int(np.sum(bc_mask)),
            "norm_l1_active8d": float(np.mean(norm_l1_samples[bc_mask])),
            "chunk_pos_l2_mm": bc_chunk["pos_l2_mm"],
            "chunk_rot_deg": bc_chunk["rot_geodesic_deg"],
            "chunk_gripper_acc_pct": bc_chunk["gripper_acc_pct"],
            "te_pos_l2_mm": bc_te["pos_l2_mm"],
            "te_rot_deg": bc_te["rot_geodesic_deg"],
            "te_gripper_acc_pct": bc_te["gripper_acc_pct"],
        }

    # 5. Stratified metrics by physical phase (approach / closing / lift_closed)
    gripper_state_joint = dataset.all_state[abs_indices, subspace.right_gripper_idx]
    phases = classify_phase_from_raw_gripper_joint(gripper_state_joint)
    phase_names = {0: "approach_open", 1: "closing_transition", 2: "post_grasp_lift"}
    by_phase: dict[str, Any] = {}
    for p_id, p_name in phase_names.items():
        p_mask = phases == p_id
        if not np.any(p_mask):
            continue
        p_te = summarize_step_errors(te_preds[p_mask], step0_gt[p_mask], subspace)
        p_te["norm_l1_active8d"] = float(np.mean(norm_l1_samples[p_mask]))
        by_phase[p_name] = p_te

    # 6. Cross-attention attribution per token source (latent, proprio_state, each camera)
    attention_summary: dict[str, Any] = {}
    if all_attn_chunk_mean:
        attn_chunk = np.concatenate(all_attn_chunk_mean, axis=0)  # (N, K_total)
        attn_s0 = np.concatenate(all_attn_step0, axis=0)          # (N, K_total)
        k_total = attn_chunk.shape[1]
        group_size = 2 if getattr(policy.config, "camera_overlay", False) else 1
        camera_labels = {
            " + ".join(group): " + ".join(CAMERA_SHORT_NAMES.get(key, key) for key in group)
            for start in range(0, len(camera_keys), group_size)
            for group in (camera_keys[start : start + group_size],)
        }
        tokens_per_cam = (k_total - 2) // max(len(camera_labels), 1)

        def _decompose_attn(arr_2d: np.ndarray) -> dict[str, float]:
            out_d: dict[str, float] = {
                "latent_pct": float(arr_2d[:, 0].mean() * 100.0),
                "proprio_state_pct": float(arr_2d[:, 1].mean() * 100.0),
            }
            offset = 2
            for ck in camera_labels:
                cam_mass = arr_2d[:, offset : offset + tokens_per_cam].sum(axis=-1).mean() * 100.0
                out_d[ck] = float(cam_mass)
                offset += tokens_per_cam
            return out_d

        attention_summary["tokens_per_camera"] = int(tokens_per_cam)
        attention_summary["total_encoder_tokens"] = int(k_total)
        attention_summary["chunk_mean_pct"] = _decompose_attn(attn_chunk)
        attention_summary["step0_pct"] = _decompose_attn(attn_s0)
        attention_summary["by_phase_chunk_mean_pct"] = {
            p_name: _decompose_attn(attn_chunk[phases == p_id])
            for p_id, p_name in phase_names.items()
            if np.any(phases == p_id)
        }
        attention_summary["camera_labels"] = camera_labels

    return {
        "num_episodes": len(dataset.episodes),
        "num_frames": len(dataset),
        "chunk_horizon_metrics": chunk_metrics,
        "step0_metrics": step0_metrics,
        "temporal_ensemble_metrics": te_metrics,
        "by_bag_count": by_bag_count,
        "by_phase": by_phase,
        "cross_attention": attention_summary,
    }


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
