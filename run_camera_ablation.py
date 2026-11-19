"""Multi-GPU launcher & report generator for the 5-setting STEA ACT camera-view ablation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.evaluate_act import save_json
from policy.splits_and_configs import (
    ALL_CAMERA_KEYS,
    CAM_FRONT,
    CAM_LEFT,
    CAM_RIGHT,
    CAM_TOP,
    CAMERA_PRESETS,
    DEFAULT_DATASET_ROOT,
    validate_splits,
)
from policy.stea_act_dataset import DEFAULT_CACHE_DIR, ensure_video_frame_cache

# Longest-Processing-Time-first order to minimize makespan across 2 GPUs
DEFAULT_PRESET_ORDER = [
    "all_4cams",
    "3cams_active",
    "2cams_top_left",
    "1cam_top",
    "1cam_left",
]

DISPLAY_ORDER = [
    "all_4cams",
    "3cams_active",
    "2cams_top_left",
    "1cam_top",
    "1cam_left",
]


def format_markdown_report(
    summary: dict[str, Any],
    presets_in_order: list[str],
    timestamp: str,
) -> str:
    cfg_any = next(iter(summary["runs"].values()))["run_config"]
    lines: list[str] = [
        f"# ACT Multi-Camera Partial-Observability Ablation Results (`{timestamp}`)",
        "",
        "## 1. Experimental Protocol",
        "",
        f"- **Dataset**: `STEA_pick_success` (`113` episodes, `10,675` frames @ `10 Hz`, `320×240` RGB)",
        f"- **Stratified Split**: `91` Train (`8,685` frames) / `11` Val (`1,020` frames) / `11` Test (`970` frames) (`bag_count` distribution `3:3:3:1:1` in Val & Test)",
        f"- **State & Action Subspace**: `{cfg_any['dof_mode']}` active right arm (`7` joints + gripper $\\rightarrow$ `7D` Cartesian EE pose `xyz + qx,qy,qz,qw` + reconstructed `{cfg_any['gripper_mode']}` gripper command `0=open, 1=closed`)",
        f"- **ACT Architecture & Training**: `ResNet18` (`IMAGENET1K_V1`), `chunk_size={cfg_any['chunk_size']}` (`{cfg_any['chunk_size']/10:.1f} s`), `batch_size={cfg_any['batch_size']}`, `steps={cfg_any['steps']}`, `lr={cfg_any['lr']}`, `lr_backbone={cfg_any['lr_backbone']}`, `kl_weight={cfg_any['kl_weight']}`, `seed={cfg_any['seed']}`",
        "- **Checkpoint Selection**: `best_val` selected strictly by minimum standardized `norm_l1` on the held-out `Val` split (zero test-set peeking).",
        "",
        "---",
        "",
        "## 2. Main Test-Split Results (`best_val` Checkpoint, `N = 11` Episodes / `970` Frames)",
        "",
        "| Setting | Cameras | Vis. Tokens | Best Step | Train Time | Chunk Norm L1 $\\downarrow$ | TE Pos L2 (`mm`) $\\downarrow$ | TE Rot (`deg`) $\\downarrow$ | TE Gripper Acc (`%`) $\\uparrow$ | TE Gripper F1 (`%`) $\\uparrow$ | Grasp Timing MAE (`frames`) $\\downarrow$ |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for preset in presets_in_order:
        if preset not in summary["runs"]:
            continue
        r = summary["runs"][preset]
        rc = r["run_config"]
        test_b = r["best_val_checkpoint"]["test"]
        cm = test_b["chunk_horizon_metrics"]
        tm = test_b["temporal_ensemble_metrics"]
        cams_short = " + ".join(ck.replace("observation.images.camera_", "") for ck in rc["camera_keys"])
        lines.append(
            f"| **`{preset}`** | `{cams_short}` | `{rc['visual_tokens_per_step']}` | "
            f"`{rc['best_val_step']}` | `{rc['total_train_sec']/60.0:.1f}m` | "
            f"`{cm['norm_l1_active8d']:.4f}` | `{tm['pos_l2_mm']:.2f}` | `{tm['rot_geodesic_deg']:.2f}` | "
            f"`{tm['gripper_acc_pct']:.2f}` | `{tm['gripper_f1_pct']:.2f}` | `{tm['grasp_timing_mae_frames']:.2f}` |"
        )

    lines.extend(
        [
            "",
            "### Companion Validation-Split Results (`best_val` Checkpoint, `N = 11` Episodes / `1,020` Frames)",
            "",
            "| Setting | Vis. Tokens | Val Norm L1 $\\downarrow$ | Val TE Pos L2 (`mm`) $\\downarrow$ | Val TE Rot (`deg`) $\\downarrow$ | Val TE Gripper Acc (`%`) $\\uparrow$ | Val Grasp MAE (`frames`) $\\downarrow$ |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for preset in presets_in_order:
        if preset not in summary["runs"]:
            continue
        r = summary["runs"][preset]
        rc = r["run_config"]
        val_b = r["best_val_checkpoint"]["val"]
        cm = val_b["chunk_horizon_metrics"]
        tm = val_b["temporal_ensemble_metrics"]
        lines.append(
            f"| **`{preset}`** | `{rc['visual_tokens_per_step']}` | `{cm['norm_l1_active8d']:.4f}` | "
            f"`{tm['pos_l2_mm']:.2f}` | `{tm['rot_geodesic_deg']:.2f}` | "
            f"`{tm['gripper_acc_pct']:.2f}` | `{tm['grasp_timing_mae_frames']:.2f}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Test-Split TE Position Error (`mm`) Stratified by Scene Clutter (`bag_count = 1..5`)",
            "",
            "| Setting | `bag=1` (3 ep) | `bag=2` (3 ep) | `bag=3` (3 ep) | `bag=4` (1 ep) | `bag=5` (1 ep) | Overall (`mm`) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for preset in presets_in_order:
        if preset not in summary["runs"]:
            continue
        test_b = summary["runs"][preset]["best_val_checkpoint"]["test"]
        bbc = test_b["by_bag_count"]
        vals = [
            f"`{bbc[str(k)]['te_pos_l2_mm']:.2f}`" if str(k) in bbc else "—"
            for k in range(1, 6)
        ]
        overall = test_b["temporal_ensemble_metrics"]["pos_l2_mm"]
        lines.append(f"| **`{preset}`** | " + " | ".join(vals) + f" | **`{overall:.2f}`** |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Test-Split TE Error Stratified by Physical Phase",
            "",
            "| Setting | Approach (`>0.70`) Pos L2 (`mm`) | Closing (`0.05..0.70`) Pos L2 (`mm`) | Post-Grasp Lift (`<0.05`) Pos L2 (`mm`) | Closing Gripper Acc (`%`) |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for preset in presets_in_order:
        if preset not in summary["runs"]:
            continue
        bp = summary["runs"][preset]["best_val_checkpoint"]["test"]["by_phase"]
        ap = bp.get("approach_open", {}).get("pos_l2_mm", float("nan"))
        cl = bp.get("closing_transition", {}).get("pos_l2_mm", float("nan"))
        lf = bp.get("post_grasp_lift", {}).get("pos_l2_mm", float("nan"))
        cl_g = bp.get("closing_transition", {}).get("gripper_acc_pct", float("nan"))
        lines.append(f"| **`{preset}`** | `{ap:.2f}` | `{cl:.2f}` | `{lf:.2f}` | `{cl_g:.2f}` |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Decoder Cross-Attention Attribution Across Token Sources (Test Split `%`)",
            "",
            "> Note: `camera_left` is mounted on the **active right wrist**; `camera_right` is mounted on the **parked left wrist**.",
            "",
            "| Setting | `latent` | `proprio_state` | `camera_top` (global) | `camera_front` | `camera_left` (active right wrist) | `camera_right` (parked left wrist) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for preset in presets_in_order:
        if preset not in summary["runs"]:
            continue
        attn = summary["runs"][preset]["best_val_checkpoint"]["test"]["cross_attention"].get(
            "chunk_mean_pct", {}
        )
        lat = f"`{attn.get('latent_pct', 0.0):.2f}%`"
        st = f"`{attn.get('proprio_state_pct', 0.0):.2f}%`"
        c_top = f"`{attn[CAM_TOP]:.2f}%`" if CAM_TOP in attn else "—"
        c_front = f"`{attn[CAM_FRONT]:.2f}%`" if CAM_FRONT in attn else "—"
        c_left = f"`{attn[CAM_LEFT]:.2f}%`" if CAM_LEFT in attn else "—"
        c_right = f"`{attn[CAM_RIGHT]:.2f}%`" if CAM_RIGHT in attn else "—"
        lines.append(f"| **`{preset}`** | {lat} | {st} | {c_top} | {c_front} | {c_left} | {c_right} |")

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-camera ACT ablation suite on STEA_pick_success")
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=list(CAMERA_PRESETS.keys()),
    )
    parser.add_argument("--gpus", nargs="+", default=["0", "1"], help="CUDA device IDs for worker pool")
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--log-freq", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--spatial-folding", action="store_true", help="Fold diagonal camera feature quadrants")
    parser.add_argument(
        "--camera-overlay",
        action="store_true",
        help="Channel-stack camera pairs into 6-channel ResNet18 conv1 in two- or four-camera presets",
    )
    parser.add_argument("--swin-after-tokenization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sep-swin-for-views",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use an independent Swin block per visual stream (default when Swin is enabled)",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--dof-mode", type=str, default="8d", choices=["8d", "16d"])
    parser.add_argument("--gripper-mode", type=str, default="binary", choices=["binary", "continuous"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--suite-dir", type=str, default="")
    parser.add_argument("--write-policy-md", action="store_true", help="Also write timestamped MD under policy/")
    args = parser.parse_args()
    overlay_presets = [preset for preset in DEFAULT_PRESET_ORDER if len(CAMERA_PRESETS[preset]) in (2, 4)]
    if args.presets is None:
        args.presets = overlay_presets if args.camera_overlay else DEFAULT_PRESET_ORDER
    if args.camera_overlay:
        if any(preset not in overlay_presets for preset in args.presets):
            parser.error("--camera-overlay requires 2cams_top_left or all_4cams")
    if args.sep_swin_for_views and not args.swin_after_tokenization:
        parser.error("--sep-swin-for-views requires --swin-after-tokenization")
    if args.sep_swin_for_views is None:
        args.sep_swin_for_views = args.swin_after_tokenization
    return args


def main() -> None:
    args = parse_args()
    validate_splits()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = (
        Path(args.suite_dir)
        if args.suite_dir
        else REPO_ROOT / "outputs" / "STEA_pick_success" / f"act_camera_ablation_{timestamp}"
    )
    suite_dir.mkdir(parents=True, exist_ok=True)

    # Pre-warm video memmap caches for all requested cameras once before spawning workers
    needed_cams: list[str] = []
    for p in args.presets:
        for ck in CAMERA_PRESETS[p]:
            if ck not in needed_cams:
                needed_cams.append(ck)
    print(f"[suite] Pre-warming sequential video frame caches for {len(needed_cams)} camera streams ...")
    ensure_video_frame_cache(
        dataset_root=args.dataset_root,
        camera_keys=needed_cams,
        cache_dir=args.cache_dir,
    )

    queue = list(args.presets)
    free_gpus = list(args.gpus)
    running: list[dict[str, Any]] = []
    finished: list[str] = []
    failed: list[tuple[str, int]] = []

    t0 = time.time()
    print(
        f"[suite] Launching {len(queue)} ACT camera-ablation runs across GPUs {free_gpus} -> {suite_dir}",
        flush=True,
    )

    while queue or running:
        while queue and free_gpus:
            preset = queue.pop(0)
            gpu_id = free_gpus.pop(0)
            run_dir = suite_dir / preset
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "train.log"
            log_file = open(log_path, "w", encoding="utf-8")

            cmd = [
                sys.executable,
                str(Path(__file__).with_name("train_act.py").resolve()),
                "--camera-preset", preset,
                "--dataset-root", args.dataset_root,
                "--cache-dir", args.cache_dir,
                "--output-dir", str(run_dir),
                "--dof-mode", args.dof_mode,
                "--gripper-mode", args.gripper_mode,
                "--chunk-size", str(args.chunk_size),
                "--batch-size", str(args.batch_size),
                "--steps", str(args.steps),
                "--warmup-steps", str(args.warmup_steps),
                "--eval-freq", str(args.eval_freq),
                "--log-freq", str(args.log_freq),
                "--lr", str(args.lr),
                "--lr-backbone", str(args.lr_backbone),
                "--num-workers", str(args.num_workers),
                "--seed", str(args.seed),
                "--device", f"cuda:{gpu_id}",
            ]
            cmd.append("--spatial-folding" if args.spatial_folding else "--no-spatial-folding")
            if not args.swin_after_tokenization:
                cmd.append("--no-swin-after-tokenization")
            if args.camera_overlay:
                cmd.append("--camera-overlay")
            if not args.sep_swin_for_views:
                cmd.append("--no-sep-swin-for-views")
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            running.append(
                {
                    "preset": preset,
                    "gpu": gpu_id,
                    "proc": proc,
                    "log_file": log_file,
                    "log_path": log_path,
                    "start_t": time.time(),
                }
            )
            print(
                f"[suite] Started `{preset}` on cuda:{gpu_id} (pid={proc.pid}) -> {log_path}",
                flush=True,
            )

        time.sleep(2.0)
        still_running: list[dict[str, Any]] = []
        for job in running:
            ret = job["proc"].poll()
            if ret is None:
                still_running.append(job)
            else:
                job["log_file"].close()
                free_gpus.append(job["gpu"])
                dur_min = (time.time() - job["start_t"]) / 60.0
                if ret == 0:
                    finished.append(job["preset"])
                    print(
                        f"[suite] Completed `{job['preset']}` on cuda:{job['gpu']} in {dur_min:.2f} min",
                        flush=True,
                    )
                else:
                    failed.append((job["preset"], ret))
                    print(
                        f"[suite] FAILED `{job['preset']}` on cuda:{job['gpu']} (exit={ret})! Check {job['log_path']}",
                        flush=True,
                    )
        running = still_running

    if failed:
        for preset, code in failed:
            tail = (suite_dir / preset / "train.log").read_text(encoding="utf-8")[-2000:]
            print(f"--- Failure tail for {preset} (exit={code}) ---\n{tail}")
        raise SystemExit(1)

    runs_data: dict[str, Any] = {}
    ordered_present = [p for p in DISPLAY_ORDER if p in finished] + [
        p for p in finished if p not in DISPLAY_ORDER
    ]
    for preset in ordered_present:
        res_path = suite_dir / preset / "eval_results.json"
        runs_data[preset] = json.loads(res_path.read_text(encoding="utf-8"))

    total_suite_sec = time.time() - t0
    summary = {
        "timestamp": timestamp,
        "suite_dir": str(suite_dir),
        "total_wall_clock_sec": round(total_suite_sec, 2),
        "presets": ordered_present,
        "runs": runs_data,
    }
    summary_json_path = suite_dir / f"ablation_summary_{timestamp}.json"
    save_json(summary, summary_json_path)

    md_text = format_markdown_report(summary, ordered_present, timestamp)
    suite_md_path = suite_dir / f"ablation_report_{timestamp}.md"
    suite_md_path.write_text(md_text, encoding="utf-8")
    print(f"[suite] Saved summary JSON : {summary_json_path}")
    print(f"[suite] Saved report MD    : {suite_md_path}")

    if args.write_policy_md:
        md_dir = REPO_ROOT / "policy" / "md"
        md_dir.mkdir(parents=True, exist_ok=True)
        policy_md_path = md_dir / f"STEA_act_camera_ablation_results_{timestamp}.md"
        policy_md_path.write_text(md_text, encoding="utf-8")
        print(f"[suite] Saved policy MD    : {policy_md_path}")

    print("\n" + md_text, flush=True)


if __name__ == "__main__":
    main()
