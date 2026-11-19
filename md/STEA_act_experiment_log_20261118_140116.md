# STEA ACT Multi-Camera Experiment Log (`20261118_140116`)

- **Related Analysis**: [`./STEA_dataset_distribution_and_act_analysis.md`](./STEA_dataset_distribution_and_act_analysis.md)
- **Stratified Splits**: [`./STEA_pick_success_splits.md`](./STEA_pick_success_splits.md)
- **Full Quantitative Report**: [`./STEA_act_camera_ablation_results_20261117_174606.md`](./STEA_act_camera_ablation_results_20261117_174606.md)
- **Artifacts Directory**: [`../../outputs/STEA_pick_success/act_camera_ablation_20261117_174606/`](../../outputs/STEA_pick_success/act_camera_ablation_20261117_174606/)

---

## 1. Exact Reproduction Commands

Working directory: `/data/xiuchao/biArm/DEM`  
Conda Python (`lerobot 0.4.4`, `PyTorch 2.7.1+cu126`, `2 × NVIDIA RTX A6000`): `/data/xiuchao/cache/conda/envs/openpi/bin/python`

### A. Full 5-Setting Camera-Ablation Suite (Exact Command Used for `20261117_174606`)

```bash
/data/xiuchao/cache/conda/envs/openpi/bin/python policy/run_camera_ablation.py \
  --gpus 0 1 \
  --steps 5000 \
  --warmup-steps 250 \
  --eval-freq 500 \
  --log-freq 250 \
  --batch-size 64 \
  --chunk-size 50 \
  --lr 8e-5 \
  --lr-backbone 2e-5 \
  --dof-mode 8d \
  --gripper-mode binary \
  --seed 42 \
  --write-policy-md
```

### B. Single Camera-Preset Training & Evaluation Example (`2cams_top_left`)

```bash
/data/xiuchao/cache/conda/envs/openpi/bin/python policy/train_act.py \
  --camera-preset 2cams_top_left \
  --device cuda:0 \
  --steps 5000 \
  --warmup-steps 250 \
  --eval-freq 500 \
  --log-freq 250 \
  --batch-size 64 \
  --chunk-size 50 \
  --lr 8e-5 \
  --lr-backbone 2e-5 \
  --dof-mode 8d \
  --gripper-mode binary \
  --seed 42
```

*(Available `--camera-preset` choices: `all_4cams`, `3cams_active`, `2cams_top_left`, `1cam_top`, `1cam_left`.)*

---

## 2. Experimental Setup & Data Alignment Fixes

1. **Dataset & Stratified Split**:
   - `STEA_pick_success` (`113` episodes, `10,675` frames @ `10 Hz`, `4 × 320×240` RGB).
   - Fixed stratified split (`bag_count = 1..5`): `91` Train (`8,685` frames) / `11` Val (`1,020` frames) / `11` Test (`970` frames).
2. **8-DoF Active Right-Arm Subspace (`--dof-mode 8d`)**:
   - Filters out the stationary left arm (`dims 0..6, 14`) so constant channels do not dilute `L1` supervision.
   - `observation.state` (`8D`): right-arm 7 joint angles (`rad`) + `2FEG_r_Joint1.pos`.
   - `action` (`8D`): right-arm Cartesian EE pose (`xyz` in meters + `qx,qy,qz,qw`) + reconstructed right gripper command.
3. **Gripper Command Reconstruction (`--gripper-mode binary`)**:
   - Raw `action[:, 15]` is identically `0.0` in `STEA_pick_success`. Cross-checking `DEM_pickplace` and `contracts/dem_pickplace.yaml` confirms `aibot2` expects a binary gripper command (`0.0 = open`, `1.0 = closed`), opposite in polarity to `observation.state[:, 15]` (`~0.80 = open`, `~0.00 = closed`).
   - Reconstructed per episode via `action_grip[t] = float(state[min(t + 1, T - 1), 15] < 0.70)`.
4. **Leak-Free Normalization & Bit-Exact Video Cache**:
   - `MEAN_STD` statistics for `observation.state` and `action` are computed strictly on the `91` training episodes (`policy/stea_act_dataset.py:compute_train_stats`); RGB inputs use ImageNet normalization to match `ResNet18` `FrozenBatchNorm2d`.
   - Sequential PyAV decoding caches read-only `uint8` CHW memmaps under `outputs/STEA_pick_success/act_frame_cache/` (`max_abs_diff == 0.0` vs `LeRobotDataset`), accelerating training from `2.4 s/sample` (AV1 random seek) to `>3.1 it/s` (`batch_size=64`, `bf16` AMP).

---

## 3. Code $\leftrightarrow$ Method Mapping

| Component | Code Location | Method Details |
| :--- | :--- | :--- |
| **Splits & Subspace** | `policy/splits_and_configs.py` | Defines `SPLITS`, `EPISODE_BAG_COUNT`, `CAMERA_PRESETS`, and `8d`/`16d` index slices |
| **Dataset & Chunking** | `policy/stea_act_dataset.py` | Builds length-`k` (`chunk_size=50`) target chunks with episode-end clamping + `action_is_pad` mask |
| **ACT CVAE Training** | `policy/train_act.py` | Instantiates `ACTPolicy` (`ResNet18` + 4-layer Encoder + 1-layer Decoder + 4-layer CVAE Encoder, `51.58M` params across all camera presets); optimizes masked `L1 + 10.0 * KL` with AdamW + warmup cosine schedule; selects `best_val` strictly by validation `norm_l1` |
| **Offline Rollout Eval** | `policy/evaluate_act.py` | Evaluates deterministic `z = 0` inference across (1) 50-step chunk horizon, (2) step-0, and (3) sequential **Temporal Ensembling (`coeff=0.01`)**, reporting 3D EE `pos_l2_mm`, SO(3) geodesic `rot_deg`, `gripper_acc / F1`, and `grasp_timing_mae_frames` |
| **Cross-Attention Hook** | `policy/evaluate_act.py` (`CrossAttentionRecorder`) | Hooks `decoder.layers[-1].multihead_attn` (`(B, 50, 2 + 80 * N_cam)`) and sums probability mass over `idx 0` (`latent`), `idx 1` (`proprio_state`), and each camera's `80`-token slice (`10×8` feature map) |

---

## 4. Headline Results Summary (`best_val` Checkpoint on Held-Out Test Split)

| Setting | Cameras | Vis. Tokens | Train Time | Test Norm L1 $\downarrow$ | Test TE Pos L2 (`mm`) $\downarrow$ | Test TE Rot (`deg`) $\downarrow$ | Closing Pos L2 (`mm`) $\downarrow$ | Lift Pos L2 (`mm`) $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`all_4cams`** | `top + front + left + right` | `320` | `26.9m` | `0.3285` | `40.01` | `6.76°` | `54.26` | `43.79` |
| **`3cams_active`** | `top + front + left` | `240` | `21.5m` | `0.3198` | `39.68` | `6.86°` | `54.54` | `38.41` |
| **`2cams_top_left`** | **`top + left` (active wrist)** | **`160`** | **`14.4m`** | **`0.3095`** | **`37.82`** | **`6.36°`** | **`51.50`** | **`37.43`** |
| **`1cam_top`** | `top` | `80` | `8.3m` | `0.3203` | `39.39` | `6.66°` | `54.16` | `38.30` |
| **`1cam_left`** | `left` (active wrist) | `80` | `8.2m` | `0.3455` | `41.42` | `6.35°` | `60.03` | `49.73` |

- **Takeaway**: `2cams_top_left` (`camera_top` + active right-wrist `camera_left`) achieves the best test generalization while cutting visual tokens by `50%` and training time by `46.5%`. Cross-attention attribution confirms that in `all_4cams`, the stationary parked-arm camera (`camera_right`) absorbs `22.52%` of decoder attention, inducing overfitting and degrading closing-phase gripper accuracy (`88.17%` vs `92.47%` without `camera_right`).
