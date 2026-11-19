# Empirical Data Distribution & Partial-Observability Analysis (`STEA_pick_success`)

- **Dataset Root**: [`../data_STEA/EVAL_24Sep_25Sep_STEA_pick_success/`](../data_STEA/EVAL_24Sep_25Sep_STEA_pick_success/)
- **Companion Failure Split**: [`../data_STEA/EVAL_24Sep_25Sep_STEA_pick_failure/`](../data_STEA/EVAL_24Sep_25Sep_STEA_pick_failure/) (`101` episodes, `10,081` frames; overall evaluation pool = `214` episodes, `52.80%` success)
- **Stratified Splits Reference**: [`./STEA_pick_success_splits.md`](./STEA_pick_success_splits.md)
- **Format**: LeRobot `v3.0` (`113` episodes, `10,675` frames, `10 Hz`, total duration `1,067.5 s` / `17.79 min`, `4 × 320×240` AV1 RGB streams)

---

## 1. Episode Horizon & Temporal Phase Distribution

Across all 113 successful episodes, the robot approaches, grasps, and lifts the **rightmost** bagged object from a flat stainless-steel tabletop:

| Metric | Mean ± Std | Min | P10 | Median (P50) | P90 | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Episode Length (frames @ 10 Hz)** | `94.47 ± 22.91` | `52` | `70.0` | `92.0` | `117.6` | `238` |
| **Episode Duration (seconds)** | `9.45 ± 2.29` | `5.2` | `7.0` | `9.2` | `11.8` | `23.8` |
| **Post-Grasp Keyframe (`offset +5`)** | `62.28 ± 11.99` | `27` | `45.4` | `63.0` | `76.0` | `90` |
| **Relative Grasp Timing (`t_grasp / T`)** | `0.668 ± 0.079` | `0.370` | `0.596` | `0.683` | `0.744` | `0.829` |

### Episode Length by Scene Clutter (`bag_count`)

| `bag_count` | Episodes | Ratio | Mean ± Std (frames) | Min | Median | Max | `task_index = 0` | `task_index = 1` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 30 | 26.55% | `102.53 ± 14.56` | 82 | 100.5 | 144 | 30 | 0 |
| **2** | 32 | 28.32% | `89.44 ± 12.89` | 54 | 90.0 | 111 | 16 | 16 |
| **3** | 34 | 30.09% | `88.12 ± 20.71` | 52 | 87.0 | 159 | 26 | 8 |
| **4** | 12 | 10.62% | `110.00 ± 46.66` | 58 | 103.5 | 238 | 10 | 2 |
| **5** | 5 | 4.42% | `84.20 ± 21.49` | 59 | 92.0 | 110 | 5 | 0 |
| **All** | **113** | **100%** | **`94.47 ± 22.91`** | **52** | **92.0** | **238** | **87 (77.0%)** | **26 (23.0%)** |

> **Task Metadata Note**: [`meta/tasks.parquet`](../data_STEA/EVAL_24Sep_25Sep_STEA_pick_success/meta/tasks.parquet) contains two instruction strings (`task_index = 0`: `"Pick object from table and put it inside the box"` for `ep_000`–`ep_086`, mean `94.26` frames; `task_index = 1`: `"pick rightmost object on flat surface"` for `ep_087`–`ep_112`, mean `95.15` frames). Visual inspection confirms **both subsets execute the exact same physical behavior**: grasping and lifting the rightmost bag off the flat table (no box is present in the workspace).

---

## 2. Multi-Camera Anatomy & Partial Observability

Note the **inverted wrist-camera naming** defined in [`../datasets.yaml`](../datasets.yaml):
- `observation.images.camera_left` $\rightarrow$ mounted on the **active right arm** (`ZPF_right`).
- `observation.images.camera_right` $\rightarrow$ mounted on the **idle left arm** (`ZPF_left`).

| Camera Stream Key | Physical Mount | Gripper Visibility | Target-Bag Coverage | Role / Failure Mode |
| :--- | :--- | :--- | :--- | :--- |
| `observation.images.camera_top` | Fixed overhead | Only when right arm descends into left/center-right workspace; **absent during home & post-lift** | Covers ~85–90% of table width; **clips rightmost bags** (`x > 95%`) in ~15 episodes (e.g., `ep_062`, `ep_083`–`ep_086`) and suffers occasional front-to-back ray alignment (`ep_089`) | **Task-relevant (Global)**: resolves global scene clutter and identifies relative object ordering |
| `observation.images.camera_left` | **Right-arm wrist** (active arm) | **100% of frames** (both fingers continuously in frame) | High bird's-eye coverage of right workspace at `t = 0` (covering `camera_top`'s right blind spot); transitions to local grasp/lift view | **Task-relevant (Local + Blind-Spot)**: only camera observing fine bag-in-finger contact and post-lift state |
| `observation.images.camera_front` | Fixed front-angled | Rarely sees gripper on far-right grasps; **absent after lift** | Misses right ~25% of tabletop (`ep_021`, `ep_025`, `ep_061`, `ep_062`, etc.); largely redundant with left/center of `camera_top` | **Partially Redundant**: adds front depth cue for left/mid clutter, but misses rightmost targets |
| `observation.images.camera_right` | **Left-arm wrist** (parked arm) | Shows only the **stationary left gripper** (never the active right gripper) | Stares at empty left tabletop throughout `t = 0..T` (occasionally clips leftmost distractor bag edge) | **Pure Distractor / Noise**: near-static image across entire episode |

### Why Partial Observability Matters for ACT

In ACT (`ResNet18` backbone, stride 32), each `320×240` RGB frame produces a `10×8 = 80` spatial token grid concatenated across cameras into the Transformer encoder/decoder:

- **4 Cameras (`top + front + left + right`)**: `4 × 80 = 320` visual tokens/step (`50%` from static `camera_right` and partially redundant/blind `camera_front`).
- **2 Cameras (`top + left`)**: `2 × 80 = 160` visual tokens/step, forming a **jointly sufficient, complementary pair**:
  1. At `t = 0` (home pose), `camera_top` + elevated `camera_left` jointly cover **100% of all bags** on the table without blind spots.
  2. During descent, grasp, and lift (`t ≈ 45..T`), `camera_left` continuously tracks gripper-to-bag alignment and lift completion even when the active gripper exits `camera_top` and `camera_front`.
  3. With only **91 training episodes (`8,685` frames)**, halving the visual token sequence eliminates static/distractor tokens and reduces overfitting / causal confusion in cross-attention.

---

## 3. Proprioceptive State & Action Distributions (Critical Pre-Training Findings)

Inspection of all `10,675` frames in [`data/chunk-000/file-000.parquet`](../data_STEA/EVAL_24Sep_25Sep_STEA_pick_success/data/chunk-000/file-000.parquet) reveals **three structural properties** that must be accounted for before training ACT:

### Finding A: Single-Arm Execution Inside a 16-DoF Bimanual Schema
The dataset records 16D `observation.state` and 16D `action`, but **the left arm (`dims 0..6` + `dim 14`) is completely parked** across all 113 episodes (`position std ≤ 0.0041 m`, `joint std ≤ 0.0228 rad`). Only the right arm (`dims 7..13` + `dim 15`) moves.

### Finding B: Heterogeneous State vs. Action Spaces
- **`observation.state` (`16D`)**: **Joint-space positions** (`7` left joints + `7` right joints in radians + `2` gripper joint positions).
- **`action` (`16D`)**: **Cartesian end-effector poses** (`left_arm` `xyz + qx,qy,qz,qw` + `right_arm` `xyz + qx,qy,qz,qw` + `2` gripper commands; right quaternion norm $\in [0.9781, 1.0116]$).

### Finding C: Zeroed Gripper Action Channels (`action[:, 14:16] == 0`)
Across **all 10,675 frames**, `action[:, 14]` (`left_gripper.pos`) and `action[:, 15]` (`right_gripper.pos`) are **identically `0.0000`** (`min = 0.0`, `max = 0.0`, `std = 0.0`)!
- The true gripper motion is recorded in **`observation.state[:, 15]` (`2FEG_r_Joint1.pos`)**, spanning `[0.0000, 0.8041]`:
  - **Open (`> 0.70`)**: `51.74%` of frames (approach phase)
  - **Transition (`0.05` – `0.70`)**: `8.78%` of frames (closing)
  - **Closed (`< 0.05`)**: `39.48%` of frames (grasp & lift phase; post-grasp occurs around `66.8%` of episode length)
- Meanwhile, `observation.state[:, 14]` (`2FEG_l_Joint1.pos`) stays parked at `0.7959 ± 0.0001` (open).
- **Required Fix Before Training ACT**: Populate the target right-gripper action channel (e.g., from `observation.state[t + 1, 15]` or binarized open/close target matching the robot controller), and optionally restrict supervision/evaluation to the active **8-DoF right-arm subspace** (`action[:, 7:14]` + reconstructed `right_gripper`) so `L1` loss is not diluted by 8 constant dimensions.

---

### Per-Dimension Summary Table (`N = 10,675` frames)

#### `action` (`float32[16]`, Cartesian EE Pose + Gripper)

| Dim | Feature Name | Min | Max | Mean | Std | Status |
| :-: | :--- | :---: | :---: | :---: | :---: | :--- |
| 0 | `left_arm.x` | `0.5804` | `0.5967` | `0.5914` | `0.0025` | Parked |
| 1 | `left_arm.y` | `0.3138` | `0.3268` | `0.3220` | `0.0015` | Parked |
| 2 | `left_arm.z` | `1.0244` | `1.0451` | `1.0352` | `0.0041` | Parked |
| 3 | `left_arm.qx` | `-0.0110` | `0.0042` | `-0.0065` | `0.0014` | Parked |
| 4 | `left_arm.qy` | `-0.0468` | `-0.0297` | `-0.0395` | `0.0044` | Parked |
| 5 | `left_arm.qz` | `-0.0147` | `0.0304` | `0.0213` | `0.0032` | Parked |
| 6 | `left_arm.qw` | `0.9984` | `0.9993` | `0.9989` | `0.0002` | Parked |
| 7 | `right_arm.x` | `0.5308` | `0.8966` | `0.6894` | `0.0888` | **Active** |
| 8 | `right_arm.y` | `-0.4516` | `-0.0886` | `-0.2833` | `0.0690` | **Active** |
| 9 | `right_arm.z` | `0.9514` | `1.1470` | `1.0369` | `0.0412` | **Active** |
| 10 | `right_arm.qx` | `-0.0788` | `0.0456` | `-0.0151` | `0.0214` | **Active** |
| 11 | `right_arm.qy` | `-0.1933` | `0.1495` | `-0.0159` | `0.0675` | **Active** |
| 12 | `right_arm.qz` | `-0.2560` | `0.4784` | `0.0784` | `0.1162` | **Active** |
| 13 | `right_arm.qw` | `0.8730` | `1.0015` | `0.9870` | `0.0160` | **Active** |
| 14 | `left_gripper.pos` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | **Zeroed** |
| 15 | `right_gripper.pos` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | **Zeroed (use `state[15]`)** |

#### `observation.state` (`float32[16]`, Joint Positions + Gripper)

| Dim | Feature Name | Min | Max | Mean | Std | Status |
| :-: | :--- | :---: | :---: | :---: | :---: | :--- |
| 0–6 | `ZPF_left_Joint1..7.pos` | `-1.9635` | `1.1617` | — | `0.0062`–`0.0228` | Parked |
| 7 | `ZPF_right_Joint1.pos` | `-0.5011` | `1.1126` | `0.1460` | `0.3409` | **Active** |
| 8 | `ZPF_right_Joint2.pos` | `0.0745` | `0.9568` | `0.4993` | `0.1524` | **Active** |
| 9 | `ZPF_right_Joint3.pos` | `-0.5179` | `0.2315` | `-0.1503` | `0.1449` | **Active** |
| 10 | `ZPF_right_Joint4.pos` | `-2.2009` | `-0.0600` | `-1.5280` | `0.4135` | **Active** |
| 11 | `ZPF_right_Joint5.pos` | `-1.5046` | `-0.6118` | `-1.0790` | `0.1301` | **Active** |
| 12 | `ZPF_right_Joint6.pos` | `-0.8672` | `0.7273` | `-0.0039` | `0.2779` | **Active** |
| 13 | `ZPF_right_Joint7.pos` | `-0.4888` | `0.4748` | `-0.0109` | `0.1824` | **Active** |
| 14 | `2FEG_l_Joint1.pos` | `0.7951` | `0.7959` | `0.7959` | `0.0001` | Parked (Open) |
| 15 | `2FEG_r_Joint1.pos` | `0.0000` | `0.8041` | `0.4467` | `0.3779` | **Active (`0`=closed, `0.8`=open)** |

---

## 4. Recommended Camera-View Ablation Matrix for ACT

Using `huggingface/lerobot` (`v0.4+`) on the fixed stratified split (`91` train / `11` val / `11` test in [`./STEA_pick_success_splits.md`](./STEA_pick_success_splits.md)):

| Setting ID | Camera Streams Included | Visual Tokens | Scientific Question Tested |
| :--- | :--- | :---: | :--- |
| **`all_4cams`** | `top + front + left + right` | `320` | Default multi-view baseline; measures distraction from static `camera_right` & redundant `camera_front` |
| **`3cams_active`** | `top + front + left` | `240` | Isolates the impact of removing the static idle-arm wrist camera (`camera_right`) |
| **`2cams_top_left`** | **`top + left`** | **`160`** | **Core Hypothesis**: minimal task-sufficient complementary pair (global layout + active right wrist) |
| **`1cam_top`** | `top` | `80` | Quantifies degradation from right-edge blind spots (`x > 95%`) and missing local grasp feedback |
| **`1cam_left`** | `left` | `80` | Quantifies wrist-only performance (initial bird's-eye view, loss of global reference after descent) |

---

## 5. Next-Session Implementation Roadmap (`policy/`)

1. **Environment Setup**:
   - Install `huggingface/lerobot` (`>= 0.4.4`, editable install `pip install -e .`) so `LeRobotDataset` (`v3.0`) and `ACTPolicy` (`lerobot/policies/act/modeling_act.py`) run natively and allow optional cross-attention weight hooks.
2. **Gripper Label & Action-Space Wrapper**:
   - Inspect/confirm downstream robot controller expectation for `right_gripper.pos` and reconstruct target gripper actions from `observation.state[:, 15]` (`2FEG_r_Joint1.pos`, e.g., `state[t + 1, 15]` or binarized open/close command) to replace the zeroed `action[:, 15]` channel.
   - Support switching between full `16-DoF` action supervision and active right-arm `8-DoF` supervision (`action[:, 7:14]` + reconstructed right gripper), and decide whether `observation.state` input keeps all `16` dims or active right-arm `8` dims (`state[:, 7:14]` + `state[:, 15]`).
3. **Configurable Camera-Subset Training & Evaluation Pipeline in `policy/`**:
   - Load exact `train (91)` / `val (11)` / `test (11)` episode lists from [`./STEA_pick_success_splits.md`](./STEA_pick_success_splits.md).
   - Parameterize `camera_keys` (`all_4cams`, `3cams_active`, `2cams_top_left`, `1cam_top`, `1cam_left`) via `ACTConfig.input_features` without duplicating dataset files.
   - Implement offline `val` / `test` evaluation (overall & stratified by `bag_count = 1..5`, separating arm pose `L1/MSE` and gripper accuracy, plus optional per-camera attention attribution).

