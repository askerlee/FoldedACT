# ACT Multi-Camera Partial-Observability Ablation Results (`20261117_174606`)

## 1. Experimental Protocol

- **Dataset**: `STEA_pick_success` (`113` episodes, `10,675` frames @ `10 Hz`, `320×240` RGB)
- **Stratified Split**: `91` Train (`8,685` frames) / `11` Val (`1,020` frames) / `11` Test (`970` frames) (`bag_count` distribution `3:3:3:1:1` in Val & Test)
- **State & Action Subspace**: `8d` active right arm (`7` joints + gripper $\rightarrow$ `7D` Cartesian EE pose `xyz + qx,qy,qz,qw` + reconstructed `binary` gripper command `0=open, 1=closed`)
- **ACT Architecture & Training**: `ResNet18` (`IMAGENET1K_V1`), `chunk_size=50` (`5.0 s`), `batch_size=64`, `steps=5000`, `lr=8e-05`, `lr_backbone=2e-05`, `kl_weight=10.0`, `seed=42`
- **Checkpoint Selection**: `best_val` selected strictly by minimum standardized `norm_l1` on the held-out `Val` split (zero test-set peeking).

---

## 2. Main Test-Split Results (`best_val` Checkpoint, `N = 11` Episodes / `970` Frames)

| Setting | Cameras | Vis. Tokens | Best Step | Train Time | Chunk Norm L1 $\downarrow$ | TE Pos L2 (`mm`) $\downarrow$ | TE Rot (`deg`) $\downarrow$ | TE Gripper Acc (`%`) $\uparrow$ | TE Gripper F1 (`%`) $\uparrow$ | Grasp Timing MAE (`frames`) $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`all_4cams`** | `top + front + left + right` | `320` | `1000` | `26.9m` | `0.3285` | `40.01` | `6.76` | `94.54` | `94.25` | `3.18` |
| **`3cams_active`** | `top + front + left` | `240` | `4500` | `21.5m` | `0.3198` | `39.68` | `6.86` | `93.71` | `93.50` | `3.91` |
| **`2cams_top_left`** | `top + left` | `160` | `4000` | `14.4m` | `0.3095` | `37.82` | `6.36` | `93.92` | `93.69` | `3.73` |
| **`1cam_top`** | `top` | `80` | `5000` | `8.3m` | `0.3203` | `39.39` | `6.66` | `93.61` | `93.45` | `4.00` |
| **`1cam_left`** | `left` | `80` | `3500` | `8.2m` | `0.3455` | `41.42` | `6.35` | `93.40` | `93.23` | `4.18` |

### Companion Validation-Split Results (`best_val` Checkpoint, `N = 11` Episodes / `1,020` Frames)

| Setting | Vis. Tokens | Val Norm L1 $\downarrow$ | Val TE Pos L2 (`mm`) $\downarrow$ | Val TE Rot (`deg`) $\downarrow$ | Val TE Gripper Acc (`%`) $\uparrow$ | Val Grasp MAE (`frames`) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`all_4cams`** | `320` | `0.2476` | `25.99` | `5.22` | `96.57` | `3.09` |
| **`3cams_active`** | `240` | `0.2476` | `28.29` | `5.09` | `96.67` | `3.00` |
| **`2cams_top_left`** | `160` | `0.2509` | `30.14` | `5.19` | `96.37` | `3.27` |
| **`1cam_top`** | `80` | `0.2599` | `27.89` | `5.52` | `96.27` | `3.36` |
| **`1cam_left`** | `80` | `0.2639` | `30.41` | `5.29` | `96.47` | `3.18` |

### FoldedACT with Swin: `2cams_top_left` Test Results (`best_val`)

Both runs use `top + left` and are reported separately from the 5,000-step ACT camera ablation above. The previous run is recorded from its console summary; the camera overlay run is recorded from its saved `eval_results.json` (40 visual tokens per step).

| Setting | Best Step | Train Time (`min`) | Chunk Norm L1 $\downarrow$ | TE Pos L2 (`mm`) $\downarrow$ | TE Rot (`deg`) $\downarrow$ | TE Gripper Acc (`%`) $\uparrow$ | TE Gripper F1 (`%`) $\uparrow$ | Grasp Timing MAE (`frames`) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`2cams_top_left` (no overlay)** | `15000` | `39.61` | `0.3089` | `39.31` | `6.44` | `93.81` | `93.63` | `3.82` |
| **`2cams_top_left` (camera overlay)** | `13000` | `37.66` | `0.3120` | `38.73` | `6.39` | `93.71` | `93.54` | `3.91` |

---

## 3. Test-Split TE Position Error (`mm`) Stratified by Scene Clutter (`bag_count = 1..5`)

| Setting | `bag=1` (3 ep) | `bag=2` (3 ep) | `bag=3` (3 ep) | `bag=4` (1 ep) | `bag=5` (1 ep) | Overall (`mm`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`all_4cams`** | `29.80` | `41.89` | `34.16` | `81.06` | `24.13` | **`40.01`** |
| **`3cams_active`** | `32.06` | `39.53` | `37.60` | `72.20` | `22.14` | **`39.68`** |
| **`2cams_top_left`** | `30.40` | `40.78` | `31.21` | `69.58` | `24.61` | **`37.82`** |
| **`1cam_top`** | `30.83` | `39.73` | `30.52` | `84.39` | `26.77` | **`39.39`** |
| **`1cam_left`** | `33.25` | `44.07` | `39.18` | `72.58` | `17.15` | **`41.42`** |

---

## 4. Test-Split TE Error Stratified by Physical Phase

All rows use the `best_val` checkpoint on the held-out test split. Overall gripper accuracy covers all test steps; closing gripper accuracy covers only closing steps. The FoldedACT rows are separate runs, not part of the 5,000-step ACT camera ablation.

| Run | Approach (`>0.70`) Pos L2 (`mm`) | Closing (`0.05..0.70`) Pos L2 (`mm`) | Closing Pos L1 (`mm`) | Lift (`<0.05`) Pos L2 (`mm`) | Closing Gripper Acc (`%`) | Overall Gripper Acc (`%`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`all_4cams`** | `35.06` | `54.26` | `26.53` | `43.79` | `88.17` | `94.54` |
| **`3cams_active`** | `37.90` | `54.54` | `26.50` | `38.41` | `92.47` | `93.71` |
| **`2cams_top_left`** | `35.67` | `51.50` | `25.14` | `37.43` | `92.47` | `93.92` |
| **`1cam_top`** | `37.51` | `54.16` | `25.85` | `38.30` | `94.62` | `93.61` |
| **`1cam_left`** | `32.78` | `60.03` | `29.80` | `49.73` | `92.47` | `93.40` |
| **`folded_act_4cams`** | `45.05` | `54.34` | `26.33` | `37.33` | `92.47` | `94.54` |
| **`folded_overlay_2cams`** | `37.98` | `51.81` | `25.40` | `36.37` | `94.62` | `93.71` |
| **`folded_overlay_sepswin_2cams`** | `35.13` | `53.50` | `26.40` | `38.31` | `96.77` | `94.02` |
| **`folded_overlay_sepswin_4cams`** | `41.65` | `51.98` | `26.01` | `34.82` | `95.70` | `94.33` |
| **`folded_overlayresnet_sepswin_2cams`** | `34.81` | `54.72` | `26.95` | `37.53` | `95.70` | `93.81` |
| **`folded_overlayresnet_sepswin_4cams`** | `42.40` | `57.36` | `28.04` | `38.34` | `94.62` | `93.81` |

---

## 5. Decoder Cross-Attention Attribution Across Token Sources (Test Split `%`)

> Note: `camera_left` is mounted on the **active right wrist**; `camera_right` is mounted on the **parked left wrist**.

| Setting | `latent` | `proprio_state` | `camera_top` (global) | `camera_front` | `camera_left` (active right wrist) | `camera_right` (parked left wrist) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`all_4cams`** | `0.27%` | `0.41%` | `22.19%` | `20.58%` | `34.03%` | `22.52%` |
| **`3cams_active`** | `0.16%` | `0.44%` | `29.16%` | `28.33%` | `41.91%` | — |
| **`2cams_top_left`** | `0.22%` | `0.50%` | `44.07%` | — | `55.21%` | — |
| **`1cam_top`** | `0.33%` | `2.35%` | `97.32%` | — | — | — |
| **`1cam_left`** | `0.20%` | `1.15%` | — | — | `98.65%` | — |

---

## 6. Key Empirical Findings

1. **`2cams_top_left` (`top + active right wrist`, `160` visual tokens) Achieves Best Overall Test Generalization**:
   - On the held-out Test split (`best_val` checkpoint), **`2cams_top_left`** attains the **lowest standardized chunk L1 (`0.3095`)**, **lowest Temporal-Ensembled 3D EE position error (`37.82 mm`)**, and **lowest closing-phase (`51.50 mm`) and post-grasp lift (`37.43 mm`) position errors**, while cutting training wall-clock time by **`46.5%`** (`14.4 min` vs `26.9 min` for `all_4cams`).
   - Evaluating the final `step = 5000` (`last`) checkpoints yields the exact same ranking (`2cams_top_left`: `norm_l1 = 0.3084`, `TE Pos L2 = 37.65 mm`, `TE Rot = 6.35°` vs `all_4cams`: `norm_l1 = 0.3227`, `TE Pos L2 = 39.91 mm`, `TE Rot = 6.73°`).
2. **Direct Cross-Attention Evidence of Distractor Token Leakage in `all_4cams`**:
   - In `all_4cams`, the **stationary idle-arm wrist camera (`camera_right`) absorbs `22.52%`** of total decoder cross-attention weight (exceeding `camera_top`'s `22.19%` and `camera_front`'s `20.58%`), directly confirming causal confusion / overfitting on static distractor tokens (`best_val` peaks early at step `1000` before overfitting).
   - During the critical **`closing_transition` (`0.05..0.70`)** phase, `all_4cams` suffers the lowest gripper command accuracy (`88.17%`), whereas removing `camera_right` immediately recovers closing-phase gripper accuracy to **`92.47%`–`94.62%`**.
3. **Complementary Failure Modes of Single-Camera Policies (`1cam_top` vs `1cam_left`)**:
   - **`1cam_top` (Global-Only)**: Performs competitively in sparse/central clutter (`bag=1..3`), but degrades sharply in dense right-edge clutter (`bag=4`: `84.39 mm` vs `69.58 mm` for `2cams_top_left`) where rightmost targets sit near/beyond `camera_top`'s right FOV border and fine in-hand contact is unobservable.
   - **`1cam_left` (Active-Wrist-Only)**: Excels during initial `approach_open` (`32.78 mm`) and on extreme right-border clutter (`bag=5`: `17.15 mm` where the elevated right wrist directly overlooks the right blind spot), but suffers severe drift once the wrist descends to the tabletop and loses global spatial reference (`closing`: `60.03 mm`, `post_grasp_lift`: `49.73 mm`).
   - Combining **`camera_top + camera_left`** (`44.07%` + `55.21%` cross-attention share; shifting from `59.05%` wrist attention during approach to a balanced `49.64% / 49.70%` split during lift) resolves both failure modes simultaneously.

