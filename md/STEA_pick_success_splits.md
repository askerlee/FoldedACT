# STEA Pick Success: Bag Count Statistics & Stratified Splits (80% / 10% / 10%)

- **Dataset**: [`EVAL_24Sep_25Sep_STEA_pick_success`](../data_STEA/EVAL_24Sep_25Sep_STEA_pick_success/)
- **Metadata**: [`meta/info.json`](../data_STEA/EVAL_24Sep_25Sep_STEA_pick_success/meta/info.json) (`total_episodes = 113`, `total_frames = 10,675`, `fps = 10`)
- **Keyframes Source**: [`keyframes_STEA_post_grasp_offset5_3cams_261001`](../outputs/STEA_pick_success/keyframes_STEA_post_grasp_offset5_3cams_261001/)

---

## 1. Bag Count Distribution Across 113 Episodes (`ep_000`–`ep_112`)

Bag counts were annotated primarily from `camera_top`, cross-checked with `camera_front`, `camera_left` (right-arm wrist camera), and `frame 0` of the raw videos for edge-of-FOV / line-of-sight occlusion cases (e.g., `ep_062` where `camera_top` appears empty while the gripper holds 1 bag just past the right border; `ep_089` where two bags align front-to-back along `camera_top`'s viewing ray; and `ep_083`–`ep_086` where the rightmost bag sits beyond `camera_top`'s right border for a total of 5 bags).

| `bag_count` | Episodes | Ratio | `episode_index` List |
| :---: | :---: | :---: | :--- |
| **1** | **30** | 26.55% | `[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 49, 50, 59, 60, 61, 62, 72]` |
| **2** | **32** | 28.32% | `[11, 12, 46, 48, 54, 55, 56, 57, 58, 63, 64, 65, 66, 73, 74, 75, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]` |
| **3** | **34** | 30.09% | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 51, 52, 53, 71, 87, 88, 89, 90, 91, 92, 93, 96]` |
| **4** | **12** | 10.62% | `[67, 68, 69, 70, 76, 77, 78, 80, 81, 82, 94, 95]` |
| **5** | **5** | 4.42% | `[79, 83, 84, 85, 86]` |
| **Total** | **113** | **100%** | `0` – `112` (mean `2.381` bags/episode) |

### Contiguous Episode Segments by `bag_count`

| Episode Range | Count | `bag_count` | `task_index` |
| :--- | :---: | :---: | :---: |
| `ep_000` – `ep_010` | 11 | **3** | 0 |
| `ep_011` – `ep_012` | 2 | **2** | 0 |
| `ep_013` – `ep_035` | 23 | **1** | 0 |
| `ep_036` – `ep_045` | 10 | **3** | 0 |
| `ep_046` | 1 | **2** | 0 |
| `ep_047` | 1 | **3** | 0 |
| `ep_048` | 1 | **2** | 0 |
| `ep_049` – `ep_050` | 2 | **1** | 0 |
| `ep_051` – `ep_053` | 3 | **3** | 0 |
| `ep_054` – `ep_058` | 5 | **2** | 0 |
| `ep_059` – `ep_062` | 4 | **1** | 0 |
| `ep_063` – `ep_066` | 4 | **2** | 0 |
| `ep_067` – `ep_070` | 4 | **4** | 0 |
| `ep_071` | 1 | **3** | 0 |
| `ep_072` | 1 | **1** | 0 |
| `ep_073` – `ep_075` | 3 | **2** | 0 |
| `ep_076` – `ep_078` | 3 | **4** | 0 |
| `ep_079` | 1 | **5** | 0 |
| `ep_080` – `ep_082` | 3 | **4** | 0 |
| `ep_083` – `ep_086` | 4 | **5** | 0 |
| `ep_087` – `ep_093` | 7 | **3** | 1 |
| `ep_094` – `ep_095` | 2 | **4** | 1 |
| `ep_096` | 1 | **3** | 1 |
| `ep_097` – `ep_112` | 16 | **2** | 1 |

> Note on `task_index`:
> - `0`: `"Pick object from table and put it inside the box"` (`ep_000`–`ep_086`, 87 episodes)
> - `1`: `"pick rightmost object on flat surface"` (`ep_087`–`ep_112`, 26 episodes)

---

## 2. Stratified Split Balance (`Train 80%` / `Val 10%` / `Test 10%`)

Splitting 113 episodes into **91 Train (80.53%) / 11 Val (9.73%) / 11 Test (9.73%)** achieves identical `bag_count` distributions (`3 : 3 : 3 : 1 : 1`) across `val` and `test` while covering all 5 `bag_count` classes in every split, balancing `task_index` and frame counts, and spacing holdout episodes evenly across time to avoid contiguous-scene clustering:

| Metric / `bag_count` | All (113) | Train (91, 80.5%) | Val (11, 9.7%) | Test (11, 9.7%) |
| :---: | :---: | :---: | :---: | :---: |
| **`bag_count = 1`** | 30 (26.5%) | **24** (26.4%) | **3** (27.3%) | **3** (27.3%) |
| **`bag_count = 2`** | 32 (28.3%) | **26** (28.6%) | **3** (27.3%) | **3** (27.3%) |
| **`bag_count = 3`** | 34 (30.1%) | **28** (30.8%) | **3** (27.3%) | **3** (27.3%) |
| **`bag_count = 4`** | 12 (10.6%) | **10** (11.0%) | **1** (9.1%) | **1** (9.1%) |
| **`bag_count = 5`** | 5 (4.4%) | **3** (3.3%) | **1** (9.1%) | **1** (9.1%) |
| **Mean `bag_count`** | 2.381 | 2.363 | 2.455 | 2.455 |
| **`task_index` (0 / 1)** | 87 / 26 | 70 / 21 | 8 / 3 | 9 / 2 |
| **Total Frames** | 10,675 (100%) | 8,685 (81.4%) | 1,020 (9.6%) | 970 (9.1%) |

### Breakdown of `Val` and `Test` by `bag_count`

- **Val (11 episodes)**:
  - `bag_count = 1` (3): `[15, 25, 35]`
  - `bag_count = 2` (3): `[46, 99, 110]`
  - `bag_count = 3` (3): `[3, 41, 89]`
  - `bag_count = 4` (1): `[69]`
  - `bag_count = 5` (1): `[83]`
- **Test (11 episodes)**:
  - `bag_count = 1` (3): `[20, 30, 61]`
  - `bag_count = 2` (3): `[58, 73, 105]`
  - `bag_count = 3` (3): `[9, 51, 93]`
  - `bag_count = 4` (1): `[80]`
  - `bag_count = 5` (1): `[85]`

---

## 3. Ready-to-Use Split Definitions

### Episode Index Arrays (`JSON`)

```json
{
  "train": [
    0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23,
    24, 26, 27, 28, 29, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45,
    47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 59, 60, 62, 63, 64, 65, 66, 67,
    68, 70, 71, 72, 74, 75, 76, 77, 78, 79, 81, 82, 84, 86, 87, 88, 90, 91,
    92, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 106, 107, 108, 109, 111, 112
  ],
  "val": [3, 15, 25, 35, 41, 46, 69, 83, 89, 99, 110],
  "test": [9, 20, 30, 51, 58, 61, 73, 80, 85, 93, 105]
}
```

### LeRobot v3 [`meta/info.json`](../data_STEA/EVAL_24Sep_25Sep_STEA_pick_success/meta/info.json) `"splits"` Slice Expression

```json
"splits": {
    "train": "0:3+4:9+10:15+16:20+21:25+26:30+31:35+36:41+42:46+47:51+52:58+59:61+62:69+70:73+74:80+81:83+84:85+86:89+90:93+94:99+100:105+106:110+111:113",
    "val": "3:4+15:16+25:26+35:36+41:42+46:47+69:70+83:84+89:90+99:100+110:111",
    "test": "9:10+20:21+30:31+51:52+58:59+61:62+73:74+80:81+85:86+93:94+105:106"
}
```

### Per-Episode `bag_count` Mapping (`0`–`112`)

```json
{
  "ep_000": 3, "ep_001": 3, "ep_002": 3, "ep_003": 3, "ep_004": 3, "ep_005": 3, "ep_006": 3, "ep_007": 3, "ep_008": 3, "ep_009": 3,
  "ep_010": 3, "ep_011": 2, "ep_012": 2, "ep_013": 1, "ep_014": 1, "ep_015": 1, "ep_016": 1, "ep_017": 1, "ep_018": 1, "ep_019": 1,
  "ep_020": 1, "ep_021": 1, "ep_022": 1, "ep_023": 1, "ep_024": 1, "ep_025": 1, "ep_026": 1, "ep_027": 1, "ep_028": 1, "ep_029": 1,
  "ep_030": 1, "ep_031": 1, "ep_032": 1, "ep_033": 1, "ep_034": 1, "ep_035": 1, "ep_036": 3, "ep_037": 3, "ep_038": 3, "ep_039": 3,
  "ep_040": 3, "ep_041": 3, "ep_042": 3, "ep_043": 3, "ep_044": 3, "ep_045": 3, "ep_046": 2, "ep_047": 3, "ep_048": 2, "ep_049": 1,
  "ep_050": 1, "ep_051": 3, "ep_052": 3, "ep_053": 3, "ep_054": 2, "ep_055": 2, "ep_056": 2, "ep_057": 2, "ep_058": 2, "ep_059": 1,
  "ep_060": 1, "ep_061": 1, "ep_062": 1, "ep_063": 2, "ep_064": 2, "ep_065": 2, "ep_066": 2, "ep_067": 4, "ep_068": 4, "ep_069": 4,
  "ep_070": 4, "ep_071": 3, "ep_072": 1, "ep_073": 2, "ep_074": 2, "ep_075": 2, "ep_076": 4, "ep_077": 4, "ep_078": 4, "ep_079": 5,
  "ep_080": 4, "ep_081": 4, "ep_082": 4, "ep_083": 5, "ep_084": 5, "ep_085": 5, "ep_086": 5, "ep_087": 3, "ep_088": 3, "ep_089": 3,
  "ep_090": 3, "ep_091": 3, "ep_092": 3, "ep_093": 3, "ep_094": 4, "ep_095": 4, "ep_096": 3, "ep_097": 2, "ep_098": 2, "ep_099": 2,
  "ep_100": 2, "ep_101": 2, "ep_102": 2, "ep_103": 2, "ep_104": 2, "ep_105": 2, "ep_106": 2, "ep_107": 2, "ep_108": 2, "ep_109": 2,
  "ep_110": 2, "ep_111": 2, "ep_112": 2
}
```
