"""Stratified splits, bag-count annotations, camera presets, and state/action subspace definitions.

Grounded in:
- policy/STEA_pick_success_splits.md
- policy/STEA_dataset_distribution_and_act_analysis.md
- contracts/dem_pickplace.yaml
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_DATASET_ROOT = "/data/xiuchao/biArm/DEM/data_STEA/EVAL_24Sep_25Sep_STEA_pick_success"
DEFAULT_REPO_ID = "local/EVAL_24Sep_25Sep_STEA_pick_success"

CAM_TOP = "observation.images.camera_top"
CAM_FRONT = "observation.images.camera_front"
CAM_LEFT = "observation.images.camera_left"    # Physical mount: active RIGHT-arm wrist
CAM_RIGHT = "observation.images.camera_right"  # Physical mount: parked LEFT-arm wrist

ALL_CAMERA_KEYS: tuple[str, ...] = (CAM_TOP, CAM_FRONT, CAM_LEFT, CAM_RIGHT)

CAMERA_PRESETS: dict[str, tuple[str, ...]] = {
    "all_4cams": (CAM_TOP, CAM_FRONT, CAM_LEFT, CAM_RIGHT),
    "3cams_active": (CAM_TOP, CAM_FRONT, CAM_LEFT),
    "2cams_top_left": (CAM_TOP, CAM_LEFT),
    "1cam_top": (CAM_TOP,),
    "1cam_left": (CAM_LEFT,),
}

CAMERA_SHORT_NAMES: dict[str, str] = {
    CAM_TOP: "top (global)",
    CAM_FRONT: "front (angled)",
    CAM_LEFT: "left_key->right_wrist (active)",
    CAM_RIGHT: "right_key->left_wrist (parked)",
}

# Exact stratified 80% / 10% / 10% split (91 train / 11 val / 11 test = 113 episodes)
SPLITS: dict[str, list[int]] = {
    "train": [
        0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23,
        24, 26, 27, 28, 29, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45,
        47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 59, 60, 62, 63, 64, 65, 66, 67,
        68, 70, 71, 72, 74, 75, 76, 77, 78, 79, 81, 82, 84, 86, 87, 88, 90, 91,
        92, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 106, 107, 108, 109, 111, 112,
    ],
    "val": [3, 15, 25, 35, 41, 46, 69, 83, 89, 99, 110],
    "test": [9, 20, 30, 51, 58, 61, 73, 80, 85, 93, 105],
}

# Per-episode scene clutter (bag_count in {1, 2, 3, 4, 5})
EPISODE_BAG_COUNT: dict[int, int] = {
    0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3,
    10: 3, 11: 2, 12: 2, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1,
    20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1,
    30: 1, 31: 1, 32: 1, 33: 1, 34: 1, 35: 1, 36: 3, 37: 3, 38: 3, 39: 3,
    40: 3, 41: 3, 42: 3, 43: 3, 44: 3, 45: 3, 46: 2, 47: 3, 48: 2, 49: 1,
    50: 1, 51: 3, 52: 3, 53: 3, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 1,
    60: 1, 61: 1, 62: 1, 63: 2, 64: 2, 65: 2, 66: 2, 67: 4, 68: 4, 69: 4,
    70: 4, 71: 3, 72: 1, 73: 2, 74: 2, 75: 2, 76: 4, 77: 4, 78: 4, 79: 5,
    80: 4, 81: 4, 82: 4, 83: 5, 84: 5, 85: 5, 86: 5, 87: 3, 88: 3, 89: 3,
    90: 3, 91: 3, 92: 3, 93: 3, 94: 4, 95: 4, 96: 3, 97: 2, 98: 2, 99: 2,
    100: 2, 101: 2, 102: 2, 103: 2, 104: 2, 105: 2, 106: 2, 107: 2, 108: 2, 109: 2,
    110: 2, 111: 2, 112: 2,
}


@dataclass(frozen=True)
class SubspaceSpec:
    """Mapping from raw 16D bimanual schema to model state/action tensors."""

    mode: str  # "8d" or "16d"
    state_indices: tuple[int, ...]
    action_indices: tuple[int, ...]
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    right_pos_slice: slice
    right_quat_slice: slice
    right_gripper_idx: int

    @property
    def state_dim(self) -> int:
        return len(self.state_indices)

    @property
    def action_dim(self) -> int:
        return len(self.action_indices)


RAW_STATE_NAMES: tuple[str, ...] = (
    "ZPF_left_Joint1.pos",
    "ZPF_left_Joint2.pos",
    "ZPF_left_Joint3.pos",
    "ZPF_left_Joint4.pos",
    "ZPF_left_Joint5.pos",
    "ZPF_left_Joint6.pos",
    "ZPF_left_Joint7.pos",
    "ZPF_right_Joint1.pos",
    "ZPF_right_Joint2.pos",
    "ZPF_right_Joint3.pos",
    "ZPF_right_Joint4.pos",
    "ZPF_right_Joint5.pos",
    "ZPF_right_Joint6.pos",
    "ZPF_right_Joint7.pos",
    "2FEG_l_Joint1.pos",
    "2FEG_r_Joint1.pos",
)

RAW_ACTION_NAMES: tuple[str, ...] = (
    "left_arm.x",
    "left_arm.y",
    "left_arm.z",
    "left_arm.qx",
    "left_arm.qy",
    "left_arm.qz",
    "left_arm.qw",
    "right_arm.x",
    "right_arm.y",
    "right_arm.z",
    "right_arm.qx",
    "right_arm.qy",
    "right_arm.qz",
    "right_arm.qw",
    "left_gripper.pos",
    "right_gripper.pos",
)


def get_subspace_spec(mode: str = "8d") -> SubspaceSpec:
    """Return active right-arm 8-DoF spec or full 16-DoF bimanual spec."""
    if mode == "8d":
        idx = (7, 8, 9, 10, 11, 12, 13, 15)
        return SubspaceSpec(
            mode="8d",
            state_indices=idx,
            action_indices=idx,
            state_names=tuple(RAW_STATE_NAMES[i] for i in idx),
            action_names=tuple(RAW_ACTION_NAMES[i] for i in idx),
            right_pos_slice=slice(0, 3),
            right_quat_slice=slice(3, 7),
            right_gripper_idx=7,
        )
    if mode == "16d":
        idx = tuple(range(16))
        return SubspaceSpec(
            mode="16d",
            state_indices=idx,
            action_indices=idx,
            state_names=RAW_STATE_NAMES,
            action_names=RAW_ACTION_NAMES,
            right_pos_slice=slice(7, 10),
            right_quat_slice=slice(10, 14),
            right_gripper_idx=15,
        )
    raise ValueError(f"Unsupported dof mode '{mode}'. Choose from ('8d', '16d').")


def validate_splits() -> None:
    """Assert split disjointness, completeness, and stratified bag_count balance."""
    train_set = set(SPLITS["train"])
    val_set = set(SPLITS["val"])
    test_set = set(SPLITS["test"])

    assert len(train_set) == 91, f"Expected 91 train episodes, got {len(train_set)}"
    assert len(val_set) == 11, f"Expected 11 val episodes, got {len(val_set)}"
    assert len(test_set) == 11, f"Expected 11 test episodes, got {len(test_set)}"
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert (train_set | val_set | test_set) == set(range(113))
    assert set(EPISODE_BAG_COUNT.keys()) == set(range(113))

    expected_val_test = {1: 3, 2: 3, 3: 3, 4: 1, 5: 1}
    for split_name in ("val", "test"):
        counts = {k: 0 for k in range(1, 6)}
        for ep in SPLITS[split_name]:
            counts[EPISODE_BAG_COUNT[ep]] += 1
        assert counts == expected_val_test, f"{split_name} bag_count mismatch: {counts}"
