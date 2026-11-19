"""ACT multi-camera partial-observability training and evaluation package for STEA."""

from policy.splits_and_configs import (
    ALL_CAMERA_KEYS,
    CAMERA_PRESETS,
    EPISODE_BAG_COUNT,
    SPLITS,
    SubspaceSpec,
    get_subspace_spec,
    validate_splits,
)

__all__ = [
    "ALL_CAMERA_KEYS",
    "CAMERA_PRESETS",
    "EPISODE_BAG_COUNT",
    "SPLITS",
    "SubspaceSpec",
    "get_subspace_spec",
    "validate_splits",
]
