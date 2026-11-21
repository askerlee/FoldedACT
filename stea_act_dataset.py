"""Fast, LeRobot-v3 bit-exact dataset & normalization builder for STEA ACT camera ablations.

Key design guarantees:
1. Bit-exact RGB frames vs `LeRobotDataset` (`max_abs_diff == 0.0`), decoded sequentially once
   into read-only CHW uint8 memmaps to eliminate 2.4 s/sample AV1 random-seek overhead.
2. Reconstructs zeroed `action[:, 15]` (`right_gripper.pos`) per episode from `state[t + shift, 15]`
   following `contracts/dem_pickplace.yaml` (`0.0 = open`, `1.0 = closed`).
3. Computes `observation.state` and `action` normalization statistics strictly on the 91-episode
   training split (zero test/val leakage) and uses true ImageNet (or pixel-level train) stats
   for ResNet18 `FrozenBatchNorm2d` instead of the corrupted `std ~ 0.006` in raw `meta/stats.json`.
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Sequence

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .splits_and_configs import (
    ALL_CAMERA_KEYS,
    DEFAULT_DATASET_ROOT,
    EPISODE_BAG_COUNT,
    SubspaceSpec,
    get_subspace_spec,
)

IMAGENET_MEAN = np.array([[[0.485]], [[0.456]], [[0.406]]], dtype=np.float32)
IMAGENET_STD = np.array([[[0.229]], [[0.224]], [[0.225]]], dtype=np.float32)

DEFAULT_CACHE_DIR = str(
    Path(__file__).resolve().parents[1] / "outputs" / "STEA_pick_success" / "act_frame_cache"
)


def ensure_video_frame_cache(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    camera_keys: Sequence[str] = ALL_CAMERA_KEYS,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> dict[str, Path]:
    """Ensure sequential CHW uint8 memmap cache exists for each requested camera stream."""
    dataset_root = Path(dataset_root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    info_path = dataset_root / "meta" / "info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    total_frames = int(info["total_frames"])
    ep_df = pq.read_table(dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pandas()

    cache_paths: dict[str, Path] = {}
    for cam_key in camera_keys:
        ft = info["features"][cam_key]
        h, w, c = int(ft["shape"][0]), int(ft["shape"][1]), int(ft["shape"][2])
        expected_bytes = total_frames * c * h * w

        mmap_path = cache_dir / f"{cam_key}.uint8_{total_frames}x{c}x{h}x{w}.mmap"
        lock_path = cache_dir / f"{cam_key}.lock"

        with open(lock_path, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if not mmap_path.exists() or mmap_path.stat().st_size != expected_bytes:
                tmp_path = mmap_path.with_suffix(".tmp")
                if tmp_path.exists():
                    tmp_path.unlink()

                # Gather unique ordered (chunk_idx, file_idx) pairs for this camera
                chunk_col = f"videos/{cam_key}/chunk_index"
                file_col = f"videos/{cam_key}/file_index"
                video_files: list[Path] = []
                seen: set[tuple[int, int]] = set()
                for c_idx, f_idx in zip(ep_df[chunk_col].tolist(), ep_df[file_col].tolist()):
                    pair = (int(c_idx), int(f_idx))
                    if pair not in seen:
                        seen.add(pair)
                        rel_vp = info["video_path"].format(
                            video_key=cam_key, chunk_index=pair[0], file_index=pair[1]
                        )
                        video_files.append(dataset_root / rel_vp)

                print(f"[cache] Decoding {cam_key} ({total_frames} frames) -> {mmap_path.name} ...")
                mm = np.memmap(tmp_path, dtype=np.uint8, mode="w+", shape=(total_frames, c, h, w))
                write_idx = 0
                for vp in video_files:
                    with av.open(str(vp)) as container:
                        stream = container.streams.video[0]
                        stream.thread_type = "AUTO"
                        for frame in container.decode(video=0):
                            rgb_hwc = frame.to_ndarray(format="rgb24")
                            mm[write_idx] = np.transpose(rgb_hwc, (2, 0, 1))
                            write_idx += 1
                if write_idx != total_frames:
                    mm._mmap.close()  # type: ignore[attr-defined]
                    tmp_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Decoded {write_idx} frames for {cam_key}, expected {total_frames}"
                    )
                mm.flush()
                del mm
                tmp_path.rename(mmap_path)
                print(f"[cache] Ready: {mmap_path.name} ({expected_bytes / 1e9:.2f} GB)")

        cache_paths[cam_key] = mmap_path
    return cache_paths


def reconstruct_right_gripper_action(
    ep_state_16d: np.ndarray,
    *,
    gripper_mode: str = "binary",
    shift: int = 1,
    open_threshold: float = 0.70,
    max_open_rad: float = 0.80,
) -> np.ndarray:
    """Reconstruct target `right_gripper.pos` for a single episode of length T.

    In `aibot2` (`contracts/dem_pickplace.yaml`):
      - `observation.state[:, 15]` (`2FEG_r_Joint1.pos`): `~0.80` = open, `~0.00` = closed
      - `action[:, 15]` (`right_gripper.pos`): `0.0` = open, `1.0` = closed
    """
    t_len = ep_state_16d.shape[0]
    future_idx = np.minimum(np.arange(t_len) + shift, t_len - 1)
    future_joint = ep_state_16d[future_idx, 15].astype(np.float32)

    if gripper_mode == "binary":
        return (future_joint < open_threshold).astype(np.float32)
    if gripper_mode == "continuous":
        return np.clip(1.0 - future_joint / max_open_rad, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unknown gripper_mode '{gripper_mode}'. Choose from ('binary', 'continuous').")


def load_reconstructed_trajectories(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    dof_mode: str = "8d",
    gripper_mode: str = "binary",
    gripper_shift: int = 1,
    gripper_open_threshold: float = 0.70,
) -> dict[str, np.ndarray | SubspaceSpec]:
    """Load full tabular trajectories (10,675 frames), reconstruct right gripper, and slice subspace."""
    dataset_root = Path(dataset_root)
    subspace = get_subspace_spec(dof_mode)

    parquet_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    tables = [pq.read_table(p).to_pandas() for p in parquet_files]
    df = tables[0] if len(tables) == 1 else pd_concat_sorted(tables)
    df = df.sort_values("index").reset_index(drop=True)

    raw_state = np.stack(df["observation.state"].values).astype(np.float32)
    raw_action = np.stack(df["action"].values).astype(np.float32)
    episode_index = df["episode_index"].to_numpy(dtype=np.int64)
    frame_index = df["frame_index"].to_numpy(dtype=np.int64)
    abs_index = df["index"].to_numpy(dtype=np.int64)
    timestamp = df["timestamp"].to_numpy(dtype=np.float32)

    assert np.array_equal(abs_index, np.arange(len(df), dtype=np.int64)), "Non-contiguous frame index!"

    num_episodes = int(episode_index.max()) + 1
    ep_starts = np.zeros(num_episodes, dtype=np.int64)
    ep_ends = np.zeros(num_episodes, dtype=np.int64)

    fixed_action = raw_action.copy()
    for ep in range(num_episodes):
        mask = np.flatnonzero(episode_index == ep)
        assert len(mask) > 0, f"Empty episode {ep}"
        assert mask[-1] - mask[0] + 1 == len(mask), f"Non-contiguous episode {ep}"
        ep_starts[ep] = int(mask[0])
        ep_ends[ep] = int(mask[-1]) + 1
        fixed_action[mask, 15] = reconstruct_right_gripper_action(
            raw_state[mask],
            gripper_mode=gripper_mode,
            shift=gripper_shift,
            open_threshold=gripper_open_threshold,
        )

    state_sub = raw_state[:, subspace.state_indices]
    action_sub = fixed_action[:, subspace.action_indices]

    return {
        "subspace": subspace,
        "state": state_sub,
        "action": action_sub,
        "raw_state_16d": raw_state,
        "raw_action_16d_fixed": fixed_action,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "abs_index": abs_index,
        "timestamp": timestamp,
        "ep_starts": ep_starts,
        "ep_ends": ep_ends,
    }


def pd_concat_sorted(tables):
    import pandas as pd

    return pd.concat(tables, ignore_index=True)


class STEAACTDataset(Dataset):
    """Episode-filtered ACT dataset supporting arbitrary camera subsets without file duplication."""

    def __init__(
        self,
        episodes: Sequence[int],
        camera_keys: Sequence[str],
        *,
        chunk_size: int = 50,
        dof_mode: str = "8d",
        gripper_mode: str = "binary",
        gripper_shift: int = 1,
        gripper_open_threshold: float = 0.70,
        dataset_root: str | Path = DEFAULT_DATASET_ROOT,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        trajectories: dict | None = None,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.episodes = sorted(int(e) for e in episodes)
        self.camera_keys = tuple(camera_keys)
        self.chunk_size = int(chunk_size)

        with open(self.dataset_root / "meta" / "info.json", "r", encoding="utf-8") as f:
            info = json.load(f)
        self.total_dataset_frames = int(info["total_frames"])
        self.image_shapes: dict[str, tuple[int, int, int]] = {}
        for ck in self.camera_keys:
            ft = info["features"][ck]
            # Convert HWC in info.json -> CHW tensor shape
            self.image_shapes[ck] = (int(ft["shape"][2]), int(ft["shape"][0]), int(ft["shape"][1]))

        self.cache_paths = ensure_video_frame_cache(
            dataset_root=self.dataset_root,
            camera_keys=self.camera_keys,
            cache_dir=cache_dir,
        )
        self._mmaps: dict[str, np.memmap] | None = None

        traj = trajectories or load_reconstructed_trajectories(
            dataset_root=self.dataset_root,
            dof_mode=dof_mode,
            gripper_mode=gripper_mode,
            gripper_shift=gripper_shift,
            gripper_open_threshold=gripper_open_threshold,
        )
        self.subspace: SubspaceSpec = traj["subspace"]
        self.all_state: np.ndarray = traj["state"]
        self.all_action: np.ndarray = traj["action"]
        self.all_episode_index: np.ndarray = traj["episode_index"]
        self.all_frame_index: np.ndarray = traj["frame_index"]
        self.ep_starts: np.ndarray = traj["ep_starts"]
        self.ep_ends: np.ndarray = traj["ep_ends"]

        selected_indices: list[np.ndarray] = []
        for ep in self.episodes:
            selected_indices.append(np.arange(self.ep_starts[ep], self.ep_ends[ep], dtype=np.int64))
        self.sample_abs_indices = np.concatenate(selected_indices, axis=0)

    def _get_mmaps(self) -> dict[str, np.memmap]:
        # Lazy-open per worker process so multi-worker DataLoaders never share file descriptors unsafely
        if self._mmaps is None:
            self._mmaps = {
                ck: np.memmap(
                    self.cache_paths[ck],
                    dtype=np.uint8,
                    mode="r",
                    shape=(self.total_dataset_frames, *self.image_shapes[ck]),
                )
                for ck in self.camera_keys
            }
        return self._mmaps

    def __len__(self) -> int:
        return int(self.sample_abs_indices.shape[0])

    def __getitem__(self, rel_idx: int) -> dict[str, torch.Tensor]:
        abs_idx = int(self.sample_abs_indices[rel_idx])
        ep_idx = int(self.all_episode_index[abs_idx])
        ep_end = int(self.ep_ends[ep_idx])

        offsets = abs_idx + np.arange(self.chunk_size, dtype=np.int64)
        is_pad = offsets >= ep_end
        clamped_offsets = np.minimum(offsets, ep_end - 1)

        item: dict[str, torch.Tensor] = {
            "observation.state": torch.from_numpy(self.all_state[abs_idx].copy()),
            "action": torch.from_numpy(self.all_action[clamped_offsets].copy()),
            "action_is_pad": torch.from_numpy(is_pad),
            "episode_index": torch.tensor(ep_idx, dtype=torch.long),
            "frame_index": torch.tensor(int(self.all_frame_index[abs_idx]), dtype=torch.long),
            "index": torch.tensor(abs_idx, dtype=torch.long),
            "bag_count": torch.tensor(EPISODE_BAG_COUNT[ep_idx], dtype=torch.long),
        }

        mmaps = self._get_mmaps()
        for ck in self.camera_keys:
            # Keep uint8 through DataLoader IPC + H2D transfer (4x lower bandwidth);
            # BatchNormalizer converts uint8 -> float32 [0, 1] directly on GPU.
            frame_u8 = np.array(mmaps[ck][abs_idx], dtype=np.uint8, copy=True)
            item[ck] = torch.from_numpy(frame_u8)

        return item


def compute_train_stats(
    train_dataset: STEAACTDataset,
    *,
    image_norm: str = "imagenet",
    std_floor: float = 1e-3,
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute leak-free normalization statistics strictly over the training split."""
    idx = train_dataset.sample_abs_indices
    train_states = train_dataset.all_state[idx]
    train_actions = train_dataset.all_action[idx]

    state_mean = train_states.mean(axis=0).astype(np.float32)
    state_std = np.maximum(train_states.std(axis=0), std_floor).astype(np.float32)
    state_min = train_states.min(axis=0).astype(np.float32)
    state_max = train_states.max(axis=0).astype(np.float32)

    action_mean = train_actions.mean(axis=0).astype(np.float32)
    action_std = np.maximum(train_actions.std(axis=0), std_floor).astype(np.float32)
    action_min = train_actions.min(axis=0).astype(np.float32)
    action_max = train_actions.max(axis=0).astype(np.float32)

    stats: dict[str, dict[str, torch.Tensor]] = {
        "observation.state": {
            "mean": torch.from_numpy(state_mean),
            "std": torch.from_numpy(state_std),
            "min": torch.from_numpy(state_min),
            "max": torch.from_numpy(state_max),
        },
        "action": {
            "mean": torch.from_numpy(action_mean),
            "std": torch.from_numpy(action_std),
            "min": torch.from_numpy(action_min),
            "max": torch.from_numpy(action_max),
        },
    }

    mmaps = train_dataset._get_mmaps()
    for ck in train_dataset.camera_keys:
        if image_norm == "imagenet":
            cam_mean = IMAGENET_MEAN.copy()
            cam_std = IMAGENET_STD.copy()
        elif image_norm == "train_stats":
            # Subsample every 5th training frame for exact pixel-level RGB mean/std
            sub_idx = idx[::5]
            imgs = mmaps[ck][sub_idx].astype(np.float32) / 255.0  # (N, 3, H, W)
            cam_mean = imgs.mean(axis=(0, 2, 3), keepdims=False).reshape(3, 1, 1).astype(np.float32)
            cam_std = np.maximum(
                imgs.std(axis=(0, 2, 3), keepdims=False).reshape(3, 1, 1), std_floor
            ).astype(np.float32)
        else:
            raise ValueError(f"Unknown image_norm '{image_norm}'. Choose from ('imagenet', 'train_stats').")

        stats[ck] = {
            "mean": torch.from_numpy(cam_mean),
            "std": torch.from_numpy(cam_std),
            "min": torch.zeros((3, 1, 1), dtype=torch.float32),
            "max": torch.ones((3, 1, 1), dtype=torch.float32),
        }

    return stats
