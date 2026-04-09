"""
LiDAR Trajectory Dataset for L-JEPA pretraining.

Loads per-episode .npz files produced by collect_data.py from the layout:
    data/
      berlin/
        ep_0000.npz
        ep_0001.npz
        ...
      vegas/
        ep_0000.npz
        ...

Each .npz has keys: scans (T, lidar_dim), vels (T, 2), actions (T, 2)

CRITICAL: training samples must never span episode boundaries.
A context window that straddles a reset would contain observations from two
unrelated track positions. We enforce this strictly by only indexing within
each episode file.

Each sample is a tuple:
    context_obs:  (h, obs_dim)     — observations o_{t-h+1 .. t}
    future_obs:   (h, obs_dim)     — observations o_{t+k-h+1 .. t+k}
    actions:      (k, action_dim)  — actions  a_{t .. t+k-1}

where obs_dim = lidar_dim + vel_dim (= 218 with default config).
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class EpisodeData:
    """
    Holds one episode's observations and actions as contiguous arrays.
    obs = concat(normalised_scan, vel) per timestep.
    """
    __slots__ = ('obs', 'actions', 'T')

    def __init__(self, scans, vels, actions):
        assert len(scans) == len(vels) == len(actions)
        self.obs = np.concatenate([scans, vels], axis=-1).astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.T = len(scans)


class LiDARTrajectoryDataset(Dataset):
    """
    Indexes all valid (context, future, action) windows across every episode.

    Args:
        data_dir:           root data directory (contains map subdirectories)
        context_window:     h — steps fed to the encoder
        prediction_horizon: k — steps ahead to predict
        maps:               list of map names to include (None = all found)
    """

    def __init__(
        self,
        data_dir: str,
        context_window: int = 4,
        prediction_horizon: int = 5,
        maps: list = None,
    ):
        self.h = context_window
        self.k = prediction_horizon

        # Discover episode files
        if maps is not None:
            search_dirs = [os.path.join(data_dir, m) for m in maps]
        else:
            search_dirs = sorted(glob.glob(os.path.join(data_dir, '*')))
            search_dirs = [d for d in search_dirs if os.path.isdir(d)]

        if not search_dirs:
            raise FileNotFoundError(
                f"No map directories found in {data_dir}. "
                f"Run collect_data.py first."
            )

        self.episodes: list[EpisodeData] = []
        # (episode_idx, t) where t is the index of the *last* context step
        self.index: list[tuple[int, int]] = []
        map_counts = {}

        for d in search_dirs:
            map_name = os.path.basename(d)
            ep_files = sorted(glob.glob(os.path.join(d, 'ep_*.npz')))
            ep_count = 0
            for path in ep_files:
                data = np.load(path)
                scans = data['scans']     # already normalised by collect_data
                vels = data['vels']
                actions = data['actions']

                ep = EpisodeData(scans, vels, actions)
                ep_idx = len(self.episodes)
                self.episodes.append(ep)

                # Valid t range: need h-1 steps before t and k steps after t
                min_t = self.h - 1
                max_t = ep.T - self.k - 1
                for t in range(min_t, max_t + 1):
                    self.index.append((ep_idx, t))
                ep_count += 1
            map_counts[map_name] = ep_count

        total_steps = sum(ep.T for ep in self.episodes)
        print(f"[Dataset] Loaded {len(self.episodes)} episodes "
              f"({total_steps:,} steps) → {len(self.index):,} training samples")
        for map_name, cnt in map_counts.items():
            print(f"  {map_name}: {cnt} episodes")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_idx, t = self.index[idx]
        ep = self.episodes[ep_idx]
        h, k = self.h, self.k

        context_obs = ep.obs[t - h + 1: t + 1]          # (h, obs_dim)
        future_obs  = ep.obs[t + k - h + 1: t + k + 1]  # (h, obs_dim)
        actions     = ep.actions[t: t + k]               # (k, action_dim)

        return (
            torch.from_numpy(context_obs),
            torch.from_numpy(future_obs),
            torch.from_numpy(actions),
        )


def make_dataloader(
    data_dir: str,
    context_window: int,
    prediction_horizon: int,
    batch_size: int,
    maps: list = None,
    num_workers: int = 4,
    shuffle: bool = True,
) -> tuple:
    dataset = LiDARTrajectoryDataset(
        data_dir=data_dir,
        context_window=context_window,
        prediction_horizon=prediction_horizon,
        maps=maps,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    return loader, dataset
