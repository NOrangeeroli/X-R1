from typing import Dict, Type, Any

# Import dataset classes
from .xdg.dataset import XDGDataset
from .xdg.reward import XDGReward
# Registry for datasets
DATASETS = {
    "xdg": XDGDataset,
    # Add more datasets here
}

REWARDS = {
    "xdg": XDGReward,
    # Add more rewards here
}

def get_dataset_class(dataset_name: str):
    """Get dataset class by name"""
    if "xiaodonggua" in dataset_name:
        dataset_name = "xdg"
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {list(DATASETS.keys())}")
    return DATASETS[dataset_name]

def get_reward_class(reward_name: str) -> Type[Any]:
    """Get reward class by name"""
    if "xiaodonggua" in reward_name:
        reward_name = "xdg"
    if reward_name not in REWARDS:
        raise ValueError(f"Reward {reward_name} not found. Available rewards: {list(REWARDS.keys())}")
    return REWARDS[reward_name]