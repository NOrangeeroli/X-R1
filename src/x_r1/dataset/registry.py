from typing import Dict, Type, Any

# Import dataset classes
from .xdg.dataset import XDGDataset
from .bigmath.dataset import BigMathDataset
from .math500.dataset import Math500Dataset
from .aime.dataset import AIMEDataset
from .coco.dataset import COCODataset
from .cifar.dataset import CifarDataset
from .reward import Reward, SVGReward
# Registry for datasets
DATASETS = {
    "xdg": XDGDataset,
    "bigmath": BigMathDataset,
    "math500": Math500Dataset,
    "aime": AIMEDataset,
    "coco": COCODataset,
    "cifar": CifarDataset,
    # Add more datasets here
}

REWARDS = {
    "xdg": Reward,
    "bigmath": Reward,
    "math500": Reward,
    "aime": Reward,
    "coco": SVGReward,
    "cifar": SVGReward,
}



def get_dataset_class(dataset_name: str):
    """Get dataset class by name"""
    if "xiaodonggua" in dataset_name:
        dataset_name = "xdg"
    elif dataset_name == "SynthLabsAI/Big-Math-RL-Verified":
        dataset_name = "bigmath"
    elif dataset_name == "HuggingFaceH4/aime_2024":
        dataset_name = "aime"
    elif dataset_name == "HuggingFaceH4/math_500":
        dataset_name = "math500"
    elif dataset_name == "phiyodr/coco2017":
        dataset_name = "coco"
    elif dataset_name == "uoft-cs/cifar100":
        dataset_name = "cifar"
        
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {list(DATASETS.keys())}")
    return DATASETS[dataset_name]

def get_reward_class(reward_name: str) -> Type[Any]:
    """Get reward class by name"""
    if "xiaodonggua" in reward_name:
        reward_name = "xdg"
    elif reward_name == "SynthLabsAI/Big-Math-RL-Verified":
        reward_name = "bigmath"
    elif reward_name == "HuggingFaceH4/aime_2024":
        reward_name = "aime"
    elif reward_name == "HuggingFaceH4/math_500":
        reward_name = "math500"
    elif reward_name == "phiyodr/coco2017":
        reward_name = "coco"
    elif reward_name == "uoft-cs/cifar100":
        reward_name = "cifar"
    if reward_name not in REWARDS:
        raise ValueError(f"Reward {reward_name} not found. Available rewards: {list(REWARDS.keys())}")
    return REWARDS[reward_name]