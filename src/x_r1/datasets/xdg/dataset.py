from typing import Dict, List, Optional, Union, Any
from datasets import load_dataset, Dataset, IterableDataset
import torch

class XDGDataset:
    """
    Dataset loader and processor for XDG Dataset
    """
    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )

    
    @staticmethod
    def load_dataset(
        dataset_name: str,
        dataset_config: Optional[str] = None,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        **kwargs
    ) -> Union[Dataset, IterableDataset]:
        """
        Load the dataset from HuggingFace or local source
        """
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        
        # Apply filtering or selection if needed
        if max_samples and max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            
        return dataset
    
    @staticmethod
    def process_example(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single example from the dataset
        """
       
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }
            
    @staticmethod
    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, List]:
        """
        Collate examples into a batch
        """
        # Standard collation for most cases
        return {key: [example[key] for example in examples] for key in examples[0].keys()}