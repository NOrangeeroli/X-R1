from typing import Dict, List, Optional, Union, Any
from datasets import load_dataset, Dataset, IterableDataset
import torch
import pandas as pd
from datasets import DatasetDict
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think>\n<answer> answer here </answer>"
)

class SimpleObjectDataset:
    """
    Dataset loader and processor for XDG Dataset
    """
    '''@staticmethod
    def load_dataset(
        dataset_name: str,
        dataset_config: Optional[Dict[str, Any]] = None,
        max_train_samples: Optional[int] = -1,
        max_test_samples: Optional[int] = -1,
        **kwargs
    ) -> Union[Dataset, IterableDataset]:
        """
        Create minimal dataset directly in memory
        """
        # Just create a simple list in memory
        simple_list = [{"prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": "Please write SVG code for generating the image corresponding to the following description: a dog."}],
                    
                    "solution": "a dog",
                    "svg": ""}] * 1000  # Repeat 1000 times
        
        # Create dataset dictionary with train/test splits
        train_size = 900
        if max_train_samples > 0:
            train_size = min(train_size, max_train_samples)
        
        test_size = 100
        if max_test_samples > 0:
            test_size = min(test_size, max_test_samples)
        
        return {
            "train": simple_list[:train_size],
            "validation": simple_list[:test_size]
        }
    '''
    @staticmethod
    def load_dataset(
        dataset_name: str,
        dataset_config: Optional[Dict[str, Any]] = None,
        max_train_samples: Optional[int] = -1,
        max_test_samples: Optional[int] = -1,
        **kwargs
    ) -> Union[Dataset, IterableDataset]:
        """
        Create a synthetic dataset with just one unique example
        """
        import os
        from datasets import Dataset, DatasetDict
        objects = ["dog", "cat", "bird", "fish", "tree", "flower", "house", "car", "boat", "plane", "sun", "moon", "table", "chair", "book", "pen", "computer", "phone", "clock", "hat", "shirt", "pants", "skirt", "sock", "glove", "scarf", "jacket", "coat", "dress", "suit", "tie", "belt", "purse", "bag", "glasses", "watch", "ring",  "umbrella"]
        colors = ["red", "blue", "green", "yellow", "black", "brown", "orange", "purple", "pink", "gray"]
        # Get distributed training info
        rank = int(os.environ.get("RANK", "0"))
        print(f"Rank {rank}: Creating single-example dataset")
        
       
        
        # Create data with position markers to avoid deduplication issues
        train_data = []
        for object in objects:
            for color in colors:
                train_data.append({
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Please write SVG code for generating the image corresponding to the following description: a {color} {object}."},
                    ],
                    "solution": f"a {color} {object}",  
                    "svg": f"a {color} {object}",  
                })
        
        
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        
        # Create dataset dictionary
        dataset = DatasetDict({
            "train": train_dataset,
            
        })
        
       
        
        # Map to the final format, dropping the position field
        
        # Apply sample limits
        if max_train_samples and max_train_samples > 0:
            if "train" in dataset:
                dataset["train"] = dataset["train"].select(range(min(max_train_samples, len(dataset["train"]))))
        
        if max_test_samples and max_test_samples > 0:
            if "validation" in dataset:
                dataset["validation"] = dataset["validation"].select(range(min(max_test_samples, len(dataset["validation"]))))
                
        return dataset
    @staticmethod
    def process_example(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single example from the dataset
        """
        
        
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Please write SVG code for generating the image corresponding to the following description: a dog."},
            ],
            # "solution": example["input"],
            # "svg": example["output"]
        }
            
   