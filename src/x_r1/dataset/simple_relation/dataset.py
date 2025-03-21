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

class SimpleRelationDataset:
    """
    Dataset loader and processor for XDG Dataset
    """
    

    @staticmethod
    def load_dataset(
        dataset_name: str,
        dataset_config: Optional[Dict[str, Any]] = None,
        max_train_samples: Optional[int] = -1,
        max_test_samples: Optional[int] = -1,
        **kwargs
    ) -> Union[Dataset, IterableDataset]:
        """
        Create a synthetic dataset with 1000 entries of 'dog'
        """
        
        
        # Create a dictionary with 1000 entries of 'dog'
        num_samples = 1000
        items = ['a triangle', 'a square', 'a circle', 'a rectangle', 'a pentagon', 'a hexagon', 'a heptagon', 'an octagon', 'a nonagon', 'a decagon', 'an arrow', 'a vertical line', 'a horizontal line', 'a diagonal line', 'a zigzag line', 'a wavy line', 'a spiral', 'a star', 'a trapezoid', 'a parallelogram']
        relations = ['above', 'below', 'on the left of', 'on the right of', 'inside',  'intersecting', 'overlapping', 'adjacent to',]
        inputs = [x + ' ' + y + ' ' + z for x in items for y in relations for z in items]
        data_dict = {
            "input": inputs,
            "output": [""""""] * len(inputs)  
        }
        
        # Create a pandas DataFrame and then convert to HuggingFace Dataset
        df = pd.DataFrame(data_dict)
        train_dataset = Dataset.from_pandas(df)
        
        # Create a smaller validation set - let's say 100 samples
        
    
        
        # Combine into a DatasetDict with train and validation splits
        dataset = DatasetDict({
            "train": train_dataset,
        })
        
        # The rest of the processing remains the same
        for split in dataset:
            if "solution" in dataset[split].column_names:
                dataset[split] = dataset[split].remove_columns("solution")
            
            dataset[split] = dataset[split].rename_column("output", "svg")
            dataset[split] = dataset[split].rename_column("input", "solution")
        
        dataset = dataset.map(SimpleRelationDataset.process_example)
        
        # Apply sample limits if needed
        if max_train_samples and max_train_samples > 0:
            for split in dataset:
                if split == "train":
                    dataset[split] = dataset[split].select(range(min(max_train_samples, len(dataset[split]))))
        
        if max_test_samples and max_test_samples > 0:       
            for split in dataset:
                if split == "test":
                    dataset[split] = dataset[split].select(range(min(max_test_samples, len(dataset[split]))))    
        
        return dataset
    @staticmethod
    def process_example(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single example from the dataset
        """
        
        
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Please write SVG code for generating the image corresponding to the following description: {example['solution']}"},
            ],
            # "solution": example["input"],
            # "svg": example["output"]
        }
            
   