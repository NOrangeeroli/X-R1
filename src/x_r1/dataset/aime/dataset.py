from typing import Dict, List, Optional, Union, Any
from datasets import load_dataset, Dataset, IterableDataset
import torch
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

class AIMEDataset:
    """
    Dataset loader and processor for XDG Dataset
    """
    

    
    @staticmethod
    def load_dataset(
        dataset_name: str,
        max_test_samples: Optional[int] = -1,
        **kwargs
    ) -> Union[Dataset, IterableDataset]:
        """
        Load the dataset from HuggingFace or local source
        """
        dataset = load_dataset(dataset_name)
        for split in dataset:
            if "solution"==dataset[split].column_names:
                dataset[split] = dataset[split].remove_columns("solution")
        
        dataset = dataset.map(AIMEDataset.process_example)
        
        
        # Apply filtering or selection if needed
        dataset = dataset['train']
        
        if max_test_samples and max_test_samples > 0:       
            
            dataset = dataset.select(range(min(max_test_samples, len(dataset))))    
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
            "solution": example["answer"],
        }
            
   