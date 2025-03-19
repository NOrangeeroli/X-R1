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
    
    
    @staticmethod
    def load_dataset(
        dataset_name: str,
        dataset_config: Optional[Dict[str, Any]] = None,
        max_train_samples: Optional[int] = -1,
        max_test_samples: Optional[int] = -1,
        **kwargs
    ) -> Union[Dataset, IterableDataset]:
        """
        Create a synthetic dataset with entries of 'dog'
        """
        import os
        import csv
        
        # Get rank info for distributed training
        rank = int(os.environ.get("RANK", "0"))
        
        # Set cache directory
        default_cache_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = kwargs.get("cache_dir", default_cache_dir)
        
        # Define file paths
        train_csv_path = f"{cache_dir}/simple_object_train.csv"
        test_csv_path = f"{cache_dir}/simple_object_test.csv"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Only rank 0 creates files if they don't exist
        if rank == 0:
            print(f"Rank {rank}: Creating CSV dataset files")
            
            # Create training data CSV
            with open(train_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['input', 'output'])  # Header
                for _ in range(500):
                    writer.writerow(['a dog.', '-'])
            
            # Create test data CSV
            with open(test_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['input', 'output'])  # Header
                for _ in range(100):
                    writer.writerow(['a dog.', ''])
            
            print(f"Rank {rank}: CSV files created at {cache_dir}")
        
        # If in distributed setting, wait for rank 0 to finish
        if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
        
        # Now all processes load the dataset using load_dataset
        print(f"Rank {rank}: Loading dataset from CSV files")
        dataset = load_dataset('csv', 
                            data_files={'train': train_csv_path, 
                                        'test': test_csv_path},
                            )
        
        # Process dataset - exactly like the working InstructSVGDataset
        for split in dataset:
            if "solution" in dataset[split].column_names:
                dataset[split] = dataset[split].remove_columns("solution")
        
        # Use the simpler map without parameters - just like InstructSVGDataset
        dataset = dataset.map(SimpleObjectDataset.process_example)
        
        # Apply sample limits if needed
        if max_train_samples and max_train_samples > 0:
            if "train" in dataset:
                dataset["train"] = dataset["train"].select(range(min(max_train_samples, len(dataset["train"]))))
        
        if max_test_samples and max_test_samples > 0:       
            if "test" in dataset:
                dataset["test"] = dataset["test"].select(range(min(max_test_samples, len(dataset["test"]))))
                
        return dataset
    
    
    @staticmethod
    def process_example(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single example from the dataset
        """
        
        
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Please write SVG code for generating the image corresponding to the following description: {example['input']}"},
            ],
            "solution": example["input"],
            "svg": example["output"]
        }
            
   