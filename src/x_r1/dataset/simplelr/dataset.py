from typing import Dict, List, Optional, Union, Any
from datasets import load_dataset, Dataset, IterableDataset
import torch
import os
from datasets import DatasetDict


SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )

class SimplelrDataset:
    """
    Dataset loader and processor for XDG Dataset
    """
   

    
    @staticmethod
    def load_dataset(
        dataset_name: str,
        dataset_config: Optional[str] = None,
        max_train_samples: Optional[int] = -1,
        max_test_samples: Optional[int] = -1,
        **kwargs
    ) -> Union[Dataset, IterableDataset]:
        """
        Load the dataset from HuggingFace or local source
        """
        dataset_path = "/home/chenyamei/codes/cache/hkust-nlp--SimpleRL-Zoo-Data/simplelr_qwen_level3to5"
        train_path = os.path.join(dataset_path, "train.parquet")
        test_path = os.path.join(dataset_path, "test.parquet")
        train_dataset = load_dataset("parquet", data_files=train_path, split="train")
        test_dataset = load_dataset("parquet", data_files=test_path, split="train")
        train_dataset = train_dataset.map(SimplelrDataset.process_example_train)
        test_dataset = test_dataset.map(SimplelrDataset.process_example_test)
        
        # dataset = load_dataset(dataset_name)
        dataset = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
            
        })
        print("Train dataset columns:", train_dataset.column_names)
        print("Train dataset example:", train_dataset[0])
        print("Test dataset columns:", test_dataset.column_names)
        print("Test dataset example:", test_dataset[0])
    
        # dataset = {
        #     "train": train_dataset,
        #     "test": test_dataset
        # }
        
        # dataset = dataset.map(SimplelrDataset.process_example)
        
        
        # Apply filtering or selection if needed
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
    def process_example_train(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single example from the dataset
        """
       
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "solution": example["gt_answer"],
        }
    @staticmethod
    def process_example_test(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single example from the dataset
        """
       
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["extra_info"]["question"]},
            ],
            "solution": example["extra_info"]["answer"],
        }
            
   