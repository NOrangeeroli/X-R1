from typing import Dict, List, Optional, Union, Any
import os
import json
from datasets import Dataset, IterableDataset, DatasetDict
import torch
from pathlib import Path
import pandas as pd
from PIL import Image

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think>\n<answer> answer here </answer>"
)

class COCOImageDataset:
    """
    Dataset loader and processor for COCO Image Dataset from local files
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
        Load the dataset from local files at ~/data/coco
        """
        coco_dir = os.path.expanduser("~/data/coco")
        
        # Load annotations
        train_captions_file = os.path.join(coco_dir, "annotations/captions_train2017.json")
        val_captions_file = os.path.join(coco_dir, "annotations/captions_val2017.json")
        
        # Process annotations to create datasets
        train_dataset = COCOImageDataset._load_from_annotations(train_captions_file, coco_dir, "train")
        val_dataset = COCOImageDataset._load_from_annotations(val_captions_file, coco_dir, "val")
        
        # Limit samples if specified
        if max_train_samples > 0:
            train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
            
        if max_test_samples > 0:
            val_dataset = val_dataset.select(range(min(max_test_samples, len(val_dataset))))
        
        # Apply basic processing WITHOUT loading images
        train_dataset = train_dataset.map(COCOImageDataset.process_example_metadata)
        val_dataset = val_dataset.map(COCOImageDataset.process_example_metadata)
        
        # Set load_image function as format
        train_dataset.set_transform(COCOImageDataset.load_image_transform)
        val_dataset.set_transform(COCOImageDataset.load_image_transform)
        
        # Combine into DatasetDict
        return DatasetDict({
            "train": train_dataset,
            "test": val_dataset
        })
    
    @staticmethod
    def _load_from_annotations(annotation_file: str, coco_dir: str, split: str) -> Dataset:
        """
        Load dataset from COCO annotation file
        """
        # Load the JSON
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Extract image info and captions
        images = {img['id']: img for img in data['images']}
        
        records = []
        for ann in data['annotations']:
            img_id = ann['image_id']
            img_info = images[img_id]
            
            # Create a record similar to HuggingFace's COCO dataset structure
            records.append({
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'image_path': os.path.join(coco_dir, f"{split}2017", img_info['file_name']),
                'sentences': {
                    'raw': ann['caption']
                }
            })
        
        # Convert to DataFrame then to Dataset
        df = pd.DataFrame(records)
        return Dataset.from_pandas(df)
    
    @staticmethod
    def process_example_metadata(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata only (no image loading)
        """
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Please write SVG code for generating the black and white image corresponding to the following description: {example['sentences']['raw']}"},
            ],
            "solution": example["sentences"]["raw"],
            "image_path": example["image_path"],  # Keep path for later loading
            "image_id": example["image_id"]
        }
    
    @staticmethod
    def load_image_transform(examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load image on-demand during dataset access
        """
        images = []
    
        # Process each image path
        for path in examples['image_path']:
            try:
                image = Image.open(path).convert('RGB')
                images.append(image)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                images.append(None)
        
        # Add the images to the examples dictionary
        examples['image'] = images
        return examples