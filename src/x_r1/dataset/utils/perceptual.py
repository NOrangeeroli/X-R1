import torch
import torch.nn as nn
import torchvision.models as models
from typing import Union, List
from PIL import Image


def vgg_image_image_distances_batch(
    reference_images: Union[Image.Image, List[Image.Image]], 
    query_images: Union[Image.Image, List[Image.Image]], 
    layer_index=8,
    device=None
) -> Union[float, List[float]]:
    """
    Computes the perceptual distance between reference images and query images using VGG19 features.
    
    Args:
        reference_images: Either a single PIL Image or a list of PIL Images.
        query_images: Either a single PIL Image or a list of PIL Images.
        layer_index: Index of the VGG19 layer to use for feature extraction (default: 8).
        device: Device to run the model on.
    
    Returns:
        If both inputs are single items: a float representing the distance
        If either input is a list: a list of distances
    """
    # Handle single inputs
    single_reference = isinstance(reference_images, Image.Image)
    single_query = isinstance(query_images, Image.Image)
    
    if single_reference:
        reference_images = [reference_images]
    if single_query:
        query_images = [query_images]
    
    # Determine device if not provided
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device is None:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    print(f"vgg_image_image_distances_batch: device: {device}")
    
    # Load the pre-trained VGG19 model and transfer it to the device
    vgg19 = models.vgg19(pretrained=True).features.to(device)
    feature_extractor = nn.Sequential(*list(vgg19.children())[:layer_index]).eval().to(device)
    
    # Freeze the feature extractor's parameters
    for param in feature_extractor.parameters():
        param.requires_grad = False
    
    # Define preprocessing for PIL Images
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Filter out None images and track valid indices
    valid_ref_indices = []
    valid_ref_images = []
    for i, img in enumerate(reference_images):
        if img is not None:
            valid_ref_indices.append(i)
            valid_ref_images.append(prepare_image(img))
    
    valid_query_indices = []
    valid_query_images = []
    for i, img in enumerate(query_images):
        if img is not None:
            valid_query_indices.append(i)
            valid_query_images.append(prepare_image(img))
    
    # Initialize distances with default value (1.0 means maximum distance)
    distances = [1.0] * len(reference_images)
    
    # Only process if we have valid images in both sets
    if valid_ref_images and valid_query_images:
        with torch.no_grad():
            try:
                # Process all reference images at once
                ref_tensors = torch.stack([preprocess(img) for img in valid_ref_images]).to(device)
                ref_features = feature_extractor(ref_tensors)
                
                # Process all query images at once
                query_tensors = torch.stack([preprocess(img) for img in valid_query_images]).to(device)
                query_features = feature_extractor(query_tensors)
                
                # Get features for corresponding pairs and compute MSE
                for i, query_idx in enumerate(valid_query_indices):
                    ref_position = valid_ref_indices.index(query_idx) if query_idx in valid_ref_indices else -1
                    if ref_position >= 0:
                        # Extract features for this specific pair
                        query_feat = query_features[i].unsqueeze(0)
                        ref_feat = ref_features[ref_position].unsqueeze(0)
                        
                        # Calculate MSE between feature representations
                        mse = nn.functional.mse_loss(query_feat, ref_feat).item()
                        
                        # Normalize MSE to 0-1 range (empirical scaling)
                        # The scaling factor (10.0) might need adjustment based on your specific use case
                        distances[query_idx] = min(1.0, mse / 10.0)
            except RuntimeError as e:
                print(f"Error processing images in one batch: {e}")
                print("Consider processing in smaller batches for large datasets")
                # Fall back to default distances (1.0)
    
    # Return single value if both inputs were single items
    if single_reference and single_query and len(distances) == 1:
        return distances[0]
    
    return distances
import torch

