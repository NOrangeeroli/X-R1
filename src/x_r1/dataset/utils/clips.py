import cairosvg
from io import BytesIO
from PIL import Image
import torch
import clip
from torchvision import transforms
from lxml import etree
from functools import lru_cache
import os
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Union, List

# Cache for SigLIP models
_siglip_models = {}

@lru_cache(maxsize=30)
def get_siglip_model(model_name="google/siglip-base-patch16-224", device=None):
    """Get SigLIP model in a distributed-friendly way
    
    Args:
        model_name (str): The SigLIP model to load from Hugging Face:
            - "google/siglip-base-patch16-224" (base)
            - "google/siglip-large-patch16-224" (large)
        device: The device to load the model on
        
    Returns:
        tuple: (model, processor) for feature extraction
    """
    from transformers import AutoProcessor, AutoModel
    
    # Get local process info
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    if device is None:
        # Default to CPU for prediction to avoid CUDA synchronization issues
        device = "cpu"
    
    # Create a unique key for this process, model and device
    model_key = f"{local_rank}_{model_name}_{device}"
    
    if model_key not in _siglip_models:
        try:
            # Load model and processor from Hugging Face
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device).eval()
            
            # Freeze parameters to ensure we're only doing inference
            for param in model.parameters():
                param.requires_grad = False
                
            _siglip_models[model_key] = (model, processor)
            
        except Exception as e:
            raise ValueError(f"Error loading SigLIP model {model_name}: {e}")
    
    return _siglip_models[model_key]

# Cache for MAE models
_mae_models = {}

@lru_cache(maxsize=30)
def get_mae_model(model_name="mae_vit_base_patch16", device=None):
    """Get MAE (Masked Autoencoder) model in a distributed-friendly way
    
    Args:
        model_name (str): The MAE model variant:
            - "mae_vit_base_patch16" (base)
            - "mae_vit_large_patch16" (large)
            - "mae_vit_huge_patch14" (huge)
        device: The device to load the model on
        
    Returns:
        tuple: (model, preprocess) for feature extraction
    """
    import torchvision.transforms as transforms
    
    # Get local process info
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    if device is None:
        # Default to CPU for prediction to avoid CUDA synchronization issues
        device = "cpu"
    
    # Create a unique key for this process, model and device
    model_key = f"{local_rank}_{model_name}_{device}"
    
    if model_key not in _mae_models:
        try:
            # Load model from torch hub
            model = torch.hub.load('facebookresearch/mae', model_name, pretrained=True)
            model = model.to(device).eval()
            
            # Freeze parameters to ensure we're only doing inference
            for param in model.parameters():
                param.requires_grad = False
            
            # Define preprocessing for MAE (similar to ViT models)
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            _mae_models[model_key] = (model, preprocess)
            
        except Exception as e:
            raise ValueError(f"Error loading MAE model {model_name}: {e}")
    
    return _mae_models[model_key]

# Cache for DinoV2 models
_dinov2_models = {}

@lru_cache(maxsize=30)
def get_dinov2_model(model_name="dinov2_vits14", device=None):
    """Get DinoV2 model in a distributed-friendly way
    
    Args:
        model_name (str): The DinoV2 model to load:
            - "dinov2_vits14" (small - 21M parameters)
            - "dinov2_vitb14" (base - 86M parameters)
            - "dinov2_vitl14" (large - 304M parameters)
            - "dinov2_vitg14" (giant - 1.1B parameters)
        device: The device to load the model on
        
    Returns:
        nn.Module: DinoV2 model for feature extraction
    """
    # Get local process info
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    if device is None:
        # Default to CPU for prediction to avoid CUDA synchronization issues
        device = "cpu"
    
    # Create a unique key for this process, model and device
    model_key = f"{local_rank}_{model_name}_{device}"
    
    if model_key not in _dinov2_models:
        try:
            # Load model from torch hub
            model = torch.hub.load('facebookresearch/dinov2', model_name)
            model = model.to(device).eval()
            
            # Freeze parameters to ensure we're only doing inference
            for param in model.parameters():
                param.requires_grad = False
                
            _dinov2_models[model_key] = model
            
        except Exception as e:
            raise ValueError(f"Error loading DinoV2 model {model_name}: {e}")
    
    return _dinov2_models[model_key]

# Load CLIP model
_clip_models = {}
@lru_cache(maxsize=30)
def get_clip_model(model_name="ViT-B/32", device=None):
    """Get CLIP model in a distributed-friendly way"""
    # Get local process info
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    if device is None:
        # Use CPU for prediction to avoid CUDA synchronization issues
        device = "cpu"
    
    # Create a unique key for this process and model
    model_key = f"{local_rank}_{model_name}_{device}"
    
    if model_key not in _clip_models:
        # Load model for this specific process
        model, preprocess = clip.load(model_name, device=device)
        _clip_models[model_key] = (model, preprocess)
    
    return _clip_models[model_key]



# Cache for VGG models
_vgg_models = {}

@lru_cache(maxsize=30)
def get_vgg_model(model_name="vgg19", layer_index=8, device=None):
    """Get VGG model in a distributed-friendly way
    
    Args:
        model_name (str): The VGG model to load ("vgg19" or "vgg16")
        layer_index (int): Index of the layer to use for feature extraction
        device: The device to load the model on
        
    Returns:
        nn.Sequential: Feature extractor model that outputs features at the specified layer
    """
    # Get local process info
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    if device is None:
        # Default to CPU for prediction to avoid CUDA synchronization issues
        device = "cpu"
    
    # Create a unique key for this process, model, layer and device
    model_key = f"{local_rank}_{model_name}_{layer_index}_{device}"
    
    if model_key not in _vgg_models:
        # Load model for this specific process
        if model_name == "vgg19":
            vgg_model = models.vgg19(pretrained=True).features.to(device)
        elif model_name == "vgg16":
            vgg_model = models.vgg16(pretrained=True).features.to(device)
        else:
            raise ValueError(f"Unsupported model: {model_name}. Use 'vgg19' or 'vgg16'")
        
        # Create feature extractor up to the specified layer
        feature_extractor = nn.Sequential(*list(vgg_model.children())[:layer_index]).eval().to(device)
        
        # Freeze parameters
        for param in feature_extractor.parameters():
            param.requires_grad = False
            
        _vgg_models[model_key] = feature_extractor
    
    return _vgg_models[model_key]

# device_id = 0  # Use the first GPU
# # device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# # model, preprocess = clip.load("ViT-B/32", device=device)
# model, preprocess = clip.load("ViT-L/14", device=device)

def svg_to_image(svg_code):
    """
    Attempts to parse and recover from errors in SVG code,
    then renders the recovered SVG to a PNG image.

    Parameters:
      svg_code (str): The SVG content as a string.
      output_filename (str): The filename for the output image.
    """
    # Create an XML parser in recovery mode. This tells lxml
    # to try to recover as much as possible from broken XML.
    try:
        parser = etree.XMLParser(recover=True)
        
        
        tree = etree.fromstring(svg_code.encode('utf-8'), parser)
        
        valid_svg = etree.tostring(tree)
        
        
        png_data = cairosvg.svg2png(bytestring=valid_svg)
        image = Image.open(BytesIO(png_data))
        return image
    except Exception as e:
        # print(f"Error converting SVG to image: {e}")
        # black_image = Image.new('RGB', (256, 256), color='black')
        return None


def clip_text_image_distance(text: str, image: Image) -> float:
    """
    Computes the cosine distance between a text and an image using CLIP embeddings.
    
    Args:
        text (str): Input text.
        image (Image): PIL Image.
    
    Returns:
        float: Cosine distance between the text and image embeddings.
    """
    # Convert text to CLIP embedding
    # Get local process info
    # local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    # # Determine device if not provided
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model, preprocess = get_clip_model(device=device)
    try:
        with torch.no_grad():
            
            text_token = clip.tokenize([text]).to(device)
            text_embedding = model.encode_text(text_token).detach().cpu()

             # Convert image to CLIP embedding
    
        
            image_input = preprocess(image).unsqueeze(0).to(device)
            image_embedding = model.encode_image(image_input).detach().cpu()

            # Compute cosine similarity
            cosine_similarity = torch.nn.functional.cosine_similarity(text_embedding, image_embedding).item()
    
            # Convert similarity to distance (1 - similarity)
            cosine_distance = 1 - cosine_similarity
            return cosine_distance
    except Exception as e:
        print(f"CLIP processing error: {e}")
        return 0.0


def clip_image_image_distance(image1: Image.Image, image2: Image.Image, device=None) -> float:
    """
    Computes the cosine distance between two images using CLIP embeddings.
    
    Args:
        image1 (Image): First PIL Image.
        image2 (Image): Second PIL Image.
        device (str, optional): Device to run CLIP on. Defaults to "cpu".
    
    Returns:
        float: Cosine distance between the two image embeddings.
    """
    # Determine device if not provided
    if device is None:
        device = "cpu"
    
    model, preprocess = get_clip_model(device=device)
    try:
        with torch.no_grad():
            # Convert first image to CLIP embedding
            image1_input = preprocess(image1).unsqueeze(0).to(device)
            image1_embedding = model.encode_image(image1_input).detach().cpu()
            
            # Convert second image to CLIP embedding
            image2_input = preprocess(image2).unsqueeze(0).to(device)
            image2_embedding = model.encode_image(image2_input).detach().cpu()

            # Normalize embeddings
            image1_embedding = image1_embedding / image1_embedding.norm(dim=-1, keepdim=True)
            image2_embedding = image2_embedding / image2_embedding.norm(dim=-1, keepdim=True)

            # Compute cosine similarity
            cosine_similarity = torch.nn.functional.cosine_similarity(image1_embedding, image2_embedding).item()
    
            # Convert similarity to distance (1 - similarity)
            cosine_distance = 1 - cosine_similarity
            return cosine_distance
    except Exception as e:
        print(f"CLIP processing error: {e}")
        return 1.0  # Return maximum distance on error

def clip_text_image_distances_batch(texts: Union[str, List[str]], images: Union[Image.Image, List[Image.Image]], device=None) -> Union[float, List[float]]:
    """
    Computes the cosine distance between texts and images using CLIP embeddings in batch mode.
    
    Args:
        texts: Either a single text string or a list of text strings.
        images: Either a single PIL Image or a list of PIL Images.
        batch_size: Maximum number of samples to process in one batch.
    
    Returns:
        If both inputs are single items: a float representing the distance
        If either input is a list: a list of distances
    """
    # Handle single inputs
    single_text = isinstance(texts, str)
    single_image = isinstance(images, Image.Image)
    
    if single_text:
        texts = [texts]
    if single_image:
        images = [images]
    
    # Make sure text and image lists have the same length
    # if len(texts) != len(images):
    #     raise ValueError(f"Number of texts ({len(texts)}) must match number of images ({len(images)})")
    
    # Determine device
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    # # Determine device if not provided
    if device is None:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"clip_text_image_distance_batch: device: {device}")
    # device = "cpu"
    # print(f"clip_text_image_distance_batch: device: {device}")
    # Get model and preprocess function
    model, preprocess = get_clip_model(device=device)
    
    distances = []
    
    # Process in batches
    # for i in range(0, len(texts), batch_size):
    batch_texts = texts
    batch_images = images
    
    # Keep track of None images
    valid_indices = []
    valid_images = []
    for i, img in enumerate(batch_images):
        if img is not None:
            valid_indices.append(i)
            valid_images.append(prepare_image(img))
    
    # Initialize distances with zeros (default value for None images)
    distances = [1.0] * len(batch_texts)
    
    # Only process if we have valid images
    if valid_images:
        with torch.no_grad():
            # Process text batch - only for valid indices
            valid_texts = [batch_texts[i] for i in valid_indices]
            text_tokens = clip.tokenize(valid_texts).to(device)
            text_embeddings = model.encode_text(text_tokens)
            
            # Process image batch - only valid images
            image_inputs = torch.stack([preprocess(img) for img in valid_images]).to(device)
            image_embeddings = model.encode_image(image_inputs)
            
            # Normalize embeddings
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            
            # Compute similarities (dot product)
            similarities = torch.sum(text_embeddings * image_embeddings, dim=-1)
            # Convert similarities to distances
            valid_distances = 1.0 - similarities
            
            # Update distances for valid indices
            for idx, valid_idx in enumerate(valid_indices):
                distances[valid_idx] = valid_distances[idx]
                
    # except Exception as e:
    #     print(f"CLIP batch processing error: {e}")
    #     # Fill remaining results with zeros if an error occurs
    #     remaining = len(texts) - len(distances)
    #     distances.extend([0.0] * remaining)
    
    # Return single value if both inputs were single items
    if single_text and single_image and len(distances) == 1:
        return distances[0]
    
    return distances
# Example Usage

def clip_image_image_distances_batch(
    reference_images: Union[Image.Image, List[Image.Image]], 
    query_images: Union[Image.Image, List[Image.Image]], 
    device=None
) -> Union[float, List[float]]:
    """
    Computes the cosine distance between reference images and query images using CLIP embeddings.
    
    Args:
        reference_images: Either a single PIL Image or a list of PIL Images.
        query_images: Either a single PIL Image or a list of PIL Images.
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
    
    print(f"clip_image_image_distances_batch: device: {device}")
    
    # Get model and preprocess function
    model, preprocess = get_clip_model(device=device)
    
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
                ref_inputs = torch.stack([preprocess(img) for img in valid_ref_images]).to(device)
                ref_embeddings = model.encode_image(ref_inputs)
                
                # Normalize embeddings
                ref_embeddings = ref_embeddings / ref_embeddings.norm(dim=-1, keepdim=True)
                
                # Process all query images at once
                query_inputs = torch.stack([preprocess(img) for img in valid_query_images]).to(device)
                query_embeddings = model.encode_image(query_inputs)
                
                # Normalize embeddings
                query_embeddings = query_embeddings / query_embeddings.norm(dim=-1, keepdim=True)
                
                # Compute similarities (dot product)
                similarities = torch.mm(query_embeddings, ref_embeddings.t())
                
                # Get similarity for corresponding pairs
                for i, query_idx in enumerate(valid_query_indices):
                    ref_position = valid_ref_indices.index(query_idx) if query_idx in valid_ref_indices else -1
                    if ref_position >= 0:
                        similarity = similarities[i, ref_position].item()
                        distances[query_idx] = 1.0 - similarity
            except RuntimeError as e:
                print(f"Error processing images in one batch: {e}")
                print("Consider using the batched version for large datasets")
                # Fall back to default distances (1.0)
    
    # Return single value if both inputs were single items
    if single_reference and single_query and len(distances) == 1:
        return distances[0]
    
    return distances
def prepare_image(img):
    """Handle RGBA images consistently by converting to RGB with white background"""
    if img.mode == 'RGBA':
        white_bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        white_bg.paste(img, mask=img.split()[3])
        return white_bg.convert('RGB')
    elif img.mode != 'RGB':
        return img.convert('RGB')
    return img

def clip_image_image_pixel_distances_batch(
    reference_images: Union[Image.Image, List[Image.Image]], 
    query_images: Union[Image.Image, List[Image.Image]]
) -> Union[float, List[float]]:
    """
    Computes the pixel-wise distance between reference images and query images.
    
    Args:
        reference_images: Either a single PIL Image or a list of PIL Images.
        query_images: Either a single PIL Image or a list of PIL Images.
    
    Returns:
        If both inputs are single items: a float representing the distance
        If either input is a list: a list of distances between 0 (identical) and 1 (maximum difference)
    """
    # Handle single inputs
    single_reference = isinstance(reference_images, Image.Image)
    single_query = isinstance(query_images, Image.Image)
    
    if single_reference:
        reference_images = [reference_images]
    if single_query:
        query_images = [query_images]
    
    # Define transform for consistent sizing and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # This scales pixel values to 0-1
    ])
    
    # Filter out None images and track valid indices
    valid_ref_indices = []
    valid_ref_images = []
    for i, img in enumerate(reference_images):
        if img is not None:
            valid_ref_indices.append(i)
            valid_ref_images.append(img)
    
    valid_query_indices = []
    valid_query_images = []
    for i, img in enumerate(query_images):
        if img is not None:
            valid_query_indices.append(i)
            valid_query_images.append(img)
    
    # Initialize distances with default value (1.0 means maximum distance)
    distances = [1.0] * len(reference_images)
    
    # Only process if we have valid images in both sets
    if valid_ref_images and valid_query_images:
        with torch.no_grad():
            # Process each valid reference and query image pair
            for ref_idx, ref_img in zip(valid_ref_indices, valid_ref_images):
                if ref_idx < len(query_images) and ref_idx in valid_query_indices:
                    query_img = prepare_image(query_images[ref_idx])
                    
                    
                    
                    # Convert images to tensors
                    ref_tensor = transform(prepare_image(ref_img))
                    query_tensor = transform(query_img)
                    
                    # Calculate absolute pixel-wise difference
                    diff = torch.abs(ref_tensor - query_tensor)
                    
                    # Average across all pixels to get a single distance value (0-1 range)
                    distance = diff.mean().item()
                    
                    # Store the distance
                    distances[ref_idx] = distance
    
    # Return single value if both inputs were single items
    if single_reference and single_query and len(distances) == 1:
        return distances[0]
    
    return distances


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
    
    feature_extractor = get_vgg_model(model_name="vgg19", layer_index=layer_index, device=device)
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
                        distances[query_idx] = min(1.0, mse / 100.0)
            except RuntimeError as e:
                print(f"Error processing images in one batch: {e}")
                print("Consider processing in smaller batches for large datasets")
                # Fall back to default distances (1.0)
    
    # Return single value if both inputs were single items
    if single_reference and single_query and len(distances) == 1:
        return distances[0]
    
    return distances


def dinov2_image_image_distances_batch(
    reference_images: Union[Image.Image, List[Image.Image]], 
    query_images: Union[Image.Image, List[Image.Image]], 
    model_name="dinov2_vits14",
    device=None
) -> Union[float, List[float]]:
    """
    Computes the feature distance between reference images and query images using DinoV2 features.
    
    Args:
        reference_images: Either a single PIL Image or a list of PIL Images.
        query_images: Either a single PIL Image or a list of PIL Images.
        model_name: DinoV2 model variant to use ("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14").
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
    
    print(f"dinov2_image_image_distances_batch: device: {device}")
    
    # Get the DinoV2 model with caching
    model = get_dinov2_model(model_name=model_name, device=device)
    
    # Define preprocessing for DinoV2
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
                ref_features = model(ref_tensors)
                
                # Normalize features (DinoV2 outputs are typically already normalized, but ensure it)
                ref_features = ref_features / ref_features.norm(dim=1, keepdim=True)
                
                # Process all query images at once
                query_tensors = torch.stack([preprocess(img) for img in valid_query_images]).to(device)
                query_features = model(query_tensors)
                query_features = query_features / query_features.norm(dim=1, keepdim=True)
                
                # Get features for corresponding pairs and compute distances
                for i, query_idx in enumerate(valid_query_indices):
                    ref_position = valid_ref_indices.index(query_idx) if query_idx in valid_ref_indices else -1
                    if ref_position >= 0:
                        # Extract features for this specific pair
                        query_feat = query_features[i]
                        ref_feat = ref_features[ref_position]
                        
                        # Calculate cosine distance (1 - cosine similarity)
                        cosine_sim = torch.sum(query_feat * ref_feat).item()
                        distances[query_idx] = 1.0 - cosine_sim
            except RuntimeError as e:
                print(f"Error processing images in one batch: {e}")
                print("Consider processing in smaller batches for large datasets")
                # Fall back to default distances (1.0)
    
    # Return single value if both inputs were single items
    if single_reference and single_query and len(distances) == 1:
        return distances[0]
    
    return distances


if __name__ == "__main__":
    svg_code = """<!-- To create an SVG for a clipboard icon that could represent a medical file, I'll design a simple version with a black and white fill. The clipboard will consist of a rectangle and two overlapping triangles on top. -->
<svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<rect x="20" y="40" width="60" height="20" fill="black"/>
<polygon points="50,30 80,70 20,70" fill="white"/>
<polygon points="50,30 80,70 20,70" fill="black" opacity="0.5"/>
</svg>"""
    svg_code_ref = """
<svg version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
	 viewBox="0 0 512 512" style="enable-background:new 0 0 512 512;" xml:space="preserve">
<g>
	<polygon style="fill:#F4B2B0;" points="97.409,61.893 255.999,61.893 149.583,193.179 	"/>
	<polygon style="fill:#F4B2B0;" points="414.589,61.893 255.999,61.893 362.415,193.179 	"/>
	<polygon style="fill:#F4B2B0;" points="146.993,194.664 255.999,446.434 365.006,194.664 	"/>
</g>
<path style="fill:#B3404A;" d="M511.985,194.034c0.017-2.515-0.654-5.029-2.054-7.225L425.782,54.76
	c-0.038-0.061-0.085-0.114-0.123-0.174c-0.032-0.048-0.054-0.1-0.086-0.146c-0.066-0.098-0.147-0.183-0.216-0.279
	c-0.202-0.281-0.411-0.555-0.63-0.816c-0.107-0.127-0.216-0.252-0.329-0.376c-0.249-0.275-0.508-0.535-0.776-0.784
	c-0.096-0.089-0.187-0.181-0.284-0.265c-0.76-0.668-1.585-1.24-2.458-1.708c-0.1-0.053-0.202-0.098-0.303-0.149
	c-0.337-0.17-0.681-0.328-1.03-0.468c-0.147-0.06-0.296-0.114-0.445-0.169c-0.328-0.118-0.66-0.222-0.995-0.315
	c-0.146-0.04-0.292-0.084-0.439-0.119c-0.472-0.113-0.948-0.206-1.429-0.267c-0.005,0-0.009-0.001-0.015-0.003
	c-0.502-0.062-1.007-0.089-1.514-0.093c-0.04,0-0.078-0.007-0.118-0.007H255.999H97.409c-4.398,0-8.511,2.179-10.981,5.818
	c-2.47,3.64-2.977,8.267-1.352,12.355l45.255,113.875H37.457l29.313-46.001c3.939-6.182,2.122-14.385-4.06-18.324
	c-6.182-3.939-14.384-2.123-18.324,4.06L2.069,186.81c-1.403,2.2-2.074,4.717-2.054,7.236c-0.162,3.486,1.033,7.031,3.626,9.761
	l242.738,255.441c0.072,0.076,0.151,0.139,0.224,0.212c0.077,0.077,0.145,0.161,0.223,0.236c0.098,0.093,0.203,0.173,0.303,0.263
	c0.161,0.145,0.321,0.287,0.487,0.422c0.183,0.15,0.37,0.291,0.559,0.429c0.17,0.125,0.34,0.248,0.515,0.364
	c0.199,0.131,0.402,0.252,0.605,0.372c0.173,0.101,0.344,0.204,0.52,0.297c0.218,0.115,0.441,0.218,0.664,0.32
	c0.17,0.078,0.338,0.159,0.511,0.23c0.24,0.098,0.484,0.18,0.729,0.264c0.162,0.056,0.322,0.117,0.487,0.165
	c0.265,0.08,0.535,0.141,0.804,0.203c0.147,0.035,0.293,0.076,0.443,0.105c0.301,0.058,0.604,0.098,0.908,0.135
	c0.121,0.015,0.242,0.038,0.362,0.05c0.425,0.041,0.851,0.064,1.277,0.064h0.001h0.001c0.425,0,0.849-0.023,1.273-0.062
	c0.119-0.012,0.239-0.036,0.358-0.05c0.303-0.037,0.605-0.076,0.905-0.134c0.15-0.029,0.297-0.07,0.446-0.105
	c0.267-0.061,0.534-0.122,0.796-0.2c0.165-0.049,0.326-0.11,0.488-0.165c0.242-0.082,0.483-0.163,0.721-0.26
	c0.174-0.07,0.344-0.153,0.515-0.23c0.22-0.101,0.439-0.2,0.656-0.315c0.178-0.093,0.35-0.196,0.524-0.297
	c0.202-0.118,0.402-0.236,0.6-0.366c0.175-0.115,0.345-0.239,0.515-0.362c0.188-0.137,0.374-0.276,0.557-0.425
	c0.165-0.134,0.324-0.273,0.483-0.415c0.1-0.089,0.206-0.169,0.304-0.261c0.08-0.076,0.149-0.159,0.226-0.238
	c0.073-0.073,0.151-0.135,0.223-0.21l122.146-127.72c5.066-5.297,4.879-13.698-0.418-18.765c-5.298-5.067-13.699-4.877-18.765,0.418
	l-73.842,77.211l79.323-183.21h94.601l-36.481,38.638c-5.033,5.33-4.791,13.73,0.539,18.761c2.564,2.422,5.84,3.622,9.108,3.622
	c3.525,0,7.042-1.395,9.653-4.161l56.933-60.299C510.965,201.045,512.15,197.509,511.985,194.034z M255.999,83.044l78.675,97.626
	h-157.35L255.999,83.044z M137.964,207.214l79.065,182.616L43.495,207.214H137.964z M417.34,90.907l57.203,89.764h-92.875
	L417.34,90.907z M395.033,75.165l-36.581,92.048l-74.611-92.048H395.033z M228.158,75.165l-74.611,92.048l-36.581-92.048H228.158z
	 M255.999,413.03l-88.797-205.094h177.595L255.999,413.03z"/>
<g>
</g>
<g>
</g>
<g>
</g>
<g>
</g>
<g>
</g>
<g>
</g>
<g>
</g>
<g>
</g>
<g>
</g>
<g>
</g>
<g>
</g>
<g>
</g>
<g>
</g>
<g>
</g>
<g>
</g>
</svg>
"""

    # Convert SVG to image
    image = svg_to_image(svg_code)
    image.save("test.png")
    # image_ref = svg_to_image(svg_code_ref)
    # print(clip_image_image_pixel_distances_batch(image, image_ref))
    # image.show()  # Display the image
   
    # Compute CLIP distance
    # text = "A black and white graphic representation of a percentage sign"
    # black_image = Image.new('RGB', (256, 256), color='black')
    # distance = clip_text_image_distance(text, image)
    # print(f"CLIP Distance: {1-distance, 1-clip_text_image_distance(text, black_image)}")
