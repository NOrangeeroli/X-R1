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
from func_timeout import func_timeout, FunctionTimedOut, func_set_timeout# Cache for SigLIP models
_siglip_models = {}

@lru_cache(maxsize=300)
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
    from transformers import AutoProcessor, AutoModel, AutoTokenizer
    
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
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Freeze parameters to ensure we're only doing inference
            for param in model.parameters():
                param.requires_grad = False
                
            _siglip_models[model_key] = (model, processor, tokenizer)
            
        except Exception as e:
            raise ValueError(f"Error loading SigLIP model {model_name}: {e}")
    
    return _siglip_models[model_key]

# Cache for MAE models
_mae_models = {}

@lru_cache(maxsize=300)
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

@lru_cache(maxsize=300)
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
@lru_cache(maxsize=300)
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

@lru_cache(maxsize=300)
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



def safe_svg_to_image(svg_code, timeout=5):
    """Convert SVG to image with timeout protection."""
    # Get process info for debugging
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    
    
    # Skip empty SVG
    if not svg_code :
        print(f"[Rank {global_rank}/{local_rank}] Empty SVG")
        return None
    
    try:
        
        result = svg_to_image(svg_code)
        return result
    except FunctionTimedOut:
        
        print(f"[Rank {global_rank}/{local_rank}] SVG to image conversion timed out")
        print(f"The SVG code is: {svg_code}")
        return None
    except Exception as e:
        print(f"[Rank {global_rank}/{local_rank}] Error in svg_to_image: {type(e).__name__}: {str(e)}")
        return None
@func_set_timeout(3)
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
        
        
        # doitReturnValue = func_timeout(5, cairosvg.svg2png, args=(valid_svg,))
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

def siglip_text_image_distances_batch(texts: Union[str, List[str]], images: Union[Image.Image, List[Image.Image]], model_name="google/siglip-base-patch16-224", device=None) -> Union[float, List[float]]:
    """
    Computes the cosine distance between texts and images using SigLIP embeddings in batch mode.
    
    Args:
        texts: Either a single text string or a list of text strings.
        images: Either a single PIL Image or a list of PIL Images.
        model_name: SigLIP model to use ("google/siglip-base-patch16-224" or "google/siglip-large-patch16-224")
        device: Device to run the model on.
    
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
    
    # Determine device
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    if device is None:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"siglip_text_image_distance_batch: device: {device}")
    
    # Get model and processor
    model, processor, tokenizer = get_siglip_model(model_name=model_name, device=device)
    
    # Keep track of None images
    valid_indices = []
    valid_images = []
    for i, img in enumerate(images):
        if img is not None:
            valid_indices.append(i)
            valid_images.append(prepare_image(img))
    
    # Initialize distances with ones (maximum distance for None images)
    batch_distances = [1.0] * len(texts)
    
    # Only process if we have valid images
    if valid_images:
        with torch.no_grad():
            # Process text batch - only for valid indices
            valid_texts = [texts[i] for i in valid_indices]
            
            
            # Process all text and images in one go - use the correct processor format
            
            text_inputs = tokenizer(
                    valid_texts,
                    padding="max_length",
                    return_tensors="pt",
                )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            image_inputs = processor(
                    images=valid_images,
                    return_tensors="pt",
                )
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            
            print("input_ids shape:", text_inputs.get("input_ids").shape)
            print("input_ids dtype:", text_inputs.get("input_ids").dtype)
            print("Model type:", type(model))

            # print(valid_texts)
            # print(text_inputs)
            text_embeddings = model.get_text_features(**text_inputs)
            image_embeddings = model.get_image_features(**image_inputs)
            
            # inputs = processor(
            #         text=valid_texts,
            #         images=valid_images,
            #         return_tensors="pt",
            #         padding="max_length",
            #     ).to(device)
            
            # text_embeddings = model.get_text_features(
            #     input_ids=inputs.input_ids,
            #     attention_mask=inputs.attention_mask
            #     )
                
            # image_embeddings = model.get_image_features(
            #     pixel_values=inputs.pixel_values
            #     )
            
            # Normalize embeddings
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
                
            # inputs = processor(
            #     text=valid_texts,
            #     images=valid_images,
            #     return_tensors="pt",
            #     padding="max_length",
            # ).to(device)
            
            # # Extract features
            # outputs = model(**inputs)
            # image_embeddings = outputs.image_embeds  # shape: [1, D]
            # text_embeddings = outputs.text_embeds    # shape: [1, D]
            
            # # Normalize embeddings
            # text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            # image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            
            # Compute similarities (dot product)
            similarities = torch.sum(text_embeddings * image_embeddings, dim=-1)
            
            # Convert similarities to distances
            valid_distances = 1.0 - similarities.cpu().numpy()
            
            # Update distances for valid indices
            for idx, valid_idx in enumerate(valid_indices):
                batch_distances[valid_idx] = valid_distances[idx]
            
           
    # Return single value if both inputs were single items
    if single_text and single_image and len(batch_distances) == 1:
        return batch_distances[0]
    
    return batch_distances

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
    svg_codes = ['<svg width="400" height="400" xmlns="http://www.w3.org/2000/svg">\n  <!-- Walls -->\n  <rect x="50" y="50" width="300" height="300" fill="#f0f0f0" stroke="#000" stroke-width="2" />\n  <rect x="100" y="100" width="200" height="200" fill="#f0f0f0" stroke="#000" stroke-width="2" />\n  \n  <!-- Floor -->\n  <rect x="50" y="350" width="300" height="50" fill="#e0e0e0" stroke="#000" stroke-width="2" />\n  \n  <!-- Window -->\n  <rect x="100" y="100" width="100" height="100" fill="#ffffff" stroke="#000" stroke-width="2" />\n  <rect x="110" y="110" width="80" height="80" fill="#ffffff" stroke="#000" stroke-width="2" />\n  <rect x="110" y="110" width="80" height="80" fill="#ffffff" stroke="#000" stroke-width="2" />\n  \n  <!-- Sink -->\n  <ellipse cx="200" cy="250" rx="50" ry="20" fill="#ffffff" stroke="#000" stroke-width="2" />\n  <ellipse cx="200" cy="250" rx="40" ry="10" fill="#ffffff" stroke="#000" stroke-width="2" />\n  \n  <!-- Counter -->\n  <rect x="50" y="200" width="300" height="50" fill="#f0f0f0" stroke="#000" stroke-width="2" />\n  <rect x="50" y="200" width="300" height="20" fill="#e0e0e0" stroke="#000" stroke-width="2" />\n  \n  <!-- Appliances -->\n  <rect x="100" y="300" width="100" height="100" fill="#f0f0f0" stroke="#000" stroke-width="2" />\n  <rect x="200" y="100" width="100" height="100" fill="#f0f0f0" stroke="#000" stroke-width="2" />\n  <ellipse cx="250" cy="150" rx="50" ry="20" fill="#ffffff" stroke="#000" stroke-width="2" />\n  \n  <!-- Light -->\n  <circle cx="200" cy="100" r="10" fill="#ffffff" stroke="#000" stroke-width="2" />\n  <circle cx="200" cy="100" r="5" fill="#000" />\n  \n  <!-- Washing Machine -->\n  <rect x="200" y="300" width="100" height="100" fill="#f0f0f0" stroke="#000" stroke-width="2" />\n  <rect x="210" y="310" width="80" height="80" fill="#f0f0f0" stroke="#000" stroke-width="2" />\n  <rect x="210" y="310" width="80" height="80" fill="#f0f0f0" stroke="#000" stroke-width="2" />\n  \n  <!-- Light Bulb -->\n  <circle cx="200" cy="100" r="10" fill="#ffffff" stroke="#000" stroke-width="2" />\n  <circle cx="200" cy="100" r="5" fill="#000" />\n</svg>', '<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">\n  <!-- Background color for the sky -->\n  <rect x="0" y="0" width="600" height="200" fill="#87CEEB" />\n  \n  <!-- Streets -->\n  <path d="M0,200 Q 100,250 200,200 Q 300,150 400,200 Q 500,250 600,200" stroke="#696969" stroke-width="5" fill="none" />\n  \n  <!-- Parade elements -->\n  <path d="M100,250 C 150,240 200,230 250,240 C 300,250 350,260 400,250 C 450,240 500,230 550,240" stroke="#FFD700" stroke-width="10" fill="#FFD700" />\n  <path d="M200,260 C 210,270 220,280 230,290 C 240,300 250,310 260,320" stroke="#FFD700" stroke-width="10" fill="#FFD700" />\n  \n  <!-- People -->\n  <rect x="100" y="280" width="50" height="100" fill="tan" />\n  <rect x="200" y="280" width="50" height="100" fill="tan" />\n  <rect x="300" y="280" width="50" height="100" fill="tan" />\n  <rect x="400" y="280" width="50" height="100" fill="tan" />\n  \n  <!-- Flags -->\n  <polygon points="100,250 150,240 200,250" fill="red" stroke="black" />\n  <polygon points="200,250 250,240 300,250" fill="blue" stroke="black" />\n  \n  <!-- Balloons -->\n  <ellipse cx="150" cy="300" rx="50" ry="20" fill="green" />\n  <ellipse cx="350" cy="300" rx="50" ry="20" fill="yellow" />\n  \n  <!-- Decorations -->\n  <circle cx="100" cy="270" r="10" fill="red" />\n  <circle cx="400" cy="270" r="10" fill="blue" />\n</svg>', '<svg width="500" height="300" xmlns="http://www.w3.org/2000/svg">\n  <!-- Street -->\n  <rect x="50" y="200" width="400" height="50" fill="lightgray" stroke="black" stroke-width="2" />\n  \n  <!-- Parade Path -->\n  <path d="M100,250 C150,200 200,150 250,200 C300,150 350,200 400,250" fill="none" stroke="red" stroke-width="10" />\n  \n  <!-- People -->\n  <rect x="100" y="200" width="50" height="100" fill="lightblue" stroke="black" stroke-width="2" />\n  <circle cx="120" cy="260" r="20" fill="blue" />\n  <rect x="200" y="200" width="50" height="100" fill="lightblue" stroke="black" stroke-width="2" />\n  <circle cx="220" cy="260" r="20" fill="blue" />\n  <rect x="300" y="200" width="50" height="100" fill="lightblue" stroke="black" stroke-width="2" />\n  <circle cx="320" cy="260" r="20" fill="blue" />\n  <rect x="400" y="200" width="50" height="100" fill="lightblue" stroke="black" stroke-width="2" />\n  <circle cx="420" cy="260" r="20" fill="blue" />\n  \n  <!-- Street Details -->\n  <rect x="50" y="250" width="100" height="10" fill="black" stroke="black" stroke-width="2" />\n  <rect x="400" y="250" width="100" height="10" fill="black" stroke="black" stroke-width="2" />\n  \n  <!-- Parade Details -->\n  <path d="M150,200 C200,150 250,200 C300,150 350,200" fill="none" stroke="red" stroke-width="10"/>\n  <path d="M200,200 C220,180 240,200 C260,220 280,200" fill="none" stroke="red" stroke-width="10" />\n</svg>', '<svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">\n  <!-- Background -->\n  <rect width="100%" height="100%" fill="#f0f0f0" />\n\n  <!-- Parade Route -->\n  <path d="M50,100 C100,150 200,100 300,150" stroke="#000" stroke-width="5" fill-opacity="0.5" />\n  \n  <!-- People -->\n  <g fill="rgba(255,255,255,0.8)">\n    <ellipse cx="100" cy="200" rx="30" ry="10" fill="#ffffff" stroke="#b6b698"/>\n    <g>\n      <path class="st0" d="M298873294L843995283H8"></path>\n      <ellipse class="st2" fill-rule="evenodd" fill="#FEFEFE" opacity="1"/>\n      <circle opacity="1"></circle>\n    </g>\n  </g>\n  \n  <!-- Flags -->\n  <g stroke-miterlimit="4">\n    <polygon fill="#FF5234" opacity=".6">\n      <path d="M677596H921v3l395v3H0v3C-5 3-4C5v3C899M39H71">\n        <stroke opacity=".5"/><path d="-69"/>\n        <stroke opacity=".5"/><path d="-9"/>\n      </path>\n    </polygon>\n  </g>\n  \n  <!-- Balloons -->\n  <g>\n    <ellipse stroke="#F6E0D3"></ellipse>\n  </g>\n  \n  <!-- Parade Floats -->\n  <!-- Add more details like floats, balloons, and flags as needed -->\n</svg>']

    # Convert SVG to image
    for i, svg_code in enumerate(svg_codes):
        print(f"Processing SVG {i+1}")
        print(safe_svg_to_image(svg_code))
    # image_ref = svg_to_image(svg_code_ref)
    # print(clip_image_image_pixel_distances_batch(image, image_ref))
    # image.show()  # Display the image
   
    # Compute CLIP distance
    # text = "A black and white graphic representation of a percentage sign"
    # black_image = Image.new('RGB', (256, 256), color='black')
    # distance = clip_text_image_distance(text, image)
    # print(f"CLIP Distance: {1-distance, 1-clip_text_image_distance(text, black_image)}")
