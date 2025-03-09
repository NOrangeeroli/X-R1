import cairosvg
from io import BytesIO
from PIL import Image
import torch
import clip
from torchvision import transforms
from lxml import etree
from functools import lru_cache
import os

from typing import Union, List

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
    if len(texts) != len(images):
        raise ValueError(f"Number of texts ({len(texts)}) must match number of images ({len(images)})")
    
    # # Determine device
    # local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    # # Determine device if not provided
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"clip_text_image_distance_batch: device: {device}")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
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
            valid_images.append(img)
    
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
            similarities = torch.sum(text_embeddings * image_embeddings, dim=-1).cpu().numpy()
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
if __name__ == "__main__":
    svg_code = """
    <svg width="800" height="500" viewBox="0 0 800 500" xmlns="http://www.w3.org/2000/svg">
    <!-- Background -->
    <rect width="100%" height="100%" fill="#87CEEB"/>
    
    <!-- Grass -->
    <rect y="350" width="100%" height="150" fill="#228B22"/>
    
    <!-- Houses -->
    <rect x="100" y="200" width="150" height="150" fill="#8B0000"/>
    <polygon points="100,200 175,130 250,200" fill="#A52A2A"/>
    <rect x="120" y="250" width="50" height="100" fill="#654321"/>
    
    <rect x="400" y="200" width="150" height="150" fill="#8B0000"/>
    <polygon points="400,200 475,130 550,200" fill="#A52A2A"/>
    <rect x="420" y="250" width="50" height="100" fill="#654321"/>
    
    <!-- Flowers -->
    <g transform="translate(300, 400)">
        <circle cx="0" cy="-10" r="10" fill="#FF0000"/>
        <circle cx="-10" cy="0" r="10" fill="#FF0000"/>
        <circle cx="10" cy="0" r="10" fill="#FF0000"/>
        <circle cx="0" cy="10" r="10" fill="#FF0000"/>
        <rect x="-2" y="10" width="4" height="20" fill="#008000"/>
    </g>
    
    <g transform="translate(500, 410)">
        <circle cx="0" cy="-10" r="10" fill="#FF00FF"/>
        <circle cx="-10" cy="0" r="10" fill="#FF00FF"/>
        <circle cx="10" cy="0" r="10" fill="#FF00FF"/>
        <circle cx="0" cy="10" r="10" fill="#FF00FF"/>
        <rect x="-2" y="10" width="4" height="20" fill="#008000"/>
    </g>
    
    <g transform="translate(600, 390)">
        <circle cx="0" cy="-10" r="10" fill="#FFFF00"/>
        <circle cx="-10" cy="0" r="10" fill="#FFFF00"/>
        <circle cx="10" cy="0" r="10" fill="#FFFF00"/>
        <circle cx="0" cy="10" r="10" fill="#FFFF00"/>
        <rect x="-2" y="10" width="4" height="20" fill="#008000"/>
    </g>
</svg>

    """

    # Convert SVG to image
    image = svg_to_image(svg_code)
    # image.show()  # Display the image

    # Compute CLIP distance
    text = "A number of flowers outside some houses"
    black_image = Image.new('RGB', (256, 256), color='black')
    distance = clip_text_image_distance(text, image)
    print(f"CLIP Distance: {distance, clip_text_image_distance(text, black_image)}")
