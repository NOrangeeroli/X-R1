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
    device=None, 
    batch_size=32
) -> Union[float, List[float]]:
    """
    Computes the cosine distance between reference images and query images using CLIP embeddings in batch mode.
    
    Args:
        reference_images: Either a single PIL Image or a list of PIL Images.
        query_images: Either a single PIL Image or a list of PIL Images.
        device: Device to run the model on.
        batch_size: Maximum number of samples to process in one batch.
    
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
        # Process images in batches to avoid memory issues
        ref_embeddings_list = []
        
        # Process reference images in batches
        for i in range(0, len(valid_ref_images), batch_size):
            batch_ref_images = valid_ref_images[i:i+batch_size]
            
            with torch.no_grad():
                # Process reference image batch
                ref_inputs = torch.stack([preprocess(img) for img in batch_ref_images]).to(device)
                batch_ref_embeddings = model.encode_image(ref_inputs)
                
                # Normalize embeddings
                batch_ref_embeddings = batch_ref_embeddings / batch_ref_embeddings.norm(dim=-1, keepdim=True)
                ref_embeddings_list.append(batch_ref_embeddings)
        
        # Concatenate all reference embeddings
        if len(ref_embeddings_list) > 1:
            ref_embeddings = torch.cat(ref_embeddings_list, dim=0)
        else:
            ref_embeddings = ref_embeddings_list[0]
            
        # Process query images in batches
        for i in range(0, len(valid_query_images), batch_size):
            batch_query_images = valid_query_images[i:i+batch_size]
            batch_query_indices = valid_query_indices[i:i+batch_size]
            
            with torch.no_grad():
                # Process query image batch
                query_inputs = torch.stack([preprocess(img) for img in batch_query_images]).to(device)
                query_embeddings = model.encode_image(query_inputs)
                
                # Normalize embeddings
                query_embeddings = query_embeddings / query_embeddings.norm(dim=-1, keepdim=True)
                
                # Compute similarities (dot product)
                similarities = torch.mm(query_embeddings, ref_embeddings.t())
                
                # Get similarity for corresponding pairs
                for idx, query_idx in enumerate(batch_query_indices):
                    if query_idx < len(ref_embeddings):
                        similarity = similarities[idx, query_idx].item()
                        distances[query_idx] = 1.0 - similarity
    
    # Return single value if both inputs were single items
    if single_reference and single_query and len(distances) == 1:
        return distances[0]
    
    return distances

if __name__ == "__main__":
    svg_code = """
    <svg width="128" height="128" style="enable-background:new 0 0 128 128;" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<linearGradient id="SVGID_1_" gradientTransform="matrix(1 0 0 -1 0 130)" gradientUnits="userSpaceOnUse" x1="27.61" x2="105.4818" y1="97.1075" y2="97.1075">
<stop offset="0" style="stop-color:#F44336"/>
<stop offset="0.3592" style="stop-color:#E73B32"/>
<stop offset="1" style="stop-color:#C62828"/>
</linearGradient>
<path d="M90.92,11.55C85.56,7.07,77.56,4.3,63.59,4.3c-25.64,0-35.98,16.53-35.98,36.87 c0,0.08,2.42,0.25,2.91,0.29c0.06,0,0.11,0.01,0.17,0.01L64,32c0,0,30.06,9.38,30.31,10.22l5.69,17.51 c0.31,1.04,1.26,1.75,2.35,1.75c1.29,0,2.36-1,2.45-2.29c0.18-2.87,0.48-8.01,0.62-12.9C105.62,38.86,106.28,24.38,90.92,11.55z" style="fill:url(#SVGID_1_);"/>
<ellipse cx="101.91" cy="63.86" rx="6.75" ry="6.75" style="fill:#E0E0E0;"/>
<g id="robe">
<linearGradient id="SVGID_2_" gradientTransform="matrix(1 0 0 -1 0 130)" gradientUnits="userSpaceOnUse" x1="64" x2="64" y1="33.1589" y2="4.6815">
<stop offset="0" style="stop-color:#F44336"/>
<stop offset="0.3592" style="stop-color:#E73B32"/>
<stop offset="1" style="stop-color:#C62828"/>
</linearGradient>
<path d="M64.14,95.97H64c-25.65,0.03-52,7.1-52,24.99V124h1.45h1.44h98.22h1.46H116v-3.04 C116,104.1,89.9,95.97,64.14,95.97z" style="fill:url(#SVGID_2_);"/>
<polygon points="55.33,105.67 55.33,124 56.83,124 58.33,124 69.67,124 71.16,124 72.66,124 72.66,105.67 " style="fill:#FFFFFF;"/>
</g>
<path id="ears_1_" d="M90.76,47.55H37.19c-5.78,0-10.5,5.17-10.5,11.5s4.73,11.5,10.5,11.5h53.57 c5.78,0,10.5-5.18,10.5-11.5S96.54,47.55,90.76,47.55z" style="fill:#E59600;"/>
<path id="head" d="M35.75,34.96c-2.32,6.06-3.55,12.89-3.55,20.05c0,11.73,3.39,21.52,9.81,28.3 c5.65,5.97,13.31,9.26,21.58,9.26c8.26,0,15.93-3.29,21.58-9.26c6.42-6.78,9.81-16.57,9.81-28.3c0-7.17-1.24-13.99-3.55-20.05H35.75 z" style="fill:#FFCA28;"/>
<g id="eyes">
<ellipse cx="47.68" cy="59.26" rx="4.83" ry="5.01" style="fill:#404040;"/>
<ellipse cx="80.32" cy="59.26" rx="4.83" ry="5.01" style="fill:#404040;"/>
</g>
<g id="eyebrows_x5F_white">
<path d="M87.41,50.44L87.41,50.44c0-0.01-2.25-3.67-7.48-3.67c-5.23,0-7.49,3.66-7.49,3.66l0,0.01 c-0.18,0.26-0.3,0.58-0.3,0.93c0,0.88,0.69,1.6,1.55,1.6c0.18,0,0.61-0.13,0.65-0.15c3.13-1.33,5.59-1.34,5.59-1.34 s2.43,0.01,5.57,1.34c0.04,0.02,0.47,0.15,0.65,0.15c0.86,0,1.55-0.72,1.55-1.6C87.71,51.03,87.6,50.71,87.41,50.44z" style="fill:#F5F5F5;"/>
<path d="M55.53,50.44L55.53,50.44c0.01-0.01-2.25-3.67-7.48-3.67s-7.49,3.66-7.49,3.66l0.01,0.01 c-0.18,0.26-0.3,0.58-0.3,0.93c0,0.88,0.69,1.6,1.55,1.6c0.18,0,0.61-0.13,0.65-0.15c3.13-1.33,5.59-1.34,5.59-1.34 s2.44,0.01,5.57,1.34c0.04,0.02,0.47,0.15,0.65,0.15c0.86,0,1.55-0.72,1.55-1.6C55.83,51.03,55.72,50.71,55.53,50.44z" style="fill:#F5F5F5;"/>
</g>
<g>
<linearGradient id="SVGID_3_" gradientUnits="userSpaceOnUse" x1="63.999" x2="63.999" y1="100.5647" y2="20.8823">
<stop offset="0.4878" style="stop-color:#F5F5F5"/>
<stop offset="0.8314" style="stop-color:#BDBDBD"/>
</linearGradient>
<path d="M102.36,73.51c-1.74-5.25-3.83-16.73-3.66-21.6c0.22-6.4,0.24-10.71,0.2-13.78 c-0.05-3.7-3.07-6.67-6.78-6.67c0,0-0.67,0-0.69,0v6.51c0.7,3.34,0.63,9.44,0.63,17.24c0,12.46-3.52,18.85-11.82,18.85 c-8.67,0-10.15-2.55-16.59-2.55c-6.45,0-7.98,2.55-16.37,2.55c-7.57,0-11.5-8.77-11.5-19.17c0-4.57-0.2-9.04-0.19-12.58h0V31.47 c-3.66,0.1-6.6,3.08-6.6,6.77c0.01,3.06,0.08,7.35,0.35,13.67c0.21,4.99-1.97,16.71-3.71,21.98c-2.64,7.96-2.22,17.72,1.49,25.24 c3.32,6.73,19.39,17.45,28.87,22.72c3.62,2.01,12.05,2.03,15.69,0.06c9.33-5.06,24.9-15.22,28.48-21.63 C104.54,92.45,105.18,82.04,102.36,73.51z" style="fill:url(#SVGID_3_);"/>
<radialGradient id="SVGID_4_" cx="64.2726" cy="75.4103" gradientUnits="userSpaceOnUse" r="48.5567">
<stop offset="0.7063" style="stop-color:#FFFFFF;stop-opacity:0"/>
<stop offset="1" style="stop-color:#BDBDBD"/>
</radialGradient>
<path d="M102.36,73.51c-1.74-5.25-3.83-16.72-3.67-21.6c0.23-6.4,0.24-10.71,0.2-13.77 c-0.05-3.71-3.07-6.67-6.77-6.67h-0.06v23.75c0,12.46-3.52,18.84-11.81,18.84c-8.68,0-10.16-2.54-16.6-2.54 c-6.45,0-7.98,2.54-16.37,2.54c-7.57,0-11.51-8.76-11.51-19.17c0-5.32-0.27-10.51-0.16-14.25h-0.03v-9.16 c-3.66,0.09-6.6,3.08-6.59,6.77c0,3.06,0.07,7.36,0.34,13.67c0.22,4.99-1.96,16.72-3.71,21.98c-2.64,7.97-2.22,17.72,1.49,25.24 c3.33,6.74,19.39,17.45,28.87,22.72c3.62,2.02,12.05,2.04,15.69,0.06c9.33-5.06,24.9-15.22,28.49-21.63 C104.54,92.45,105.18,82.04,102.36,73.51z" style="fill:url(#SVGID_4_);"/>
</g>
<path id="mouth" d="M72.23,76.15c-3.13,1.86-13.37,1.86-16.5,0c-1.79-1.07-3.63,0.56-2.88,2.2 c0.73,1.6,6.32,5.32,11.16,5.32s10.35-3.72,11.08-5.32C75.84,76.72,74.03,75.08,72.23,76.15z" style="fill:#795548;"/>
<path id="nose" d="M67.79,68.38c-0.1-0.04-0.21-0.07-0.32-0.08h-6.94c-0.11,0.01-0.21,0.04-0.32,0.08 c-0.63,0.25-0.98,0.91-0.68,1.6s1.68,2.64,4.46,2.64c2.79,0,4.17-1.95,4.46-2.64C68.76,69.29,68.42,68.64,67.79,68.38z" style="fill:#E59600;"/>
<path d="M97.98,32.14C97.03,31.44,83,23.43,64,23.43c-19,0.01-33.03,8.01-33.98,8.72 c-3.91,2.91-5.06,6-4.47,10.86c0.34,2.85,2.97,3.93,4.66,3.34c0,0,16.26-3.93,33.78-3.94c17.53,0,33.78,3.94,33.78,3.94 c1.7,0.59,4.32-0.49,4.66-3.34C103.03,38.14,101.88,35.05,97.98,32.14z" style="fill:#F5F5F5;"/>
</svg>

    """

    # Convert SVG to image
    image = svg_to_image(svg_code)
    # image.show()  # Display the image

    # Compute CLIP distance
    text = "Santa Claus"
    black_image = Image.new('RGB', (256, 256), color='black')
    distance = clip_text_image_distance(text, image)
    print(f"CLIP Distance: {1-distance, 1-clip_text_image_distance(text, black_image)}")
