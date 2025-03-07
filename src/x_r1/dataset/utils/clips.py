import cairosvg
from io import BytesIO
from PIL import Image
import torch
import clip
from torchvision import transforms
from lxml import etree


# Load CLIP model
device_id = 0  # Use the first GPU
device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

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

# Example Usage
if __name__ == "__main__":
    svg_code = """
    <svg width="300" height="400" xmlns="http://www.w3.org/2000/svg">
  <polygon x="50" y="180" fill="purple" stroke="white" stroke-width="16" points="150,220 170,300 70,300 100,220"/>
  <circle cx="100" cy="250" r="60" fill="white"/>
  <path d="M70,160 Q65,250,125,115 T190,55 Q245,15,290,55 T355,115 Q365,250,325,160 T290,90 Q275,160,245,95 T210,60 Q160,40,100,75 T15,55 Q15,160,55,250 T120,165 Q135,250,190,90 T250,55 Q260,1
    """

    # Convert SVG to image
    image = svg_to_image(svg_code)
    # image.show()  # Display the image

    # Compute CLIP distance
    text = "A tall man"
    distance = clip_text_image_distance(text, image)
    print(f"CLIP Distance: {distance}")
