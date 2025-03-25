from ..reward import SVGReward
extract_svg = SVGReward.extract_svg
from .clips import svg_to_image, safe_svg_to_image

def render_svg_from_text(text):
    """
    Extract SVG code from text and render it as an image.
    
    Args:
        text (str): The text containing SVG code
        
    Returns:
        PIL.Image.Image or None: Rendered image if successful, None otherwise
    """
    
    try:
        # Extract SVG code from the text
        svg_code = extract_svg(text)
        
        # If no SVG code was found, return None
        if not svg_code:
            print("No SVG code found in the text")
            return None
        
        # Convert SVG code to image
        image = safe_svg_to_image(svg_code)
        
        return image
    except Exception as e:
        print(f"Error rendering SVG from text: {e}")
        return None