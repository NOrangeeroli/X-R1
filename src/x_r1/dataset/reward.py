import re
from typing import Dict
import os
from openai import OpenAI
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
# Add this at the top of the file with other imports
import os
import torch
import datetime
import time

# Add this utility function for debug printing
def debug_print(message):
    """Print debug information with rank information."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    # Get GPU memory info if using CUDA
    mem_info = ""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(device) / (1024**3)    # GB
        mem_info = f", Memory: {allocated:.2f}GB/{reserved:.2f}GB"
    
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] Reward Rank {global_rank}/{local_rank} ({world_size}){mem_info}: {message}")
class Reward:
    """Reward functions for the XDG dataset"""
    
    @staticmethod
    def normalize_text(text):
        """Normalize text by removing extra whitespace, converting to lowercase."""
        if text is None:
            return ""
        # Remove extra whitespace and convert to lowercase
        text = re.sub(r'\s+', ' ', text.strip().lower())
        return text
    
    @staticmethod
    def extract_answer(text):
        """Extract content between <answer> tags."""
        if text is None:
            return ""
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    @staticmethod
    def evaluate_answer_similarity(answer, solution):
        """Use GPT4O-mini to evaluate answer similarity."""
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical answer evaluator. Compare the student's answer with the correct solution and output ONLY '1.0' if they match in meaning, or '0.0' if they don't match. No other output is allowed."
                    },
                    {
                        "role": "user",
                        "content": f"Student answer: {answer}\nCorrect solution: {solution}\nOutput only 1.0 or 0.0:"
                    }
                ],
                temperature=0
            )
            result = response.choices[0].message.content.strip()
            return float(result)
        except Exception as e:
            print(f"Error in GPT evaluation: {e}")
            # If API call fails, fall back to simple text matching
            return 1.0 if Reward.normalize_text(answer) == Reward.normalize_text(solution) else 0.0
    


    
    @staticmethod
    def accuracy_reward(completions, solution, **kwargs):
        """Reward function that checks if the completion is the same as the ground truth."""
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for content, sol in zip(contents, solution):
            # First try latex parsing
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            gold_parsed2 = parse(
                f"${sol}$",
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) != 0 or len(gold_parsed2) != 0:
                # print('latex gold parsed')
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_text = Reward.extract_answer(content)
                answer_parsed = parse(
                    answer_text,
                    # extraction_config=[
                    #     LatexExtractionConfig(
                    #         normalization_config=NormalizationConfig(
                    #             nits=False,
                    #             malformed_operators=False,
                    #             basic_latex=True,
                    #             equations=True,
                    #             boxed="all",
                    #             units=True,
                    #         ),
                    #         # Ensures that boxed is tried first
                    #         boxed_match_priority=0,
                    #         try_extract_without_anchor=False,
                    #     )
                    # ],
                    extraction_mode="first_match",
                )
                answer_parsed2 = parse(f"${Reward.normalize_text(answer_text)}$")
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = 0.0
                for g in [gold_parsed, gold_parsed2]:
                    for a in [answer_parsed, answer_parsed2]:
                        if verify(a, g) == 1:
                            reward = 1.0
                            break
                
                    
                    
                # print('\nprompt:', prompt)
                print('-'*100)
                print(f"\nanswer text: {answer_text}\n")
                print(f"\solution text: {sol}\n")
                print('\nanswer_parsed:', answer_parsed, answer_parsed2, '\ngold_parsed:', gold_parsed, gold_parsed2,'\nreward:', reward, '\n')
            else:
                # For medical text answers, extract from <answer> tags and use GPT4O-mini for evaluation
                # answer_content = XDGReward.extract_answer(content)
                # normalized_content = XDGReward.normalize_text(answer_content)
                # normalized_solution = XDGReward.normalize_text(sol)
                # reward = XDGReward.evaluate_answer_similarity(normalized_content, normalized_solution)
                reward = 0.0
            rewards.append(reward)

        #print('\naccuracy rewards:', rewards)

        return rewards


        
    
    
    
    @staticmethod
    def format_reward(completions, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<think>.*?</think><answer>.*?</answer>$"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content) for content in completion_contents]

        rewards = [1.0 if match else 0.0 for match in matches]
        return rewards
    
    
from .utils.clips import  safe_svg_to_image,clip_text_image_distances_batch, clip_image_image_distances_batch, clip_image_image_pixel_distances_batch,vgg_image_image_distances_batch,dinov2_image_image_distances_batch,siglip_text_image_distances_batch

class SVGReward:
    @staticmethod
    def extract_answer(text):
        """Extract content between <answer> tags."""
        if text is None:
            return ""
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return SVGReward.extract_answer_half(text)
    @staticmethod
    def extract_answer_half(text):
        """Extract content between <answer> tags."""
        if text is None:
            return ""
        match = re.search(r'<answer>(.*?)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def extract_svg(text):
        if text is None:
            return ""
        # ans = SVGReward.extract_answer(text)
        ans = text
        match = re.search(r'(<svg .*?</svg>)', ans, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            match = re.search(r'(<svg.*?)', ans, re.DOTALL)
            if match:
                return match.group(1).strip() 
            
            return ""
    @staticmethod
    def single_format_reward(content, **kwargs):
        """Reward function that checks if the completion has a specific format with exactly one of each tag."""
        # content = completion[0]["content"]
        
        
        # Check if the overall structure is correct
        structure_match = re.match(r"^<think>.*?</think>\n<answer>.*?</answer>$", content, re.DOTALL)
        
        # Count occurrences of each tag
        think_open_count = content.count("<think>")
        think_close_count = content.count("</think>")
        answer_open_count = content.count("<answer>")
        answer_close_count = content.count("</answer>")
        
        # Check if exactly one of each tag exists
        tags_valid = (think_open_count == 1 and 
                    think_close_count == 1 and 
                    answer_open_count == 1 and 
                    answer_close_count == 1)
        no_text = "</text>" not in content    
        # Reward is 1.0 only if both structure and tag counts are correct
        
        reward = 0.5 if (structure_match and tags_valid and no_text) else 0.0
        # reward = 0.5 if no_text else 0.0
        
        
        return reward
 
    @staticmethod
    def format_reward(completions, **kwargs):
        """Reward function that checks if the completion has a specific format with exactly one of each tag."""
        return [SVGReward.single_format_reward(completion[0]["content"]) for completion in completions]
        

    @staticmethod
    def accuracy_reward(completions, solution, **kwargs):
        
        
        
        completion_contents = [completion[0]["content"] for completion in completions]
        ans = [SVGReward.extract_svg(content) for content in completion_contents]
        rewards = []
        images = [safe_svg_to_image(content) for content in ans]
        
        distances = clip_text_image_distances_batch(solution, images)
        rewards = [1.0 - distance for distance in distances]
        return rewards
    @staticmethod
    def perceptual_reward(completions, svg, **kwargs):
        
        
        
        completion_contents = [completion[0]["content"] for completion in completions]
        ans = [SVGImageReward.extract_svg(content) for content in completion_contents]
        ans_ref = [SVGImageReward.extract_svg(content) for content in svg]
        rewards = []
        images = [safe_svg_to_image(content) for content in ans]
        ref_images = [safe_svg_to_image(content) for content in ans_ref]
        
        distances = dinov2_image_image_distances_batch(ref_images, images)
        rewards = [1.0 - distance for distance in distances]
        return rewards
        
class SVGImageReward:
    @staticmethod
    def extract_answer(text):
        """Extract content between <answer> tags."""
        if text is None:
            return ""
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return SVGImageReward.extract_answer_half(text)
    @staticmethod
    def extract_answer_half(text):
        """Extract content between <answer> tags."""
        if text is None:
            return ""
        match = re.search(r'<answer>(.*?)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def extract_svg(text):
        if text is None:
            return ""
        # ans = SVGReward.extract_answer(text)
        ans = text
        match = re.search(r'(<svg .*?</svg>)', ans, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            match = re.search(r'(<svg.*?)', ans, re.DOTALL)
            if match:
                return match.group(1).strip() 
            
            return ""
    @staticmethod
    def single_format_reward(content, **kwargs):
        """Reward function that checks if the completion has a specific format with exactly one of each tag."""
        # content = completion[0]["content"]
        
        
        # Check if the overall structure is correct
        structure_match = re.match(r"^<think>.*?</think>\n<answer>.*?</answer>$", content, re.DOTALL)
        
        # Count occurrences of each tag
        think_open_count = content.count("<think>")
        think_close_count = content.count("</think>")
        answer_open_count = content.count("<answer>")
        answer_close_count = content.count("</answer>")
        
        # Check if exactly one of each tag exists
        tags_valid = (think_open_count == 1 and 
                    think_close_count == 1 and 
                    answer_open_count == 1 and 
                    answer_close_count == 1)
        no_text = "</text>" not in content    
        # Reward is 1.0 only if both structure and tag counts are correct
        
        reward = 0.5 if (structure_match and tags_valid and no_text) else 0.0
        # reward = 0.5 if no_text else 0.0
        
        
        return reward
 
    @staticmethod
    def format_reward(completions, **kwargs):
        """Reward function that checks if the completion has a specific format with exactly one of each tag."""
        return [SVGImageReward.single_format_reward(completion[0]["content"]) for completion in completions]
        

    @staticmethod
    def accuracy_reward(completions, solution, **kwargs):
        
        
        
        completion_contents = [completion[0]["content"] for completion in completions]
        ans = [SVGImageReward.extract_svg(content) for content in completion_contents]
        ans_ref = [SVGImageReward.extract_svg(content) for content in solution]
        rewards = []
        images = [safe_svg_to_image(content) for content in ans]
        ref_images = [safe_svg_to_image(content) for content in ans_ref]
        
        distances = vgg_image_image_distances_batch(ref_images, images)
        rewards = [1.0 - distance for distance in distances]
        return rewards
        

class SVGRawImageReward:
    @staticmethod
    def extract_answer(text):
        """Extract content between <answer> tags."""
        if text is None:
            return ""
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return SVGReward.extract_answer_half(text)
    @staticmethod
    def extract_answer_half(text):
        """Extract content between <answer> tags."""
        if text is None:
            return ""
        match = re.search(r'<answer>(.*?)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def extract_svg(text):
        if text is None:
            return ""
        # ans = SVGReward.extract_answer(text)
        ans = text
        match = re.search(r'(<svg .*?</svg>)', ans, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            match = re.search(r'(<svg.*?)', ans, re.DOTALL)
            if match:
                return match.group(1).strip() 
            
            return ""
    @staticmethod
    def single_format_reward(content, **kwargs):
        """Reward function that checks if the completion has a specific format with exactly one of each tag."""
        # content = completion[0]["content"]
        
        
        # Check if the overall structure is correct
        structure_match = re.match(r"^<think>.*?</think>\n<answer>.*?</answer>$", content, re.DOTALL)
        
        # Count occurrences of each tag
        think_open_count = content.count("<think>")
        think_close_count = content.count("</think>")
        answer_open_count = content.count("<answer>")
        answer_close_count = content.count("</answer>")
        
        # Check if exactly one of each tag exists
        tags_valid = (think_open_count == 1 and 
                    think_close_count == 1 and 
                    answer_open_count == 1 and 
                    answer_close_count == 1)
        no_text = "</text>" not in content    
        # Reward is 1.0 only if both structure and tag counts are correct
        
        reward = 0.5 if (structure_match and tags_valid and no_text) else 0.0
        # reward = 0.5 if no_text else 0.0
        
        
        return reward
 
    @staticmethod
    def format_reward(completions, **kwargs):
        """Reward function that checks if the completion has a specific format with exactly one of each tag."""
        return [SVGReward.single_format_reward(completion[0]["content"]) for completion in completions]
        

    @staticmethod
    def accuracy_reward(completions, solution, **kwargs):
        
        # debug_print("start accuracy_reward")
        
        completion_contents = [completion[0]["content"] for completion in completions]
        ans = [SVGReward.extract_svg(content) for content in completion_contents]
        rewards = []
        # debug_print(ans)
        images = [safe_svg_to_image(content) for content in ans]
        # debug_print("start accuracy_reward distances")
        distances = clip_text_image_distances_batch(solution, images)
        # debug_print("finished accuracy_reward distances")
        rewards = [1.0 - distance for distance in distances]
        return rewards
    @staticmethod
    def perceptual_reward(completions, image, **kwargs):
        
        
        
        completion_contents = [completion[0]["content"] for completion in completions]
        ans = [SVGImageReward.extract_svg(content) for content in completion_contents]
        
        rewards = []
        images = [safe_svg_to_image(content) for content in ans]
        ref_images = image
        
        distances = dinov2_image_image_distances_batch(ref_images, images)
        rewards = [(1.0 - distance)*10 for distance in distances]
        return rewards
    @staticmethod
    def is_grayscale(image, threshold=10):
        """
        Check if an image is grayscale (black and white).
        
        Args:
            image: PIL Image object
            threshold: Maximum allowed difference between RGB channels to still be considered grayscale
            
        Returns:
            float: 1.0 if grayscale, 0.0 if colored
        """
        if image is None:
            return 0.0
            
        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Get image data as numpy array
        import numpy as np
        img_array = np.array(image)
        
        # Check if RGB channels are approximately equal
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        
        # Calculate maximum difference between channels for each pixel
        diff_rg = np.abs(r.astype(int) - g.astype(int))
        diff_rb = np.abs(r.astype(int) - b.astype(int))
        diff_gb = np.abs(g.astype(int) - b.astype(int))
        
        max_diff = np.maximum(np.maximum(diff_rg, diff_rb), diff_gb)
        
        # If more than 95% of pixels have channel differences below threshold,
        # consider it grayscale
        grayscale_percentage = (max_diff <= threshold).mean()
        
        return 1.0 if grayscale_percentage > 0.95 else 0.0

    @staticmethod
    def color_reward(completions, **kwargs):
        """
        Reward function that penalizes grayscale images.
        Returns 0.0 for grayscale images, 1.0 for colored images.
        """
        completion_contents = [completion[0]["content"] for completion in completions]
        svg_contents = [SVGRawImageReward.extract_svg(content) for content in completion_contents]
        images = [safe_svg_to_image(content) for content in svg_contents]
        
        # Check if images are grayscale and penalize them
        color_scores = [0.5 if SVGRawImageReward.is_grayscale(img) else 0 for img in images]
        return color_scores
        


if __name__ ==  "__main__":
    text = """<svg width="200" height="300" xmlns="http://www.w3.org/2000/svg">
  <path id="table" d="M100 210 Q130 140 170 190 Q210 240 200 300 Q190 250 160 190 Q130 150 100 140 Q70 190 70 300 Q70 260 100 300 Q130 250 160 190 Q190 150 200 100 Q210 70 200 30" fill="gray" stroke="black" stroke-width="2" />
  
  <path id="chair" d="M120 280 Q120 275 125 275 Q130 275 130 280 Q130 290 125 290 Q123 290 120 280 Z" fill="gray" stroke="black" stroke-width="2" />
  
  <path id="girl" d="M150 180 Q160 160 180 180 Q200 200 180 220 Q160 240 150 220 M150 180 Q160 160 180 180 Q200 200 180 220 Q160 240 150 220" fill="lightblue" stroke="black" stroke-width="2" />

  <polygon id="hot-dog" points="160,240 160,220 160,155 180,155 180,120" fill="orange" />
  
  <text x="140" y="150" font-family="Comic Sans MS" font-size="36" text-anchor="middle">A</text>
  <text x="160" y="145" font-family="Comic Sans MS" font-size="36" text-anchor="middle">n</text>
  <text x="180" y="140" font-family="Comic Sans MS" font-size="36" text-anchor="middle">y</text>
  <text x="200" y="135" font-family="Comic Sans MS" font-size="36" text-anchor="middle">l</text>
  <text x="220" y="130" font-family="Comic Sans MS" font-size="36" text-anchor="middle">l</text>
  <text x="240" y="125" font-family="Comic Sans MS" font-size="36" text-anchor="middle">i</text>
  <text x="260" y="120" font-family="Comic Sans MS" font-size="36" text-anchor="middle">t</text>
  <text x="280" y="115" font-family="Comic Sans MS" font-size="36" text-anchor="middle">h</text>
  <text x="300" y="110" font-family="Comic Sans MS" font-size="36" text-anchor="middle">o</text>
  <text x="320" y="105" font-family="Comic Sans MS" font-size="36" text-anchor="middle">g</text>
  <text x="340" y="100" font-family="Comic Sans MS" font-size="36" text-anchor="middle"""
    print(SVGReward.extract_svg(text))
   