import re
from typing import Dict
import os
from openai import OpenAI
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

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
                print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward, '\n')
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
    
    
from .utils.clips import svg_to_image, clip_text_image_distance
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
        """Extract content between <answer> tags."""
        if text is None:
            return ""
        ans = SVGReward.extract_answer(text)
        match = re.search(r'(<svg.*?</svg>)', ans, re.DOTALL)
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
            
        # Reward is 1.0 only if both structure and tag counts are correct
        reward = 1.0 if (structure_match and tags_valid) else 0.0
        
        
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
        for content, sol, text in zip(ans, solution, completion_contents):
            # print(f"CONTENT: {text}\n")
            # print(f"SVG CODE: {content}\n")    
            image = svg_to_image(content)
            if not image:
                reward=0.0
                
            # elif SVGReward.single_format_reward(content) == 0.0:
            #     reward=0.0
            else:
                distance = clip_text_image_distance(sol, image)
                reward=1.0 - distance
                # if reward <0.3:
                #     reward=0.0
            rewards.append(reward)
            # print(f"ACC: {reward}\n") 
                    
            

        return rewards

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
   