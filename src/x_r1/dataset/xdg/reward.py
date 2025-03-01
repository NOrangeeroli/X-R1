import re
from typing import Dict
import os
from openai import OpenAI
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

class XDGReward:
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
            return 1.0 if XDGReward.normalize_text(answer) == XDGReward.normalize_text(solution) else 0.0
    
    
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
            if len(gold_parsed) != 0:
                # print('latex gold parsed')
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_text = XDGReward.extract_answer(content)
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
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
                if reward ==0:
                    answer_parsed = parse(f"${XDGReward.normalize_text(answer_text)}$")
                    reward = float(verify(answer_parsed, gold_parsed))
                    
                # print('\nprompt:', prompt)
                print('-'*100)
                print(f"\nanswer text: {answer_text}\n")
                print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward, '\n')
            else:
                # For medical text answers, extract from <answer> tags and use GPT4O-mini for evaluation
                # answer_content = XDGReward.extract_answer(content)
                # normalized_content = XDGReward.normalize_text(answer_content)
                # normalized_solution = XDGReward.normalize_text(sol)
                # reward = XDGReward.evaluate_answer_similarity(normalized_content, normalized_solution)
                assert False
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

