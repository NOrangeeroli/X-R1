import re
from typing import Dict
import os
from openai import OpenAI
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from ..reward import Reward

class XDGReward(Reward):
    """Reward functions for the XDG dataset"""
    

    


