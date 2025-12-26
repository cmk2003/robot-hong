"""
多 Agent Prompts 模块
各 Agent 的系统提示词定义
"""

from .emotion_prompt import EMOTION_ANALYSIS_PROMPT
from .memory_prompt import MEMORY_RETRIEVAL_PROMPT
from .response_prompt import RESPONSE_GENERATION_PROMPT
from .save_prompt import MEMORY_SAVE_PROMPT
from .review_prompt import QUALITY_REVIEW_PROMPT

__all__ = [
    "EMOTION_ANALYSIS_PROMPT",
    "MEMORY_RETRIEVAL_PROMPT",
    "RESPONSE_GENERATION_PROMPT",
    "MEMORY_SAVE_PROMPT",
    "QUALITY_REVIEW_PROMPT",
]

