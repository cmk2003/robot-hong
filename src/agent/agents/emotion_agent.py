"""
情感分析 Agent
深度分析用户消息中的情感状态
"""

from typing import Dict, Any

from .base import BaseAgent
from ..prompts.emotion_prompt import EMOTION_ANALYSIS_PROMPT


class EmotionAgent(BaseAgent):
    """
    情感分析 Agent
    
    职责：分析用户消息的情绪类型、强度、触发因素和情感需求
    """
    
    def __init__(self, llm_client):
        super().__init__(llm_client, "emotion")
    
    def run(self, user_message: str, context: str = None) -> Dict[str, Any]:
        """
        分析用户消息的情感
        
        Args:
            user_message: 用户消息
            context: 可选的上下文信息（如之前的对话）
        
        Returns:
            情感分析结果：
            {
                "emotion_type": str,
                "intensity": float,
                "trigger": str,
                "needs": str,
                "reasoning": str
            }
        """
        self.logger.info(f"[EmotionAgent] 分析消息: {user_message[:50]}...")
        
        messages = self._build_messages(
            system_prompt=EMOTION_ANALYSIS_PROMPT,
            user_content=user_message,
            context=context
        )
        
        result = self._call_llm_json(messages, temperature=0.3)
        
        # 设置默认值
        if not result:
            result = self._get_default_result()
        else:
            result = self._validate_result(result)
        
        self.logger.info(
            f"[EmotionAgent] 结果: {result.get('emotion_type')} "
            f"(强度: {result.get('intensity')})"
        )
        
        return result
    
    def _get_default_result(self) -> Dict[str, Any]:
        """返回默认的情感分析结果"""
        return {
            "emotion_type": "平静",
            "intensity": 0.3,
            "trigger": None,
            "needs": "陪伴",
            "reasoning": "无法分析，使用默认值"
        }
    
    def _validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """验证并修正结果"""
        # 确保必要字段存在
        if "emotion_type" not in result:
            result["emotion_type"] = "平静"
        
        # 确保 intensity 在有效范围内
        intensity = result.get("intensity", 0.5)
        if not isinstance(intensity, (int, float)):
            intensity = 0.5
        result["intensity"] = max(0.0, min(1.0, float(intensity)))
        
        # 设置默认 needs
        if "needs" not in result:
            result["needs"] = "陪伴"
        
        return result

