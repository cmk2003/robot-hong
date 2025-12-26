"""
质量评审 Agent
检查回复是否符合人设和质量要求
"""

from typing import Dict, Any

from .base import BaseAgent
from ..prompts.review_prompt import QUALITY_REVIEW_PROMPT


class ReviewAgent(BaseAgent):
    """
    质量评审 Agent
    
    职责：检查 AI 回复是否符合"小虹"人设和质量要求
    """
    
    def __init__(self, llm_client):
        super().__init__(llm_client, "review")
    
    def run(
        self,
        user_message: str,
        ai_response: str
    ) -> Dict[str, Any]:
        """
        评审 AI 回复
        
        Args:
            user_message: 用户消息
            ai_response: AI 回复
        
        Returns:
            {
                "approved": bool,
                "score": int,
                "issues": list,
                "suggestion": str,
                "reasoning": str
            }
        """
        self.logger.info(f"[ReviewAgent] 评审回复: {ai_response[:50]}...")
        
        # 构建评审内容
        review_content = f"""## 用户消息
{user_message}

## AI 回复
{ai_response}

请评审这个回复是否符合"小虹"的人设和质量要求。"""
        
        messages = self._build_messages(
            system_prompt=QUALITY_REVIEW_PROMPT,
            user_content=review_content
        )
        
        result = self._call_llm_json(messages, temperature=0.3)
        
        # 确保必要字段存在
        result = self._validate_result(result, ai_response)
        
        self.logger.info(
            f"[ReviewAgent] 评审结果: {'通过' if result['approved'] else '不通过'} "
            f"(分数: {result['score']})"
        )
        
        return result
    
    def _validate_result(
        self,
        result: Dict[str, Any],
        ai_response: str
    ) -> Dict[str, Any]:
        """验证并修正结果"""
        # 默认值
        if "approved" not in result:
            # 简单规则检查
            result["approved"] = self._quick_check(ai_response)
        
        if "score" not in result:
            result["score"] = 7 if result["approved"] else 5
        
        if "issues" not in result:
            result["issues"] = []
        
        if "suggestion" not in result:
            result["suggestion"] = ""
        
        if "reasoning" not in result:
            result["reasoning"] = ""
        
        # 确保 score 是数字
        try:
            result["score"] = int(result["score"])
        except (ValueError, TypeError):
            result["score"] = 5
        
        # 根据 score 调整 approved
        if result["score"] < 6:
            result["approved"] = False
        
        return result
    
    def _quick_check(self, response: str) -> bool:
        """
        快速规则检查
        
        Args:
            response: AI 回复
        
        Returns:
            是否通过基本检查
        """
        # 检查禁止项
        forbidden_patterns = [
            "1.", "2.", "3.",  # 列表
            "• ", "- ",  # 列表符号
            "首先", "其次", "最后",  # 结构化
            "比如：", "例如：",  # 冒号解释
        ]
        
        for pattern in forbidden_patterns:
            if pattern in response:
                return False
        
        # 检查长度
        if len(response) > 200:
            return False
        
        return True

