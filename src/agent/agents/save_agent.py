"""
记忆保存 Agent
决定需要保存什么信息到长期记忆
"""

from typing import Dict, Any, List

from .base import BaseAgent
from ..prompts.save_prompt import MEMORY_SAVE_PROMPT


class MemorySaveAgent(BaseAgent):
    """
    记忆保存 Agent
    
    职责：分析对话内容，决定需要保存什么到长期记忆
    """
    
    def __init__(self, llm_client, memory_manager):
        super().__init__(llm_client, "memory_save")
        self.memory = memory_manager
    
    def run(
        self,
        user_message: str,
        ai_response: str
    ) -> Dict[str, Any]:
        """
        分析对话并保存必要信息
        
        Args:
            user_message: 用户消息
            ai_response: AI 回复
        
        Returns:
            {
                "save_actions": list,
                "saved_count": int,
                "reasoning": str
            }
        """
        self.logger.info("[MemorySaveAgent] 分析需要保存的信息...")
        
        # 构建分析内容
        conversation = f"""用户: {user_message}

小虹: {ai_response}"""
        
        messages = self._build_messages(
            system_prompt=MEMORY_SAVE_PROMPT,
            user_content=conversation
        )
        
        result = self._call_llm_json(messages, temperature=0.3)
        
        # 执行保存操作
        save_actions = result.get("save_actions", [])
        saved_count = 0
        
        for action in save_actions[:3]:  # 最多执行3个保存操作
            try:
                self._execute_save_action(action)
                saved_count += 1
            except Exception as e:
                self.logger.error(f"[MemorySaveAgent] 保存失败: {e}")
        
        self.logger.info(
            f"[MemorySaveAgent] 保存了 {saved_count} 条信息"
        )
        
        return {
            "save_actions": save_actions,
            "saved_count": saved_count,
            "reasoning": result.get("reasoning", "")
        }
    
    def _execute_save_action(self, action: Dict[str, Any]):
        """
        执行单个保存操作
        
        Args:
            action: 保存操作定义
        """
        action_type = action.get("type")
        
        if action_type == "user_profile":
            field = action.get("field")
            value = action.get("value")
            if field and value:
                self.memory.update_user_profile(field=field, value=value)
                self.logger.info(f"[MemorySaveAgent] 更新用户画像: {field}={value}")
        
        elif action_type == "life_event":
            self.memory.save_life_event(
                event_type=action.get("event_type", "life"),
                title=action.get("title", ""),
                description=action.get("description"),
                importance=action.get("importance", 3)
            )
            self.logger.info(f"[MemorySaveAgent] 保存事件: {action.get('title')}")
        
        elif action_type == "emotion":
            self.memory.save_emotion(
                emotion_type=action.get("emotion_type", ""),
                intensity=action.get("intensity", 0.5),
                trigger=action.get("trigger")
            )
            self.logger.info(f"[MemorySaveAgent] 保存情感: {action.get('emotion_type')}")
        
        elif action_type == "follow_up":
            item = action.get("item")
            if item:
                self.memory.working_context.add_follow_up(item)
                self.logger.info(f"[MemorySaveAgent] 添加跟进: {item}")

