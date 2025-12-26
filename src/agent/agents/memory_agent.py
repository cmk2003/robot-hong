"""
记忆检索 Agent
智能决定是否需要检索历史记忆，并执行检索
"""

from typing import Dict, Any, List

from .base import BaseAgent
from ..prompts.memory_prompt import MEMORY_RETRIEVAL_PROMPT


class MemoryRetrievalAgent(BaseAgent):
    """
    记忆检索 Agent
    
    职责：决定是否需要检索历史记忆，生成检索查询，执行检索
    """
    
    def __init__(self, llm_client, memory_manager):
        super().__init__(llm_client, "memory_retrieval")
        self.memory = memory_manager
    
    def run(
        self,
        user_message: str,
        emotion_result: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        决定是否需要检索并执行检索
        
        Args:
            user_message: 用户消息
            emotion_result: 情感分析结果
        
        Returns:
            检索结果：
            {
                "should_search": bool,
                "retrieved_context": str,
                "search_queries": list,
                "reasoning": str
            }
        """
        self.logger.info(f"[MemoryAgent] 分析是否需要检索: {user_message[:50]}...")
        
        # 构建上下文
        context = ""
        if emotion_result:
            context = f"用户当前情绪: {emotion_result.get('emotion_type', '未知')}"
        
        messages = self._build_messages(
            system_prompt=MEMORY_RETRIEVAL_PROMPT,
            user_content=user_message,
            context=context
        )
        
        decision = self._call_llm_json(messages, temperature=0.3)
        
        # 如果决定需要检索，执行检索
        retrieved_context = ""
        if decision.get("should_search", False):
            retrieved_context = self._execute_search(
                queries=decision.get("search_queries", []),
                search_types=decision.get("search_types", ["messages"])
            )
        
        result = {
            "should_search": decision.get("should_search", False),
            "retrieved_context": retrieved_context,
            "search_queries": decision.get("search_queries", []),
            "reasoning": decision.get("reasoning", "")
        }
        
        self.logger.info(
            f"[MemoryAgent] 需要检索: {result['should_search']}, "
            f"检索到: {len(retrieved_context)} 字符"
        )
        
        return result
    
    def _execute_search(
        self,
        queries: List[str],
        search_types: List[str]
    ) -> str:
        """
        执行记忆检索
        
        Args:
            queries: 检索关键词列表
            search_types: 检索类型列表
        
        Returns:
            格式化的检索结果文本
        """
        results = []
        
        for query in queries[:3]:  # 最多3个查询
            # 检索消息
            if "messages" in search_types:
                messages = self.memory.search_messages(query, limit=3)
                for msg in messages:
                    results.append(
                        f"[历史对话] {msg.get('role', 'user')}: {msg.get('content', '')[:100]}"
                    )
            
            # 检索事件
            if "events" in search_types:
                events = self.memory.get_life_events(limit=5)
                for event in events:
                    if query.lower() in event.get("title", "").lower():
                        results.append(
                            f"[生活事件] {event.get('title', '')} - {event.get('description', '')[:50]}"
                        )
            
            # 检索情感记录
            if "emotions" in search_types:
                emotions = self.memory.get_emotion_history(limit=5)
                for emotion in emotions:
                    if query.lower() in emotion.get("emotion_type", "").lower():
                        results.append(
                            f"[情感记录] {emotion.get('emotion_type', '')} "
                            f"(强度: {emotion.get('intensity', 0)})"
                        )
        
        # 去重并格式化
        unique_results = list(dict.fromkeys(results))
        return "\n".join(unique_results[:10])  # 最多10条结果

