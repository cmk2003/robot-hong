"""
记忆检索 Agent
智能决定是否需要检索历史记忆，并执行检索
支持向量语义搜索
"""

from typing import Dict, Any, List

from .base import BaseAgent
from ..prompts.memory_prompt import MEMORY_RETRIEVAL_PROMPT
from ...utils.embedding import get_embedding_service


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
        执行记忆检索（使用向量语义搜索 + 关键词搜索）
        
        Args:
            queries: 检索关键词列表
            search_types: 检索类型列表
        
        Returns:
            格式化的检索结果文本
        """
        results = []
        
        # 确保参数有效
        if not queries:
            return ""
        if not search_types:
            search_types = ["messages"]
        
        # 获取向量服务
        embedding_service = get_embedding_service()
        
        # 合并查询词用于向量搜索
        combined_query = " ".join(queries[:3])
        
        try:
            # 1. 向量搜索事件（优先使用语义搜索）
            if "events" in search_types:
                events = self.memory.get_life_events(limit=10)
                if events and embedding_service:
                    query_vec = embedding_service.get_embedding(combined_query)
                    if query_vec:
                        import json
                        for event in events:
                            title = event.get("title") or ""
                            embedding_str = event.get("embedding")
                            
                            if embedding_str and title:
                                try:
                                    event_vec = json.loads(embedding_str)
                                    similarity = embedding_service.cosine_similarity(query_vec, event_vec)
                                    if similarity >= 0.4:  # 相似度阈值
                                        desc = event.get('description', '')[:50] if event.get('description') else ''
                                        results.append(
                                            f"[生活事件] {title} - {desc} (相似度: {similarity:.2f})"
                                        )
                                except (json.JSONDecodeError, TypeError):
                                    pass
                    
                    # 如果向量搜索没结果，回退到关键词匹配
                    if not any("[生活事件]" in r for r in results):
                        for event in events:
                            title = event.get("title", "").lower()
                            for query in queries:
                                if query.lower() in title:
                                    results.append(
                                        f"[生活事件] {event.get('title', '')} - {event.get('description', '')[:50]}"
                                    )
                                    break
            
            # 2. 向量搜索消息
            if "messages" in search_types:
                # 先尝试关键词搜索
                for query in queries[:3]:
                    if not query:
                        continue
                    messages = self.memory.search_messages(query, limit=3)
                    if messages:
                        for msg in messages:
                            if msg:
                                results.append(
                                    f"[历史对话] {msg.get('role', 'user')}: {msg.get('content', '')[:100]}"
                                )
                
                # 如果关键词搜索没结果，尝试向量搜索最近消息
                if not any("[历史对话]" in r for r in results) and embedding_service:
                    recent_messages = self.memory.get_recent_messages(limit=20)
                    if recent_messages:
                        query_vec = embedding_service.get_embedding(combined_query)
                        if query_vec:
                            candidates = []
                            for msg in recent_messages:
                                content = msg.get("content", "")
                                if content:
                                    msg_vec = embedding_service.get_embedding(content[:200])
                                    if msg_vec:
                                        similarity = embedding_service.cosine_similarity(query_vec, msg_vec)
                                        if similarity >= 0.45:
                                            candidates.append((msg, similarity))
                            
                            # 取相似度最高的3条
                            candidates.sort(key=lambda x: x[1], reverse=True)
                            for msg, sim in candidates[:3]:
                                results.append(
                                    f"[历史对话] {msg.get('role', 'user')}: {msg.get('content', '')[:100]}"
                                )
            
            # 3. 检索情感记录
            if "emotions" in search_types:
                emotions = self.memory.get_emotion_history(limit=5)
                if emotions:
                    for emotion in emotions:
                        if emotion:
                            for query in queries:
                                if query.lower() in emotion.get("emotion_type", "").lower():
                                    results.append(
                                        f"[情感记录] {emotion.get('emotion_type', '')} "
                                        f"(强度: {emotion.get('intensity', 0)})"
                                    )
                                    break
                                    
        except Exception as e:
            self.logger.warning(f"[MemoryAgent] 检索失败: {e}")
        
        # 去重并格式化
        unique_results = list(dict.fromkeys(results))
        return "\n".join(unique_results[:10])  # 最多10条结果

