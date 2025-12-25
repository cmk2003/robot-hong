"""
向量嵌入模块
使用千问的 text-embedding API 实现语义相似度匹配
"""

import os
from typing import List, Optional
from openai import OpenAI

from ..utils.logger import get_logger

logger = get_logger("embedding")


class EmbeddingService:
    """向量嵌入服务"""
    
    def __init__(self):
        """初始化嵌入服务"""
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model = "text-embedding-v3"
        
        if self.api_key:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            self._client = None
            logger.warning("未配置 DASHSCOPE_API_KEY，向量搜索功能不可用")
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        获取文本的向量嵌入
        
        Args:
            text: 输入文本
        
        Returns:
            向量列表，失败返回 None
        """
        if not self._client or not text.strip():
            return None
        
        try:
            response = self._client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=512  # 使用较小维度，节省存储和计算
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"获取 embedding 失败: {e}")
            return None
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
        
        Returns:
            相似度 (0-1)
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar(
        self, 
        query: str, 
        candidates: List[str], 
        threshold: float = 0.5,
        top_k: int = 3
    ) -> List[tuple]:
        """
        找出与查询最相似的候选项
        
        Args:
            query: 查询文本
            candidates: 候选文本列表
            threshold: 相似度阈值
            top_k: 返回前 k 个结果
        
        Returns:
            [(候选文本, 相似度), ...] 按相似度降序
        """
        query_vec = self.get_embedding(query)
        if not query_vec:
            return []
        
        results = []
        for candidate in candidates:
            candidate_vec = self.get_embedding(candidate)
            if candidate_vec:
                similarity = self.cosine_similarity(query_vec, candidate_vec)
                if similarity >= threshold:
                    results.append((candidate, similarity))
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# 全局单例
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """获取嵌入服务单例"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

