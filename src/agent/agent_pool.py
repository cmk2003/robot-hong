"""
Agent 实例池
管理多用户的 EmotionalAgent 实例，实现用户数据隔离
"""

import hashlib
import threading
from typing import Dict, Optional

from .emotional_agent import EmotionalAgent
from ..llm.client import LLMClient
from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("agent_pool")


def username_to_user_id(username: str) -> str:
    """
    将用户名转换为 user_id
    使用 MD5 哈希确保唯一性和一致性
    
    Args:
        username: 用户名
    
    Returns:
        16 字符的 user_id
    """
    normalized = username.strip().lower()
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]


class AgentPool:
    """
    Agent 实例池
    
    - 按 user_id 管理 EmotionalAgent 实例
    - 线程安全
    - 懒加载：首次访问时创建
    """
    
    def __init__(self, config: Config):
        """
        初始化 Agent 池
        
        Args:
            config: 应用配置
        """
        self._config = config
        self._agents: Dict[str, EmotionalAgent] = {}
        self._lock = threading.Lock()
        
        # 预创建 LLM 客户端（所有 Agent 共享）
        self._llm_config = config.get_llm_config()
        self._llm_client = LLMClient(self._llm_config)
        
        # 多 Agent 模式的 LLM 客户端
        self._agent_llm_clients: Dict[str, LLMClient] = {}
        if config.agent_mode == "multi":
            agent_names = ["emotion", "memory", "response", "save", "review"]
            for agent_name in agent_names:
                agent_config = config.get_agent_llm_config(agent_name)
                if agent_config.model != self._llm_config.model:
                    self._agent_llm_clients[agent_name] = LLMClient(agent_config)
                else:
                    self._agent_llm_clients[agent_name] = self._llm_client
        
        logger.info(f"[AgentPool] 初始化完成，模式: {config.agent_mode}")
    
    def get_agent(self, user_id: str) -> EmotionalAgent:
        """
        获取指定用户的 Agent 实例
        如果不存在则创建
        
        Args:
            user_id: 用户 ID
        
        Returns:
            EmotionalAgent 实例
        """
        with self._lock:
            if user_id not in self._agents:
                self._agents[user_id] = self._create_agent(user_id)
                logger.info(f"[AgentPool] 创建新 Agent: user_id={user_id}")
            return self._agents[user_id]
    
    def get_agent_by_username(self, username: str) -> EmotionalAgent:
        """
        通过用户名获取 Agent 实例
        
        Args:
            username: 用户名
        
        Returns:
            EmotionalAgent 实例
        """
        user_id = username_to_user_id(username)
        return self.get_agent(user_id)
    
    def _create_agent(self, user_id: str) -> EmotionalAgent:
        """
        创建新的 Agent 实例
        
        Args:
            user_id: 用户 ID
        
        Returns:
            初始化完成的 EmotionalAgent
        """
        agent = EmotionalAgent(
            db_path=self._config.database_path,
            user_id=user_id,
            llm_client=self._llm_client,
            mode=self._config.agent_mode,
            agent_llm_clients=self._agent_llm_clients if self._config.agent_mode == "multi" else None
        )
        agent.init()
        return agent
    
    def remove_agent(self, user_id: str) -> None:
        """
        移除指定用户的 Agent 实例
        
        Args:
            user_id: 用户 ID
        """
        with self._lock:
            if user_id in self._agents:
                try:
                    self._agents[user_id].close()
                except Exception as e:
                    logger.warning(f"[AgentPool] 关闭 Agent 失败: {e}")
                del self._agents[user_id]
                logger.info(f"[AgentPool] 移除 Agent: user_id={user_id}")
    
    def get_active_user_count(self) -> int:
        """
        获取当前活跃用户数
        
        Returns:
            活跃 Agent 实例数量
        """
        with self._lock:
            return len(self._agents)
    
    def close_all(self) -> None:
        """关闭所有 Agent 实例"""
        with self._lock:
            for user_id, agent in self._agents.items():
                try:
                    agent.close()
                    logger.info(f"[AgentPool] 关闭 Agent: user_id={user_id}")
                except Exception as e:
                    logger.warning(f"[AgentPool] 关闭 Agent 失败: user_id={user_id}, error={e}")
            self._agents.clear()
            logger.info("[AgentPool] 所有 Agent 已关闭")

