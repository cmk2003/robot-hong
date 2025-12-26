"""
Agent 基类
所有 Agent 的公共基础设施
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from ...utils.logger import get_logger


class BaseAgent(ABC):
    """
    Agent 基类
    提供 LLM 调用、JSON 解析等公共方法
    """
    
    def __init__(self, llm_client, name: str):
        """
        初始化 Agent
        
        Args:
            llm_client: LLM 客户端实例
            name: Agent 名称，用于日志
        """
        self.llm = llm_client
        self.name = name
        self.logger = get_logger(f"agent.{name}")
    
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        执行 Agent 逻辑
        
        子类必须实现此方法
        """
        raise NotImplementedError
    
    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        调用 LLM 获取响应
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
        
        Returns:
            LLM 响应文本
        """
        self.logger.info(f"[{self.name}] 调用 LLM...")
        
        response = self.llm.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        content = response.get("content", "")
        self.logger.info(f"[{self.name}] LLM 响应: {content[:100]}...")
        
        return content
    
    def _call_llm_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        调用 LLM 并解析 JSON 响应
        
        Args:
            messages: 消息列表
            temperature: 温度参数（JSON 响应建议低温度）
            max_tokens: 最大 token 数
            max_retries: 最大重试次数
        
        Returns:
            解析后的 JSON 字典
        """
        for attempt in range(max_retries + 1):
            content = self._call_llm(messages, temperature, max_tokens)
            
            try:
                result = self._parse_json(content)
                return result
            except json.JSONDecodeError as e:
                self.logger.warning(
                    f"[{self.name}] JSON 解析失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}"
                )
                if attempt < max_retries:
                    # 添加重试提示
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": "请只返回有效的 JSON 格式，不要包含其他文字。"
                    })
        
        # 全部失败，返回空字典
        self.logger.error(f"[{self.name}] JSON 解析最终失败，返回空字典")
        return {}
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """
        从文本中解析 JSON
        
        支持从 markdown 代码块中提取 JSON
        
        Args:
            text: 包含 JSON 的文本
        
        Returns:
            解析后的字典
        """
        # 尝试直接解析
        text = text.strip()
        
        # 尝试从 markdown 代码块提取
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            text = json_match.group(1).strip()
        
        # 尝试找到 JSON 对象
        brace_match = re.search(r'\{[\s\S]*\}', text)
        if brace_match:
            text = brace_match.group(0)
        
        return json.loads(text)
    
    def _build_messages(
        self,
        system_prompt: str,
        user_content: str,
        context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        构建消息列表
        
        Args:
            system_prompt: 系统提示词
            user_content: 用户内容
            context: 可选的上下文信息
        
        Returns:
            格式化的消息列表
        """
        full_system = system_prompt
        if context:
            full_system += f"\n\n## 上下文信息\n{context}"
        
        return [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_content}
        ]

