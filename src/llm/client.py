"""
LLM客户端模块
支持千问和DeepSeek，兼容OpenAI接口
"""

import json
from typing import List, Dict, Any, Optional
from openai import OpenAI

from ..config import LLMProviderConfig


class LLMClient:
    """LLM客户端 - 封装大语言模型调用"""
    
    def __init__(self, config: LLMProviderConfig):
        """
        初始化LLM客户端
        
        Args:
            config: LLM提供商配置
        """
        self.config = config
        self.base_url = config.base_url
        self.model = config.model
        self.api_key = config.api_key
        
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def build_messages(
        self,
        system_prompt: str,
        user_message: str,
        history: List[Dict[str, str]] = None,
        working_context: str = None
    ) -> List[Dict[str, str]]:
        """
        构建消息列表
        
        Args:
            system_prompt: 系统提示词
            user_message: 用户当前消息
            history: 历史对话列表
            working_context: 工作上下文（会追加到system_prompt后）
        
        Returns:
            格式化的消息列表
        """
        messages = []
        
        # 系统提示词（包含工作上下文）
        full_system = system_prompt
        if working_context:
            full_system += f"\n\n## 当前上下文\n{working_context}"
        
        messages.append({
            "role": "system",
            "content": full_system
        })
        
        # 添加历史对话
        if history:
            messages.extend(history)
        
        # 添加当前用户消息
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            tools: 工具/函数定义列表
            temperature: 温度参数
            max_tokens: 最大token数
        
        Returns:
            响应内容，包含 content 和 tool_calls
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        response = self._client.chat.completions.create(**kwargs)
        
        message = response.choices[0].message
        
        result = {
            "content": message.content,
            "tool_calls": None
        }
        
        # 处理工具调用
        if message.tool_calls:
            result["tool_calls"] = []
            for tool_call in message.tool_calls:
                result["tool_calls"].append({
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments)
                    }
                })
        
        return result
    
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        流式聊天请求
        
        Args:
            messages: 消息列表
            tools: 工具/函数定义列表
            temperature: 温度参数
            max_tokens: 最大token数
        
        Yields:
            响应内容片段
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        response = self._client.chat.completions.create(**kwargs)
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ============ LangChain ChatModel 工厂 ============

def create_langchain_chat_model(config: LLMProviderConfig):
    """
    创建 LangChain ChatModel 实例
    
    Args:
        config: LLM 提供商配置
    
    Returns:
        ChatOpenAI 实例
    """
    from langchain_openai import ChatOpenAI
    
    return ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=0.7,
        max_tokens=2000
    )

