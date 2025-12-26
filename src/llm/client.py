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
        流式聊天请求（纯文本，不支持工具调用）
        
        Args:
            messages: 消息列表
            tools: 工具/函数定义列表（流式模式下忽略）
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
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        
        # 注意：流式模式不传递 tools，确保只输出文本
        response = self._client.chat.completions.create(**kwargs)
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def chat_stream_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        流式聊天请求，支持工具调用检测
        
        Args:
            messages: 消息列表
            tools: 工具/函数定义列表
            temperature: 温度参数
            max_tokens: 最大token数
        
        Yields:
            dict: {"type": "content", "data": str} 或 {"type": "tool_call", "data": dict}
        
        最终会yield: {"type": "done", "has_tool_calls": bool, "tool_calls": list}
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        response = self._client.chat.completions.create(**kwargs)
        
        # 收集工具调用
        tool_calls_buffer = {}  # id -> {name, arguments}
        has_tool_calls = False
        
        for chunk in response:
            if not chunk.choices:
                continue
                
            delta = chunk.choices[0].delta
            
            # 处理文本内容
            if delta.content:
                yield {"type": "content", "data": delta.content}
            
            # 处理工具调用（流式拼接）
            if delta.tool_calls:
                has_tool_calls = True
                for tc in delta.tool_calls:
                    tc_id = tc.id or list(tool_calls_buffer.keys())[-1] if tool_calls_buffer else "0"
                    
                    if tc.id:  # 新的工具调用
                        tool_calls_buffer[tc.id] = {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name if tc.function else "",
                                "arguments": tc.function.arguments if tc.function else ""
                            }
                        }
                    elif tool_calls_buffer:
                        # 追加 arguments
                        last_id = list(tool_calls_buffer.keys())[-1]
                        if tc.function and tc.function.arguments:
                            tool_calls_buffer[last_id]["function"]["arguments"] += tc.function.arguments
        
        # 解析工具调用的 arguments
        tool_calls = []
        for tc_data in tool_calls_buffer.values():
            try:
                tc_data["function"]["arguments"] = json.loads(tc_data["function"]["arguments"])
            except json.JSONDecodeError:
                pass
            tool_calls.append(tc_data)
        
        yield {
            "type": "done",
            "has_tool_calls": has_tool_calls,
            "tool_calls": tool_calls if tool_calls else None
        }


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

