"""
回复生成 Agent
基于上下文生成符合人设的回复，支持工具调用和流式输出
"""

from typing import Dict, Any, List, Generator

from .base import BaseAgent
from ..prompts.response_prompt import RESPONSE_GENERATION_PROMPT, build_response_prompt
from ..tools import STATELESS_TOOLS


class ResponseAgent(BaseAgent):
    """
    回复生成 Agent
    
    职责：基于情感分析和记忆上下文，生成符合"小虹"人设的回复
    支持工具调用（时间、天气等）和流式输出
    """
    
    def __init__(self, llm_client):
        super().__init__(llm_client, "response")
        self.tools = STATELESS_TOOLS
    
    def run(
        self,
        user_message: str,
        emotion_result: Dict[str, Any] = None,
        memory_context: str = None,
        user_profile: Dict[str, Any] = None,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        生成回复（非流式）
        
        Args:
            user_message: 用户消息
            emotion_result: 情感分析结果
            memory_context: 检索到的记忆上下文
            user_profile: 用户画像
            chat_history: 对话历史
        
        Returns:
            {
                "content": str,
                "tool_calls": list
            }
        """
        self.logger.info(f"[ResponseAgent] 生成回复: {user_message[:50]}...")
        
        # 构建完整的系统提示词
        system_prompt = build_response_prompt(
            emotion_result=emotion_result,
            memory_context=memory_context,
            user_profile=user_profile
        )
        
        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]
        
        # 添加对话历史
        if chat_history:
            messages.extend(chat_history[-10:])  # 最多10轮历史
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})
        
        # 构建工具 schema
        tools_schema = self._get_tools_schema()
        
        # 调用 LLM（可能有工具调用）
        response = self.llm.chat(
            messages=messages,
            tools=tools_schema,
            temperature=0.7,
            max_tokens=500
        )
        
        # 处理工具调用
        tool_calls = response.get("tool_calls")
        content = response.get("content", "")
        
        if tool_calls:
            # 执行工具并重新生成
            content = self._handle_tool_calls(messages, tool_calls, tools_schema)
        
        self.logger.info(f"[ResponseAgent] 回复: {content[:50]}...")
        
        return {
            "content": content,
            "tool_calls": tool_calls
        }
    
    def run_stream(
        self,
        user_message: str,
        emotion_result: Dict[str, Any] = None,
        memory_context: str = None,
        user_profile: Dict[str, Any] = None,
        chat_history: List[Dict[str, str]] = None
    ) -> Generator[str, None, None]:
        """
        流式生成回复
        
        Args:
            user_message: 用户消息
            emotion_result: 情感分析结果
            memory_context: 检索到的记忆上下文
            user_profile: 用户画像
            chat_history: 对话历史
        
        Yields:
            回复内容片段
        """
        self.logger.info(f"[ResponseAgent] 流式生成回复: {user_message[:50]}...")
        
        # 构建完整的系统提示词
        system_prompt = build_response_prompt(
            emotion_result=emotion_result,
            memory_context=memory_context,
            user_profile=user_profile
        )
        
        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]
        
        if chat_history:
            messages.extend(chat_history[-10:])
        
        messages.append({"role": "user", "content": user_message})
        
        # 构建工具 schema
        tools_schema = self._get_tools_schema()
        
        # 先检查是否需要工具调用（非流式）
        check_response = self.llm.chat(
            messages=messages,
            tools=tools_schema,
            temperature=0.7,
            max_tokens=500
        )
        
        if check_response.get("tool_calls"):
            # 有工具调用，先处理工具，再流式输出
            self._handle_tool_calls_for_stream(
                messages, check_response["tool_calls"]
            )
        
        # 流式输出最终回复
        for chunk in self.llm.chat_stream(
            messages=messages,
            temperature=0.7,
            max_tokens=500
        ):
            yield chunk
    
    def rewrite(
        self,
        original_response: str,
        feedback: str,
        messages: List[Dict[str, str]]
    ) -> str:
        """
        根据反馈重写回复
        
        Args:
            original_response: 原始回复
            feedback: 质量评审反馈
            messages: 原始消息列表
        
        Returns:
            重写后的回复
        """
        self.logger.info(f"[ResponseAgent] 重写回复，反馈: {feedback[:50]}...")
        
        # 添加重写指令
        rewrite_messages = messages.copy()
        rewrite_messages.append({
            "role": "assistant",
            "content": original_response
        })
        rewrite_messages.append({
            "role": "user",
            "content": f"""你的回复需要修改。问题：{feedback}

请根据反馈重新回复，注意：
- 保持小虹的人设
- 修正指出的问题
- 不要解释，直接给出修改后的回复"""
        })
        
        response = self.llm.chat(
            messages=rewrite_messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.get("content", original_response)
    
    def _get_tools_schema(self) -> List[Dict[str, Any]]:
        """获取工具的 OpenAI schema"""
        schemas = []
        for tool in self.tools:
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {"type": "object", "properties": {}}
                }
            }
            if hasattr(tool, 'args') and tool.args:
                schema["function"]["parameters"] = {
                    "type": "object",
                    "properties": tool.args
                }
            schemas.append(schema)
        return schemas
    
    def _handle_tool_calls(
        self,
        messages: List[Dict[str, str]],
        tool_calls: List[Dict],
        tools_schema: List[Dict]
    ) -> str:
        """处理工具调用并获取最终回复"""
        # 添加助手消息（包含工具调用）
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": str(tc["function"]["arguments"])
                    }
                }
                for tc in tool_calls
            ]
        })
        
        # 执行工具
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            result = self._execute_tool(tool_name, tc["function"]["arguments"])
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(result)
            })
        
        # 重新调用 LLM 生成最终回复
        response = self.llm.chat(
            messages=messages,
            tools=tools_schema,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.get("content", "")
    
    def _handle_tool_calls_for_stream(
        self,
        messages: List[Dict[str, str]],
        tool_calls: List[Dict]
    ):
        """处理工具调用（为流式输出准备）"""
        # 添加助手消息
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": str(tc["function"]["arguments"])
                    }
                }
                for tc in tool_calls
            ]
        })
        
        # 执行工具
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            result = self._execute_tool(tool_name, tc["function"]["arguments"])
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(result)
            })
    
    def _execute_tool(self, tool_name: str, tool_args: Any) -> str:
        """执行工具"""
        import json
        
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    if isinstance(tool_args, str):
                        tool_args = json.loads(tool_args) if tool_args else {}
                    result = tool.invoke(tool_args)
                    self.logger.info(f"[ResponseAgent] 工具 {tool_name} 结果: {result}")
                    return str(result)
                except Exception as e:
                    self.logger.error(f"[ResponseAgent] 工具 {tool_name} 失败: {e}")
                    return f"工具执行失败: {e}"
        
        return f"未找到工具: {tool_name}"

