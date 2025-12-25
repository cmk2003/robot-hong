"""
情感Agent模块
核心Agent类，整合所有模块
"""

import json
import re
from typing import Dict, Any, List, Optional

from .memory import MemoryManager
from .context import WorkingContext
from ..emotion.analyzer import EmotionAnalyzer, EmotionResult
from ..llm.client import LLMClient
from ..llm.prompts import SYSTEM_PROMPT
from ..llm.functions import TOOL_DEFINITIONS
from ..tools.realtime import get_current_datetime, get_weather
from ..utils.logger import get_logger

# 初始化日志
logger = get_logger("emotional_agent")


class FunctionExecutor:
    """
    函数执行器
    执行LLM调用的工具函数
    """
    
    def __init__(self, memory: MemoryManager):
        """
        初始化执行器
        
        Args:
            memory: 记忆管理器
        """
        self.memory = memory
    
    def execute(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行函数
        
        Args:
            function_name: 函数名
            arguments: 函数参数
        
        Returns:
            执行结果
        """
        try:
            if function_name == "save_emotion":
                return self._save_emotion(arguments)
            elif function_name == "save_life_event":
                return self._save_life_event(arguments)
            elif function_name == "update_user_profile":
                return self._update_user_profile(arguments)
            elif function_name == "search_memory":
                return self._search_memory(arguments)
            elif function_name == "set_follow_up":
                return self._set_follow_up(arguments)
            # 实时信息工具
            elif function_name == "get_current_datetime":
                return get_current_datetime()
            elif function_name == "get_weather":
                city = arguments.get("city", "深圳")
                return get_weather(city)
            else:
                return {
                    "success": False,
                    "error": f"未知函数: {function_name}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _save_emotion(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """保存情感记录"""
        self.memory.save_emotion(
            emotion_type=args["emotion_type"],
            intensity=args["intensity"],
            trigger=args.get("trigger")
        )
        return {"success": True, "message": "情感记录已保存"}
    
    def _save_life_event(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """保存生活事件"""
        self.memory.save_life_event(
            event_type=args["event_type"],
            title=args["title"],
            description=args.get("description"),
            importance=args.get("importance", 3),
            emotion_impact=args.get("emotion_impact")
        )
        return {"success": True, "message": "生活事件已保存"}
    
    def _update_user_profile(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """更新用户画像"""
        logger.info(f"[用户画像更新] field={args['field']}, value={args['value']}")
        self.memory.update_user_profile(
            field=args["field"],
            value=args["value"]
        )
        logger.info(f"[用户画像更新] 保存成功!")
        return {"success": True, "message": "用户画像已更新"}
    
    def _search_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """搜索记忆"""
        query = args["query"]
        search_type = args.get("search_type", "messages")
        
        if search_type == "messages":
            results = self.memory.search_messages(query)
        elif search_type == "events":
            results = self.memory.get_life_events()
            results = [e for e in results if query.lower() in e["title"].lower()]
        elif search_type == "emotions":
            results = self.memory.get_emotion_history()
            results = [e for e in results if query.lower() in e.get("emotion_type", "").lower()]
        else:
            results = []
        
        return {
            "success": True,
            "results": results[:5]  # 限制返回数量
        }
    
    def _set_follow_up(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """设置待跟进事项"""
        self.memory.working_context.add_follow_up(args["item"])
        return {"success": True, "message": "待跟进事项已添加"}


class EmotionalAgent:
    """
    情感陪伴Agent
    基于MemGPT架构，具备长期记忆能力
    """
    
    def __init__(
        self,
        db_path: str,
        user_id: str,
        llm_client: LLMClient,
        system_prompt: str = None
    ):
        """
        初始化Agent
        
        Args:
            db_path: 数据库路径
            user_id: 用户ID
            llm_client: LLM客户端
            system_prompt: 系统提示词（可选，默认使用预设）
        """
        self.db_path = db_path
        self.user_id = user_id
        self.llm_client = llm_client
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        
        # 初始化模块
        self.memory = MemoryManager(db_path, user_id)
        self.emotion_analyzer = EmotionAnalyzer(llm_client)
        self.function_executor = FunctionExecutor(self.memory)
    
    def init(self) -> None:
        """初始化Agent"""
        self.memory.init()
    
    def close(self) -> None:
        """关闭Agent"""
        self.memory.close()
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        处理用户消息
        
        Args:
            user_message: 用户消息
        
        Returns:
            响应结果 {"content": str, "emotion": dict, ...}
        """
        logger.info("=" * 50)
        logger.info(f"[用户消息] {user_message}")
        logger.info(f"[当前用户画像] user_name={self.memory.working_context.user_name}, user_info={self.memory.working_context.user_info}")
        
        # 1. 分析用户情感
        emotion_result = self.emotion_analyzer.analyze(user_message)
        if emotion_result:
            self.memory.working_context.update_emotion(
                emotion_result.emotion_type,
                emotion_result.intensity
            )
        
        # 2. 智能搜索相关历史记忆（方案B核心）
        relevant_context = self.memory.search_relevant_context(user_message)
        
        # 3. 保存用户消息
        self.memory.save_message(
            role="user",
            content=user_message,
            emotion_type=emotion_result.emotion_type if emotion_result else None,
            emotion_intensity=emotion_result.intensity if emotion_result else None
        )
        
        # 4. 组合工作上下文和相关历史
        working_context = self.memory.get_context_for_llm()
        if relevant_context:
            working_context = working_context + "\n" + relevant_context if working_context else relevant_context
        
        # 5. 构建LLM消息
        messages = self.llm_client.build_messages(
            system_prompt=self.system_prompt,
            user_message=user_message,
            history=self.memory.get_messages_for_llm()[:-1],  # 不包括刚添加的用户消息
            working_context=working_context
        )
        
        # 6. 调用LLM
        response = self.llm_client.chat(
            messages=messages,
            tools=TOOL_DEFINITIONS
        )
        
        # 7. 处理工具调用
        assistant_content = response.get("content") or ""
        
        # 日志：记录 LLM 响应
        logger.info(f"[LLM响应] content: {assistant_content[:100] if assistant_content else 'None'}...")
        logger.info(f"[LLM响应] tool_calls: {response.get('tool_calls')}")
        
        # 如果有工具调用，清理content中可能混入的函数调用文本
        # 某些LLM（如千问、DeepSeek）会在content中也写入函数名
        if response.get("tool_calls"):
            # 收集所有被调用的函数名
            called_functions = [tc["function"]["name"] for tc in response["tool_calls"]]
            logger.info(f"[工具调用] 检测到工具调用: {called_functions}")
            # 从content中移除函数调用文本（如 "get_current_datetime()" ）
            for func_name in called_functions:
                # 匹配 函数名() 或 函数名(参数) 的模式
                pattern = rf'{func_name}\([^)]*\)'
                assistant_content = re.sub(pattern, '', assistant_content)
            # 清理多余的空白
            assistant_content = assistant_content.strip()
        else:
            logger.info("[工具调用] 没有检测到工具调用")
        
        if response.get("tool_calls"):
            # 收集函数执行结果
            tool_results = []
            # 标记是否有需要返回结果给用户的工具（如时间、天气）
            has_info_tools = False
            INFO_TOOLS = {"get_current_datetime", "get_weather"}  # 需要把结果告诉用户的工具
            
            for tool_call in response["tool_calls"]:
                func_name = tool_call["function"]["name"]
                func_args = tool_call["function"]["arguments"]
                
                logger.info(f"[工具执行] 执行函数: {func_name}, 参数: {func_args}")
                
                if func_name in INFO_TOOLS:
                    has_info_tools = True
                
                # 执行函数
                result = self.function_executor.execute(func_name, func_args)
                logger.info(f"[工具执行] 函数 {func_name} 执行结果: {result}")
                
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "name": func_name,
                    "result": result
                })
            
            # 如果没有文本回复，或者有需要返回信息的工具，需要再次调用 LLM 生成回复
            if not assistant_content or has_info_tools:
                # 构建包含函数结果的消息
                follow_up_messages = messages.copy()
                
                # 添加助手的 tool_calls 消息
                follow_up_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": json.dumps(tc["function"]["arguments"], ensure_ascii=False)
                            }
                        }
                        for tc in response["tool_calls"]
                    ]
                })
                
                # 添加函数执行结果
                for tr in tool_results:
                    follow_up_messages.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_call_id"],
                        "content": json.dumps(tr["result"], ensure_ascii=False)
                    })
                
                # 再次调用 LLM 获取最终回复
                follow_up_response = self.llm_client.chat(
                    messages=follow_up_messages,
                    tools=None  # 不再需要工具
                )
                assistant_content = follow_up_response.get("content") or "好的，我记住了。"
        
        # 8. 保存助手回复
        if assistant_content:
            self.memory.save_message(
                role="assistant",
                content=assistant_content
            )
        
        # 9. 返回结果
        return {
            "content": assistant_content,
            "emotion": emotion_result.to_dict() if emotion_result else None,
            "tool_calls": response.get("tool_calls")
        }
    
    def chat_stream(self, user_message: str):
        """
        流式处理用户消息
        
        Args:
            user_message: 用户消息
        
        Yields:
            响应内容片段
        """
        # 1. 分析用户情感
        emotion_result = self.emotion_analyzer.analyze(user_message)
        if emotion_result:
            self.memory.working_context.update_emotion(
                emotion_result.emotion_type,
                emotion_result.intensity
            )
        
        # 2. 智能搜索相关历史记忆（方案B核心）
        relevant_context = self.memory.search_relevant_context(user_message)
        
        # 3. 保存用户消息
        self.memory.save_message(
            role="user",
            content=user_message,
            emotion_type=emotion_result.emotion_type if emotion_result else None,
            emotion_intensity=emotion_result.intensity if emotion_result else None
        )
        
        # 4. 组合工作上下文和相关历史
        working_context = self.memory.get_context_for_llm()
        if relevant_context:
            working_context = working_context + "\n" + relevant_context if working_context else relevant_context
        
        # 5. 构建LLM消息
        messages = self.llm_client.build_messages(
            system_prompt=self.system_prompt,
            user_message=user_message,
            history=self.memory.get_messages_for_llm()[:-1],
            working_context=working_context
        )
        
        # 6. 流式调用LLM
        full_content = ""
        for chunk in self.llm_client.chat_stream(messages=messages):
            full_content += chunk
            yield chunk
        
        # 7. 保存助手回复
        if full_content:
            self.memory.save_message(
                role="assistant",
                content=full_content
            )
    
    def get_chat_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取对话历史
        
        Args:
            limit: 返回数量
        
        Returns:
            消息列表
        """
        return self.memory.get_recent_messages(limit=limit)
    
    def get_user_context(self) -> Dict[str, Any]:
        """
        获取用户上下文
        
        Returns:
            上下文字典
        """
        return self.memory.working_context.to_dict()
    
    def update_user_info(self, **kwargs) -> None:
        """
        更新用户信息
        
        Args:
            **kwargs: 用户信息键值对
        """
        self.memory.working_context.set_user_info(**kwargs)
        self.memory.save_working_context()

