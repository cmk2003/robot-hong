"""
情感Agent模块
核心Agent类，支持单 Agent 和多 Agent 两种模式
支持最后一轮对话的流式输出
"""

from typing import Dict, Any, List, Generator, Literal, Optional

from .memory import MemoryManager
from .graph import create_agent_graph
from .tools import create_memory_tools, STATELESS_TOOLS
from ..emotion.analyzer import EmotionAnalyzer
from ..llm.client import LLMClient, create_langchain_chat_model
from ..llm.prompts import SYSTEM_PROMPT
from ..config import LLMProviderConfig
from ..utils.logger import get_logger

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

logger = get_logger("emotional_agent")


# Agent 模式类型
AgentMode = Literal["single", "multi"]


class EmotionalAgent:
    """
    情感陪伴Agent
    支持两种模式：
    - single: 单 Agent 模式（默认），使用 LangGraph 单节点架构
    - multi: 多 Agent 模式，使用情感分析、记忆检索、回复生成、记忆保存、质量评审 5 个 Agent
    """
    
    def __init__(
        self,
        db_path: str,
        user_id: str,
        llm_client: LLMClient = None,
        llm_config: LLMProviderConfig = None,
        system_prompt: str = None,
        mode: AgentMode = "single",
        agent_llm_clients: Dict[str, LLMClient] = None
    ):
        """
        初始化Agent
        
        Args:
            db_path: 数据库路径
            user_id: 用户ID
            llm_client: LLM客户端（默认，旧接口兼容）
            llm_config: LLM 配置（用于创建 LangChain ChatModel）
            system_prompt: 系统提示词
            mode: Agent 模式，"single" 或 "multi"
            agent_llm_clients: 各 Agent 的 LLM 客户端（多 Agent 模式）
                {
                    "emotion": LLMClient,   # 情感分析
                    "memory": LLMClient,    # 记忆检索
                    "response": LLMClient,  # 回复生成
                    "save": LLMClient,      # 记忆保存
                    "review": LLMClient,    # 质量评审
                }
        """
        self.db_path = db_path
        self.user_id = user_id
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.mode = mode
        self.agent_llm_clients = agent_llm_clients or {}
        
        # 初始化模块
        self.memory = MemoryManager(db_path, user_id)
        
        # 保存 llm_client 用于情感分析和多 Agent 模式
        self.llm_client = llm_client
        self.emotion_analyzer = EmotionAnalyzer(llm_client) if llm_client else None
        
        # LangChain ChatModel（单 Agent 模式使用）
        if llm_config:
            self._llm = create_langchain_chat_model(llm_config)
        elif llm_client:
            # 从旧的 llm_client 创建 LangChain 模型
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                api_key=llm_client.api_key,
                base_url=llm_client.base_url,
                model=llm_client.model,
                temperature=0.7,
                max_tokens=2000
            )
        else:
            raise ValueError("必须提供 llm_client 或 llm_config")
        
        # LangGraph（延迟初始化）
        self._graph = None
        
        # 多 Agent 运行器（延迟初始化）
        self._multi_agent_runner = None
        
        logger.info(f"[EmotionalAgent] 创建实例，模式: {mode}")
    
    def init(self) -> None:
        """初始化Agent"""
        self.memory.init()
        
        if self.mode == "multi":
            # 多 Agent 模式
            from .multi_agent_graph import MultiAgentRunner
            self._multi_agent_runner = MultiAgentRunner(
                llm_client=self.llm_client,
                memory=self.memory,
                max_rewrites=2,
                agent_llm_clients=self.agent_llm_clients
            )
            logger.info("[EmotionalAgent] 初始化完成，使用多 Agent 架构")
        else:
            # 单 Agent 模式
            self._graph = create_agent_graph(
                llm=self._llm,
                memory=self.memory,
                emotion_analyzer=self.emotion_analyzer,
                system_prompt=self.system_prompt
            )
            logger.info("[EmotionalAgent] 初始化完成，使用单 Agent 架构")
    
    def close(self) -> None:
        """关闭Agent"""
        self.memory.close()
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        处理用户消息
        
        Args:
            user_message: 用户消息
        
        Returns:
            响应结果 {"content": str, "emotion": dict, "tool_calls": list}
        """
        logger.info("=" * 50)
        logger.info(f"[用户消息] {user_message}")
        
        if self.mode == "multi":
            # 多 Agent 模式
            result = self._multi_agent_runner.chat(user_message)
            return {
                "content": result.get("content", ""),
                "emotion": result.get("emotion"),
                "tool_calls": None
            }
        else:
            # 单 Agent 模式
            result = self._graph.invoke({
                "user_input": user_message,
                "messages": [],
                "emotion_result": None,
                "working_context": "",
                "relevant_context": "",
                "final_response": ""
            })
            
            # 提取工具调用信息
            tool_calls = self._extract_tool_calls(result["messages"])
            
            return {
                "content": result["final_response"],
                "emotion": result.get("emotion_result"),
                "tool_calls": tool_calls
            }
    
    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        """
        流式处理用户消息
        实现：工具调用使用非流式，最后一轮对话使用流式输出
        
        Args:
            user_message: 用户消息
        
        Yields:
            响应内容片段（str）
        """
        logger.info("=" * 50)
        logger.info(f"[用户消息-流式] {user_message}")
        
        # 1. 预处理：情感分析 + 搜索历史
        emotion_result = None
        if self.emotion_analyzer:
            try:
                result = self.emotion_analyzer.analyze(user_message)
                if result:
                    emotion_result = result.to_dict()
                    self.memory.working_context.update_emotion(
                        result.emotion_type,
                        result.intensity
                    )
                    logger.info(f"[chat_stream] 情感分析: {emotion_result}")
            except Exception as e:
                logger.warning(f"[chat_stream] 情感分析失败: {e}")
        
        # 2. 搜索相关历史
        relevant_context = self.memory.search_relevant_context(user_message)
        
        # 3. 保存用户消息
        self.memory.save_message(
            role="user",
            content=user_message,
            emotion_type=emotion_result.get("emotion_type") if emotion_result else None,
            emotion_intensity=emotion_result.get("intensity") if emotion_result else None
        )
        
        # 4. 构建上下文和消息
        working_context = self.memory.get_context_for_llm()
        if relevant_context:
            working_context = f"{working_context}\n{relevant_context}" if working_context else relevant_context
        
        full_system = self.system_prompt
        if working_context:
            full_system += f"\n\n## 当前上下文\n{working_context}"
        
        # 获取历史消息
        history = self.memory.get_messages_for_llm()[:-1]  # 不包括刚添加的
        
        messages = [{"role": "system", "content": full_system}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})
        
        # 5. 构建工具定义
        memory_tools = create_memory_tools(self.memory)
        all_tools = memory_tools + STATELESS_TOOLS
        tools_schema = [self._tool_to_openai_schema(t) for t in all_tools]
        
        # 6. 工具调用循环 + 最后一轮流式输出
        max_iterations = 5
        iteration = 0
        full_response = ""
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"[chat_stream] 迭代 {iteration}")
            
            # 使用支持工具调用检测的流式请求
            content_buffer = ""
            final_event = None
            
            for event in self.llm_client.chat_stream_with_tools(
                messages=messages,
                tools=tools_schema,
                temperature=0.7,
                max_tokens=2000
            ):
                if event["type"] == "content":
                    content_buffer += event["data"]
                    # 如果这可能是最后一轮，立即输出
                    yield event["data"]
                elif event["type"] == "done":
                    final_event = event
                    break
            
            if final_event and final_event.get("has_tool_calls"):
                # 有工具调用，需要执行工具
                logger.info(f"[chat_stream] 检测到工具调用: {final_event['tool_calls']}")
                
                # 添加 AI 消息（包含工具调用）
                messages.append({
                    "role": "assistant",
                    "content": content_buffer or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": str(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], dict) else tc["function"]["arguments"]
                            }
                        }
                        for tc in final_event["tool_calls"]
                    ]
                })
                
                # 执行每个工具
                for tool_call in final_event["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]
                    
                    # 查找并执行工具
                    tool_result = self._execute_tool(all_tools, tool_name, tool_args)
                    
                    # 添加工具结果消息
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(tool_result)
                    })
                
                # 清空已输出的内容，因为需要重新生成
                full_response = ""
                # 注意：已经 yield 的内容无法撤回，需要在前端处理
                # 这里简单处理：继续循环，下一轮会生成完整回复
                
            else:
                # 没有工具调用，这是最后一轮
                full_response = content_buffer
                break
        
        # 7. 保存助手回复
        if full_response:
            self.memory.save_message(role="assistant", content=full_response)
            logger.info(f"[chat_stream] 保存回复: {full_response[:50]}...")
    
    def chat_stream_final_only(self, user_message: str) -> Generator[str, None, None]:
        """
        流式处理用户消息（只在最后一轮流式输出）
        工具调用过程完全使用非流式，避免中间输出
        
        Args:
            user_message: 用户消息
        
        Yields:
            响应内容片段（str）
        """
        logger.info("=" * 50)
        logger.info(f"[用户消息-流式] {user_message}")
        
        # 多 Agent 模式：使用 MultiAgentRunner 的流式方法
        if self.mode == "multi":
            for chunk in self._multi_agent_runner.chat_stream(user_message):
                yield chunk
            return
        
        # ========== 以下是单 Agent 模式 ==========
        
        # 1. 预处理：情感分析 + 搜索历史
        emotion_result = None
        if self.emotion_analyzer:
            try:
                result = self.emotion_analyzer.analyze(user_message)
                if result:
                    emotion_result = result.to_dict()
                    self.memory.working_context.update_emotion(
                        result.emotion_type,
                        result.intensity
                    )
            except Exception as e:
                logger.warning(f"[chat_stream_final_only] 情感分析失败: {e}")
        
        # 2. 搜索相关历史
        relevant_context = self.memory.search_relevant_context(user_message)
        
        # 3. 保存用户消息
        self.memory.save_message(
            role="user",
            content=user_message,
            emotion_type=emotion_result.get("emotion_type") if emotion_result else None,
            emotion_intensity=emotion_result.get("intensity") if emotion_result else None
        )
        
        # 4. 构建上下文和消息
        working_context = self.memory.get_context_for_llm()
        if relevant_context:
            working_context = f"{working_context}\n{relevant_context}" if working_context else relevant_context
        
        full_system = self.system_prompt
        if working_context:
            full_system += f"\n\n## 当前上下文\n{working_context}"
        
        history = self.memory.get_messages_for_llm()[:-1]
        
        messages = [{"role": "system", "content": full_system}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})
        
        # 5. 构建工具定义
        memory_tools = create_memory_tools(self.memory)
        all_tools = memory_tools + STATELESS_TOOLS
        tools_schema = [self._tool_to_openai_schema(t) for t in all_tools]
        
        # 6. 工具调用循环（非流式）
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # 非流式调用，检查是否需要工具
            response = self.llm_client.chat(
                messages=messages,
                tools=tools_schema,
                temperature=0.7,
                max_tokens=2000
            )
            
            if response.get("tool_calls"):
                # 有工具调用，执行工具
                logger.info(f"[chat_stream_final_only] 工具调用: {response['tool_calls']}")
                
                messages.append({
                    "role": "assistant",
                    "content": response.get("content"),
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": str(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], dict) else tc["function"]["arguments"]
                            }
                        }
                        for tc in response["tool_calls"]
                    ]
                })
                
                for tool_call in response["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]
                    tool_result = self._execute_tool(all_tools, tool_name, tool_args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(tool_result)
                    })
            else:
                # 没有工具调用，最后一轮使用流式输出
                logger.info("[chat_stream_final_only] 最后一轮，使用流式输出")
                
                full_response = ""
                for chunk in self.llm_client.chat_stream(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                ):
                    full_response += chunk
                    yield chunk
                
                # 保存回复
                if full_response:
                    self.memory.save_message(role="assistant", content=full_response)
                    logger.info(f"[chat_stream_final_only] 保存回复: {full_response[:50]}...")
                
                break
    
    def _tool_to_openai_schema(self, tool) -> Dict[str, Any]:
        """将 LangChain 工具转换为 OpenAI 函数定义格式"""
        # 获取参数 schema
        parameters = {"type": "object", "properties": {}}
        
        try:
            # 方法1: 使用 LangChain 工具的 args 属性（推荐）
            if hasattr(tool, 'args') and tool.args:
                parameters = {
                    "type": "object",
                    "properties": tool.args,
                    "required": []
                }
            # 方法2: 使用 get_input_schema（如果可用）
            elif hasattr(tool, 'get_input_schema'):
                schema = tool.get_input_schema()
                if hasattr(schema, 'model_json_schema'):
                    parameters = schema.model_json_schema()
                elif hasattr(schema, 'schema'):
                    parameters = schema.schema()
            # 方法3: 直接访问 args_schema
            elif hasattr(tool, 'args_schema') and tool.args_schema:
                schema_class = tool.args_schema
                if hasattr(schema_class, 'model_json_schema'):
                    parameters = schema_class.model_json_schema()
                elif hasattr(schema_class, 'schema'):
                    parameters = schema_class.schema()
        except Exception as e:
            logger.warning(f"[_tool_to_openai_schema] 获取工具 {tool.name} schema 失败: {e}")
        
        # 移除不需要的字段
        parameters.pop('title', None)
        parameters.pop('description', None)
        
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters
            }
        }
    
    def _execute_tool(self, tools: List, tool_name: str, tool_args: Dict) -> str:
        """执行指定的工具"""
        for tool in tools:
            if tool.name == tool_name:
                try:
                    if isinstance(tool_args, str):
                        import json
                        tool_args = json.loads(tool_args)
                    result = tool.invoke(tool_args)
                    logger.info(f"[execute_tool] {tool_name} 结果: {result}")
                    return str(result)
                except Exception as e:
                    logger.error(f"[execute_tool] {tool_name} 失败: {e}")
                    return f"工具执行失败: {e}"
        return f"未找到工具: {tool_name}"
    
    def _extract_tool_calls(self, messages: List) -> List[Dict]:
        """从消息中提取工具调用信息"""
        tool_calls = []
        for msg in messages:
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "id": tc.get("id"),
                        "function": {
                            "name": tc.get("name"),
                            "arguments": tc.get("args", {})
                        }
                    })
        return tool_calls if tool_calls else None
    
    def get_chat_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return self.memory.get_recent_messages(limit=limit)
    
    def get_user_context(self) -> Dict[str, Any]:
        """获取用户上下文"""
        return self.memory.working_context.to_dict()
    
    def update_user_info(self, **kwargs) -> None:
        """更新用户信息"""
        self.memory.working_context.set_user_info(**kwargs)
        self.memory.save_working_context()
