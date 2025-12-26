"""
情感Agent模块
核心Agent类，基于 LangGraph 重构
"""

from typing import Dict, Any, List

from .memory import MemoryManager
from .graph import create_agent_graph
from ..emotion.analyzer import EmotionAnalyzer
from ..llm.client import LLMClient, create_langchain_chat_model
from ..llm.prompts import SYSTEM_PROMPT
from ..config import LLMProviderConfig
from ..utils.logger import get_logger

from langchain_core.messages import AIMessage

logger = get_logger("emotional_agent")


class EmotionalAgent:
    """
    情感陪伴Agent
    基于 LangGraph 架构，具备长期记忆能力
    """
    
    def __init__(
        self,
        db_path: str,
        user_id: str,
        llm_client: LLMClient = None,
        llm_config: LLMProviderConfig = None,
        system_prompt: str = None
    ):
        """
        初始化Agent
        
        Args:
            db_path: 数据库路径
            user_id: 用户ID
            llm_client: LLM客户端（旧接口兼容）
            llm_config: LLM 配置（用于创建 LangChain ChatModel）
            system_prompt: 系统提示词
        """
        self.db_path = db_path
        self.user_id = user_id
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        
        # 初始化模块
        self.memory = MemoryManager(db_path, user_id)
        
        # 保存 llm_client 用于情感分析
        self.llm_client = llm_client
        self.emotion_analyzer = EmotionAnalyzer(llm_client) if llm_client else None
        
        # LangChain ChatModel
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
    
    def init(self) -> None:
        """初始化Agent"""
        self.memory.init()
        
        # 创建 LangGraph
        self._graph = create_agent_graph(
            llm=self._llm,
            memory=self.memory,
            emotion_analyzer=self.emotion_analyzer,
            system_prompt=self.system_prompt
        )
        
        logger.info("[EmotionalAgent] 初始化完成，使用 LangGraph 架构")
    
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
        
        # 调用 LangGraph
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
    
    def chat_stream(self, user_message: str):
        """
        流式处理用户消息
        
        Args:
            user_message: 用户消息
        
        Yields:
            响应内容片段
        """
        logger.info("=" * 50)
        logger.info(f"[用户消息-流式] {user_message}")
        
        full_content = ""
        
        # 使用 LangGraph stream
        for event in self._graph.stream(
            {
                "user_input": user_message,
                "messages": [],
                "emotion_result": None,
                "working_context": "",
                "relevant_context": "",
                "final_response": ""
            },
            stream_mode="values"
        ):
            # 检查 agent 节点的输出
            if "messages" in event:
                messages = event["messages"]
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        # 计算增量
                        new_content = last_msg.content
                        if len(new_content) > len(full_content):
                            delta = new_content[len(full_content):]
                            full_content = new_content
                            yield delta
    
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
