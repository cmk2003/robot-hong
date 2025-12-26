"""
LangGraph 状态图定义
基于 LangGraph 的情感 Agent 工作流
"""

from __future__ import annotations
from typing import TypedDict, Annotated, Sequence, Literal, Optional, Dict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .tools import create_memory_tools, STATELESS_TOOLS
from .memory import MemoryManager
from ..emotion.analyzer import EmotionAnalyzer
from ..llm.prompts import SYSTEM_PROMPT
from ..utils.logger import get_logger

logger = get_logger("langgraph")


class AgentState(TypedDict):
    """Agent 状态定义"""
    # 消息历史（自动累加）
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # 用户原始输入
    user_input: str
    # 情感分析结果
    emotion_result: Optional[Dict]
    # 工作上下文
    working_context: str
    # 相关历史上下文
    relevant_context: str
    # 最终回复
    final_response: str


def create_agent_graph(
    llm,
    memory: MemoryManager,
    emotion_analyzer: EmotionAnalyzer = None,
    system_prompt: str = None
):
    """
    创建 Agent 状态图
    
    Args:
        llm: LangChain ChatModel 实例
        memory: MemoryManager 实例
        emotion_analyzer: EmotionAnalyzer 实例（可选）
        system_prompt: 系统提示词
    
    Returns:
        编译后的 StateGraph
    """
    system_prompt = system_prompt or SYSTEM_PROMPT
    
    # 创建工具列表
    memory_tools = create_memory_tools(memory)
    all_tools = memory_tools + STATELESS_TOOLS
    
    # 绑定工具到 LLM
    llm_with_tools = llm.bind_tools(all_tools)
    
    # 创建工具节点
    tool_node = ToolNode(all_tools)
    
    # ============ 节点定义 ============
    
    def preprocess(state: AgentState) -> dict:
        """预处理节点：情感分析 + 搜索历史 + 构建上下文"""
        user_input = state["user_input"]
        
        logger.info(f"[preprocess] 用户输入: {user_input}")
        
        # 1. 情感分析
        emotion_result = None
        if emotion_analyzer:
            try:
                result = emotion_analyzer.analyze(user_input)
                if result:
                    emotion_result = result.to_dict()
                    memory.working_context.update_emotion(
                        result.emotion_type,
                        result.intensity
                    )
                    logger.info(f"[preprocess] 情感分析: {emotion_result}")
            except Exception as e:
                logger.warning(f"[preprocess] 情感分析失败: {e}")
        
        # 2. 搜索相关历史
        relevant_context = memory.search_relevant_context(user_input)
        
        # 3. 保存用户消息
        memory.save_message(
            role="user",
            content=user_input,
            emotion_type=emotion_result.get("emotion_type") if emotion_result else None,
            emotion_intensity=emotion_result.get("intensity") if emotion_result else None
        )
        
        # 4. 获取工作上下文
        working_context = memory.get_context_for_llm()
        if relevant_context:
            working_context = f"{working_context}\n{relevant_context}" if working_context else relevant_context
        
        # 5. 构建消息
        full_system = system_prompt
        if working_context:
            full_system += f"\n\n## 当前上下文\n{working_context}"
        
        # 获取历史消息
        history = memory.get_messages_for_llm()[:-1]  # 不包括刚添加的
        
        messages = [SystemMessage(content=full_system)]
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=user_input))
        
        return {
            "messages": messages,
            "emotion_result": emotion_result,
            "working_context": working_context,
            "relevant_context": relevant_context
        }
    
    def agent(state: AgentState) -> dict:
        """Agent 节点：调用 LLM"""
        messages = state["messages"]
        logger.info(f"[agent] 调用 LLM, 消息数: {len(messages)}")
        
        response = llm_with_tools.invoke(messages)
        logger.info(f"[agent] LLM 响应: {response.content[:100] if response.content else 'None'}...")
        
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(f"[agent] 工具调用: {[tc['name'] for tc in response.tool_calls]}")
        
        return {"messages": [response]}
    
    def postprocess(state: AgentState) -> dict:
        """后处理节点：保存回复"""
        messages = state["messages"]
        
        # 找到最后一条 AI 消息
        final_content = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_content = msg.content
                break
        
        # 保存助手回复
        if final_content:
            memory.save_message(role="assistant", content=final_content)
            logger.info(f"[postprocess] 保存回复: {final_content[:50]}...")
        
        return {"final_response": final_content}
    
    # ============ 条件边 ============
    
    def should_continue(state: AgentState) -> Literal["tools", "postprocess"]:
        """判断是否需要调用工具"""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "postprocess"
    
    # ============ 构建图 ============
    
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("preprocess", preprocess)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    workflow.add_node("postprocess", postprocess)
    
    # 添加边
    workflow.set_entry_point("preprocess")
    workflow.add_edge("preprocess", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "postprocess": "postprocess"
        }
    )
    workflow.add_edge("tools", "agent")  # 工具执行后返回 agent
    workflow.add_edge("postprocess", END)
    
    return workflow.compile()

