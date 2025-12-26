"""
多 Agent 工作流
基于 LangGraph 的多 Agent 协作架构
"""

from typing import TypedDict, Annotated, Optional, Dict, Any, List, Generator
from langgraph.graph import StateGraph, END

from .agents import (
    EmotionAgent,
    MemoryRetrievalAgent,
    ResponseAgent,
    MemorySaveAgent,
    ReviewAgent,
)
from .memory import MemoryManager
from ..utils.logger import get_logger

logger = get_logger("multi_agent")


def _get_model_name(llm_client) -> str:
    """获取 LLM 客户端的模型名称"""
    if hasattr(llm_client, 'model'):
        return llm_client.model
    return "unknown"


class MultiAgentState(TypedDict):
    """多 Agent 状态定义"""
    # 用户输入
    user_input: str
    # 对话历史
    chat_history: List[Dict[str, str]]
    # 用户画像
    user_profile: Dict[str, Any]
    
    # 情感分析结果
    emotion_result: Optional[Dict[str, Any]]
    # 记忆检索结果
    memory_context: str
    
    # AI 回复
    ai_response: str
    # 回复是否通过评审
    review_approved: bool
    # 评审反馈
    review_feedback: str
    # 重写次数
    rewrite_count: int
    
    # 最终回复
    final_response: str
    # 保存操作结果
    save_result: Optional[Dict[str, Any]]


def create_multi_agent_graph(
    llm_client,
    memory: MemoryManager,
    max_rewrites: int = 2
):
    """
    创建多 Agent 工作流
    
    Args:
        llm_client: LLM 客户端
        memory: MemoryManager 实例
        max_rewrites: 最大重写次数
    
    Returns:
        编译后的 StateGraph
    """
    # 初始化各 Agent
    emotion_agent = EmotionAgent(llm_client)
    memory_agent = MemoryRetrievalAgent(llm_client, memory)
    response_agent = ResponseAgent(llm_client)
    save_agent = MemorySaveAgent(llm_client, memory)
    review_agent = ReviewAgent(llm_client)
    
    # ============ 节点定义 ============
    
    def emotion_node(state: MultiAgentState) -> dict:
        """情感分析节点"""
        logger.info("[Node: emotion] 开始情感分析...")
        
        result = emotion_agent.run(
            user_message=state["user_input"],
            context=None  # 可以添加历史上下文
        )
        
        # 更新 memory 中的情感状态
        if result.get("emotion_type"):
            memory.working_context.update_emotion(
                result["emotion_type"],
                result.get("intensity", 0.5)
            )
        
        return {"emotion_result": result}
    
    def memory_node(state: MultiAgentState) -> dict:
        """记忆检索节点"""
        logger.info("[Node: memory] 开始记忆检索...")
        
        result = memory_agent.run(
            user_message=state["user_input"],
            emotion_result=state.get("emotion_result")
        )
        
        # 合并现有的上下文
        existing_context = memory.get_context_for_llm() or ""
        retrieved = result.get("retrieved_context", "")
        
        combined_context = existing_context
        if retrieved:
            combined_context = f"{existing_context}\n\n{retrieved}" if existing_context else retrieved
        
        return {"memory_context": combined_context}
    
    def response_node(state: MultiAgentState) -> dict:
        """回复生成节点"""
        logger.info("[Node: response] 生成回复...")
        
        # 如果是重写，使用 rewrite 方法
        if state.get("rewrite_count", 0) > 0 and state.get("ai_response"):
            new_response = response_agent.rewrite(
                original_response=state["ai_response"],
                feedback=state.get("review_feedback", ""),
                messages=_build_messages(state)
            )
            return {"ai_response": new_response}
        
        # 首次生成
        result = response_agent.run(
            user_message=state["user_input"],
            emotion_result=state.get("emotion_result"),
            memory_context=state.get("memory_context"),
            user_profile=state.get("user_profile"),
            chat_history=state.get("chat_history")
        )
        
        return {"ai_response": result.get("content", "")}
    
    def review_node(state: MultiAgentState) -> dict:
        """质量评审节点"""
        logger.info("[Node: review] 评审回复...")
        
        result = review_agent.run(
            user_message=state["user_input"],
            ai_response=state["ai_response"]
        )
        
        return {
            "review_approved": result.get("approved", True),
            "review_feedback": result.get("suggestion", "")
        }
    
    def save_node(state: MultiAgentState) -> dict:
        """记忆保存节点"""
        logger.info("[Node: save] 保存记忆...")
        
        # 保存用户消息
        emotion_result = state.get("emotion_result", {})
        memory.save_message(
            role="user",
            content=state["user_input"],
            emotion_type=emotion_result.get("emotion_type"),
            emotion_intensity=emotion_result.get("intensity")
        )
        
        # 保存 AI 回复
        memory.save_message(
            role="assistant",
            content=state["final_response"]
        )
        
        # 调用保存 Agent 分析并保存其他信息
        result = save_agent.run(
            user_message=state["user_input"],
            ai_response=state["final_response"]
        )
        
        return {"save_result": result}
    
    def finalize_node(state: MultiAgentState) -> dict:
        """最终处理节点"""
        logger.info("[Node: finalize] 完成处理")
        return {"final_response": state["ai_response"]}
    
    def increment_rewrite(state: MultiAgentState) -> dict:
        """增加重写计数"""
        return {"rewrite_count": state.get("rewrite_count", 0) + 1}
    
    # ============ 条件边 ============
    
    def should_rewrite(state: MultiAgentState) -> str:
        """判断是否需要重写"""
        if state.get("review_approved", True):
            return "finalize"
        
        if state.get("rewrite_count", 0) >= max_rewrites:
            logger.warning(f"[should_rewrite] 达到最大重写次数 {max_rewrites}，强制通过")
            return "finalize"
        
        logger.info(f"[should_rewrite] 需要重写，当前次数: {state.get('rewrite_count', 0)}")
        return "rewrite"
    
    # ============ 构建图 ============
    
    workflow = StateGraph(MultiAgentState)
    
    # 添加节点
    workflow.add_node("emotion", emotion_node)
    workflow.add_node("memory", memory_node)
    workflow.add_node("response", response_node)
    workflow.add_node("review", review_node)
    workflow.add_node("increment_rewrite", increment_rewrite)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("save", save_node)
    
    # 添加边
    workflow.set_entry_point("emotion")
    workflow.add_edge("emotion", "memory")
    workflow.add_edge("memory", "response")
    workflow.add_edge("response", "review")
    
    # 条件边：评审通过/不通过
    workflow.add_conditional_edges(
        "review",
        should_rewrite,
        {
            "finalize": "finalize",
            "rewrite": "increment_rewrite"
        }
    )
    
    # 重写循环
    workflow.add_edge("increment_rewrite", "response")
    
    # 最终处理
    workflow.add_edge("finalize", "save")
    workflow.add_edge("save", END)
    
    return workflow.compile()


def _build_messages(state: MultiAgentState) -> List[Dict[str, str]]:
    """构建消息列表"""
    messages = []
    
    if state.get("chat_history"):
        messages.extend(state["chat_history"][-10:])
    
    messages.append({
        "role": "user",
        "content": state["user_input"]
    })
    
    return messages


class MultiAgentRunner:
    """
    多 Agent 运行器
    提供便捷的调用接口，支持为每个 Agent 配置不同的模型
    """
    
    def __init__(
        self,
        llm_client,
        memory: MemoryManager,
        max_rewrites: int = 2,
        agent_llm_clients: Dict[str, Any] = None
    ):
        """
        初始化多 Agent 运行器
        
        Args:
            llm_client: 默认 LLM 客户端
            memory: MemoryManager 实例
            max_rewrites: 最大重写次数
            agent_llm_clients: 各 Agent 的 LLM 客户端字典
                {
                    "emotion": LLMClient,
                    "memory": LLMClient,
                    "response": LLMClient,
                    "save": LLMClient,
                    "review": LLMClient,
                }
                如果某个 Agent 没有配置，则使用默认 llm_client
        """
        self.llm_client = llm_client
        self.memory = memory
        self.agent_llm_clients = agent_llm_clients or {}
        
        # 获取各 Agent 的 LLM 客户端
        emotion_llm = self.agent_llm_clients.get("emotion", llm_client)
        memory_llm = self.agent_llm_clients.get("memory", llm_client)
        response_llm = self.agent_llm_clients.get("response", llm_client)
        save_llm = self.agent_llm_clients.get("save", llm_client)
        review_llm = self.agent_llm_clients.get("review", llm_client)
        
        # 创建各 Agent
        self.emotion_agent = EmotionAgent(emotion_llm)
        self.memory_retrieval_agent = MemoryRetrievalAgent(memory_llm, memory)
        self.response_agent = ResponseAgent(response_llm)
        self.save_agent = MemorySaveAgent(save_llm, memory)
        self.review_agent = ReviewAgent(review_llm)
        
        # 创建工作流图（使用默认 llm_client，因为图内部会创建自己的 Agent）
        # 注意：如果需要在图中使用不同的 LLM，需要修改 create_multi_agent_graph
        self.graph = create_multi_agent_graph(llm_client, memory, max_rewrites)
        
        # 日志记录各 Agent 使用的模型
        self._log_agent_models()
    
    def _log_agent_models(self):
        """记录各 Agent 使用的模型"""
        models = {
            "emotion": _get_model_name(self.emotion_agent.llm),
            "memory": _get_model_name(self.memory_retrieval_agent.llm),
            "response": _get_model_name(self.response_agent.llm),
            "save": _get_model_name(self.save_agent.llm),
            "review": _get_model_name(self.review_agent.llm),
        }
        logger.info(f"[MultiAgentRunner] Agent 模型配置: {models}")
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        处理用户消息（非流式）
        
        Args:
            user_message: 用户消息
        
        Returns:
            {
                "content": str,
                "emotion": dict,
                "save_result": dict
            }
        """
        logger.info("=" * 50)
        logger.info(f"[MultiAgent] 用户消息: {user_message}")
        
        # 准备初始状态
        initial_state = {
            "user_input": user_message,
            "chat_history": self.memory.get_messages_for_llm(),
            "user_profile": self.memory.get_user_profile(),
            "emotion_result": None,
            "memory_context": "",
            "ai_response": "",
            "review_approved": False,
            "review_feedback": "",
            "rewrite_count": 0,
            "final_response": "",
            "save_result": None
        }
        
        # 运行工作流
        result = self.graph.invoke(initial_state)
        
        return {
            "content": result["final_response"],
            "emotion": result.get("emotion_result"),
            "save_result": result.get("save_result")
        }
    
    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        """
        处理用户消息（流式输出）
        只有回复生成使用流式，其他 Agent 使用非流式
        
        Args:
            user_message: 用户消息
        
        Yields:
            回复内容片段
        """
        logger.info("=" * 50)
        logger.info(f"[MultiAgent-Stream] 用户消息: {user_message}")
        
        # 1. 情感分析（非流式）
        emotion_result = self.emotion_agent.run(user_message)
        
        # 更新情感状态
        if emotion_result.get("emotion_type"):
            self.memory.working_context.update_emotion(
                emotion_result["emotion_type"],
                emotion_result.get("intensity", 0.5)
            )
        
        # 2. 记忆检索（非流式）
        memory_result = self.memory_retrieval_agent.run(
            user_message=user_message,
            emotion_result=emotion_result
        )
        
        # 合并上下文
        existing_context = self.memory.get_context_for_llm() or ""
        retrieved = memory_result.get("retrieved_context", "")
        memory_context = f"{existing_context}\n\n{retrieved}" if existing_context and retrieved else (existing_context or retrieved)
        
        # 3. 保存用户消息
        self.memory.save_message(
            role="user",
            content=user_message,
            emotion_type=emotion_result.get("emotion_type"),
            emotion_intensity=emotion_result.get("intensity")
        )
        
        # 4. 回复生成（流式）
        full_response = ""
        for chunk in self.response_agent.run_stream(
            user_message=user_message,
            emotion_result=emotion_result,
            memory_context=memory_context,
            user_profile=self.memory.get_user_profile(),
            chat_history=self.memory.get_messages_for_llm()[:-1]  # 不包括刚保存的
        ):
            full_response += chunk
            yield chunk
        
        # 5. 保存 AI 回复
        self.memory.save_message(role="assistant", content=full_response)
        
        # 6. 记忆保存（后台，非流式）
        try:
            self.save_agent.run(
                user_message=user_message,
                ai_response=full_response
            )
        except Exception as e:
            logger.error(f"[MultiAgent-Stream] 记忆保存失败: {e}")
        
        logger.info(f"[MultiAgent-Stream] 完成，回复长度: {len(full_response)}")

