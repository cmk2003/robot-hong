"""
情感机器人主入口
基于Gradio的对话界面
"""

import os
import sys
from pathlib import Path

# 确保src目录在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

from src.config import Config, config
from src.agent.emotional_agent import EmotionalAgent
from src.llm.client import LLMClient
from src.utils.logger import get_logger

# 初始化日志
logger = get_logger("main")

# 全局Agent实例
_agent: EmotionalAgent = None


def get_agent() -> EmotionalAgent:
    """获取或创建Agent实例"""
    global _agent
    
    if _agent is None:
        logger.info("初始化情感机器人...")
        
        # 确保数据目录存在
        config.ensure_data_dir()
        
        # 获取LLM配置
        try:
            llm_config = config.get_llm_config()
        except ValueError as e:
            logger.error(f"LLM配置错误: {e}")
            raise
        
        # 创建默认 LLM 客户端
        llm_client = LLMClient(llm_config)
        
        # 获取 Agent 模式
        agent_mode = config.agent_mode
        if agent_mode not in ["single", "multi"]:
            logger.warning(f"无效的 AGENT_MODE: {agent_mode}，使用默认值 single")
            agent_mode = "single"
        
        # 为多 Agent 模式创建各 Agent 的 LLM 客户端
        agent_llm_clients = {}
        if agent_mode == "multi":
            agent_names = ["emotion", "memory", "response", "save", "review"]
            for agent_name in agent_names:
                agent_config = config.get_agent_llm_config(agent_name)
                # 如果模型不同，创建新的客户端
                if agent_config.model != llm_config.model:
                    agent_llm_clients[agent_name] = LLMClient(agent_config)
                    logger.info(f"  {agent_name} Agent 使用模型: {agent_config.model}")
                else:
                    agent_llm_clients[agent_name] = llm_client
            
            # 打印多 Agent 模型配置
            logger.info("多 Agent 模型配置:")
            for name, client in agent_llm_clients.items():
                logger.info(f"  {name}: {client.model}")
        
        # 创建Agent
        _agent = EmotionalAgent(
            db_path=config.database_path,
            user_id="default-user",
            llm_client=llm_client,
            mode=agent_mode,
            agent_llm_clients=agent_llm_clients
        )
        _agent.init()
        
        logger.info(f"情感机器人初始化完成！默认模型: {llm_config.model}, 模式: {agent_mode}")
    
    return _agent


def chat(message: str, history: list) -> str:
    """
    处理聊天消息（使用非流式以支持工具调用）
    
    Args:
        message: 用户消息
        history: 对话历史 [[user, bot], ...]
    
    Returns:
        机器人回复
    """
    if not message.strip():
        return ""
    
    try:
        agent = get_agent()
        
        # 使用非流式方法（支持工具调用：时间、天气等）
        response = agent.chat(message)
        content = response.get("content", "")
        
        # 如果没有内容，返回默认消息
        if not content:
            return "好的，我记住了~"
        
        return content
    
    except Exception as e:
        logger.error(f"对话出错: {e}")
        return f"抱歉，我遇到了一些问题：{str(e)}"


def chat_stream(message: str, history: list):
    """
    处理聊天消息（流式输出）
    工具调用使用非流式，最后一轮对话使用流式输出
    
    Args:
        message: 用户消息
        history: 对话历史 [[user, bot], ...]
    
    Yields:
        机器人回复片段
    """
    if not message.strip():
        yield ""
        return
    
    try:
        agent = get_agent()
        
        # 使用流式方法（最后一轮对话流式输出）
        full_response = ""
        for chunk in agent.chat_stream_final_only(message):
            full_response += chunk
            yield full_response
        
        # 如果没有内容，返回默认消息
        if not full_response:
            yield "好的~"
    
    except Exception as e:
        logger.error(f"对话出错: {e}")
        yield f"抱歉，我遇到了一些问题：{str(e)}"


def create_ui(use_stream: bool = True):
    """
    创建Gradio界面 - 使用简化的ChatInterface
    
    Args:
        use_stream: 是否使用流式输出，默认True
    """
    
    demo = gr.ChatInterface(
        fn=chat_stream if use_stream else chat,
        title="🌸 小虹 - 情感陪伴机器人",
        description="""
        你好！我是小虹，一个温暖的情感陪伴机器人。
        我会记住我们的对话，理解你的情感，陪伴你度过每一天。
        
        💡 **提示**: 你可以和我分享任何心情和经历，我会认真倾听和回应。
        """,
        examples=[
            "你好，我今天心情不太好",
            "最近工作压力很大，感觉很焦虑",
            "我升职了！想和你分享这个好消息",
            "有时候感觉很孤独",
        ],
        theme=gr.themes.Soft(
            primary_hue="pink",
            secondary_hue="purple",
        ),
        retry_btn="🔄 重试",
        undo_btn="↩️ 撤销",
        clear_btn="🗑️ 清空",
    )
    
    return demo


def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("启动情感机器人...")
    logger.info(f"环境: {config.env}")
    logger.info(f"LLM提供商: {config.llm_provider}")
    logger.info(f"数据库路径: {config.database_path}")
    logger.info("=" * 50)
    
    # 创建UI
    demo = create_ui()
    
    # 启动服务
    demo.launch(
        server_name=config.gradio_server_name,
        server_port=config.gradio_server_port,
        share=False,
        show_error=config.is_development,
    )


if __name__ == "__main__":
    main()
