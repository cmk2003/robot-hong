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
        
        # 创建LLM客户端
        llm_client = LLMClient(llm_config)
        
        # 创建Agent
        _agent = EmotionalAgent(
            db_path=config.database_path,
            user_id="default-user",
            llm_client=llm_client
        )
        _agent.init()
        
        logger.info(f"情感机器人初始化完成！使用模型: {llm_config.model}")
    
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


def create_ui():
    """创建Gradio界面 - 使用简化的ChatInterface"""
    
    demo = gr.ChatInterface(
        fn=chat,
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
