"""
情感机器人主入口
基于 Gradio 的多用户对话界面
支持用户名 + 激活码验证
"""

import os
import sys
import json
from pathlib import Path

# 确保src目录在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

from src.config import Config, config
from src.agent.agent_pool import AgentPool, username_to_user_id
from src.utils.logger import get_logger

# 初始化日志
logger = get_logger("main")

# 全局 Agent 池
_agent_pool: AgentPool = None


def get_agent_pool() -> AgentPool:
    """获取或创建 Agent 池"""
    global _agent_pool
    
    if _agent_pool is None:
        logger.info("初始化 Agent 池...")
        
        # 确保数据目录存在
        config.ensure_data_dir()
        
        # 创建 Agent 池
        _agent_pool = AgentPool(config)
        
        logger.info(f"Agent 池初始化完成！模式: {config.agent_mode}")
    
    return _agent_pool


def verify_activation_code(code: str) -> bool:
    """
    验证激活码
    
    Args:
        code: 用户输入的激活码
    
    Returns:
        是否验证通过
    """
    expected_code = config.activation_code
    
    # 如果未配置激活码，允许所有访问
    if not expected_code:
        logger.warning("未配置激活码，允许所有用户访问")
        return True
    
    return code.strip() == expected_code


def parse_user_state(state_str: str) -> dict:
    """解析用户状态 JSON 字符串"""
    if not state_str:
        return {}
    try:
        return json.loads(state_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def login(username: str, activation_code: str):
    """
    用户登录
    
    Args:
        username: 用户名
        activation_code: 激活码
    
    Returns:
        (user_state, login_visible, chat_visible, error_msg, error_visible, welcome_msg)
    """
    # 验证用户名
    if not username or not username.strip():
        return (
            "",             # user_state（空字符串表示未登录）
            gr.update(visible=True),   # login_page
            gr.update(visible=False),  # chat_page
            "❌ 请输入用户名",          # error_msg
            gr.update(visible=True),   # error_visible
            ""              # welcome_msg
        )
    
    # 验证激活码
    if not verify_activation_code(activation_code):
        logger.warning(f"用户 {username} 激活码验证失败")
        return (
            "",
            gr.update(visible=True),
            gr.update(visible=False),
            "❌ 激活码无效，请检查后重试",
            gr.update(visible=True),
            ""
        )
    
    # 登录成功
    username = username.strip()
    user_id = username_to_user_id(username)
    
    logger.info(f"用户登录成功: {username} (user_id: {user_id})")
    
    # 获取或创建该用户的 Agent
    pool = get_agent_pool()
    agent = pool.get_agent(user_id)
    
    # 获取用户名（可能之前保存过）
    display_name = agent.memory.working_context.user_name or username
    
    # 如果是新用户，更新用户名
    if not agent.memory.working_context.user_name:
        agent.memory.working_context.set_user_info(name=username)
        agent.memory.save_working_context()
    
    user_state = json.dumps({
        "user_id": user_id,
        "username": display_name
    })
    
    welcome_msg = f"### 👋 欢迎回来，{display_name}！"
    
    return (
        user_state,
        gr.update(visible=False),  # 隐藏登录页
        gr.update(visible=True),   # 显示对话页
        "",
        gr.update(visible=False),
        welcome_msg
    )


def chat(message: str, history: list, user_state_str: str):
    """
    处理聊天消息
    
    Args:
        message: 用户消息
        history: 对话历史
        user_state_str: 用户状态 JSON 字符串
    
    Returns:
        机器人回复
    """
    if not message.strip():
        return ""
    
    user_state = parse_user_state(user_state_str)
    if not user_state.get("user_id"):
        return "❌ 请先登录"
    
    try:
        user_id = user_state.get("user_id")
        pool = get_agent_pool()
        agent = pool.get_agent(user_id)
        
        # 使用非流式方法
        response = agent.chat(message)
        content = response.get("content", "")
        
        if not content:
            return "好的，我记住了~"
        
        return content
    
    except Exception as e:
        logger.error(f"对话出错: {e}")
        return f"抱歉，我遇到了一些问题：{str(e)}"


def chat_stream(message: str, history: list, user_state_str: str):
    """
    流式处理聊天消息
    
    Args:
        message: 用户消息
        history: 对话历史
        user_state_str: 用户状态 JSON 字符串
    
    Yields:
        机器人回复片段
    """
    if not message.strip():
        yield ""
        return
    
    user_state = parse_user_state(user_state_str)
    if not user_state.get("user_id"):
        yield "❌ 请先登录"
        return
    
    try:
        user_id = user_state.get("user_id")
        pool = get_agent_pool()
        agent = pool.get_agent(user_id)
        
        full_response = ""
        for chunk in agent.chat_stream_final_only(message):
            full_response += chunk
            yield full_response
        
        if not full_response:
            yield "好的~"
    
    except Exception as e:
        logger.error(f"对话出错: {e}")
        yield f"抱歉，我遇到了一些问题：{str(e)}"


def logout(user_state_str: str):
    """
    用户登出
    
    Args:
        user_state_str: 用户状态 JSON 字符串
    
    Returns:
        (user_state, login_visible, chat_visible, history)
    """
    user_state = parse_user_state(user_state_str)
    if user_state.get("user_id"):
        username = user_state.get("username", "未知用户")
        logger.info(f"用户登出: {username}")
    
    return (
        "",             # 清空用户状态（空字符串）
        gr.update(visible=True),   # 显示登录页
        gr.update(visible=False),  # 隐藏对话页
        []              # 清空历史
    )


def create_ui():
    """创建 Gradio 界面"""
    
    # 自定义 CSS
    custom_css = """
    .login-container {
        max-width: 400px;
        margin: 100px auto;
        padding: 40px;
        border-radius: 16px;
        background: linear-gradient(135deg, #fff5f5 0%, #fff0f6 100%);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    .login-title {
        text-align: center;
        color: #e91e63;
        margin-bottom: 30px;
    }
    .chat-header {
        padding: 16px;
        background: linear-gradient(135deg, #fce4ec 0%, #f3e5f5 100%);
        border-radius: 12px;
        margin-bottom: 16px;
    }
    .logout-btn {
        float: right;
    }
    """
    
    with gr.Blocks(
        title="🌸 小虹 - 情感陪伴机器人",
        theme=gr.themes.Soft(
            primary_hue="pink",
            secondary_hue="purple",
        ),
        css=custom_css,
        analytics_enabled=False  # 禁用分析，避免 API schema 问题
    ) as demo:
        
        # 用户状态（使用字符串避免复杂类型导致的 schema 问题）
        user_state = gr.State(value="")
        
        # ========== 登录页面 ==========
        with gr.Column(visible=True, elem_classes="login-container") as login_page:
            gr.Markdown(
                "# 🌸 小虹\n### 情感陪伴机器人",
                elem_classes="login-title"
            )
            
            gr.Markdown(
                """
                你好！我是小虹，一个温暖的情感陪伴机器人。
                我会记住我们的对话，理解你的情感，陪伴你度过每一天。
                
                请输入用户名和激活码开始对话 ✨
                """
            )
            
            username_input = gr.Textbox(
                label="用户名",
                placeholder="请输入您的名字",
                max_lines=1
            )
            
            code_input = gr.Textbox(
                label="激活码",
                placeholder="请输入激活码",
                type="password",
                max_lines=1
            )
            
            login_btn = gr.Button("🚀 开始对话", variant="primary", size="lg")
            
            login_error = gr.Markdown(visible=False, elem_classes="error-msg")
        
        # ========== 对话页面 ==========
        with gr.Column(visible=False) as chat_page:
            
            # 顶部欢迎栏
            with gr.Row(elem_classes="chat-header"):
                welcome_msg = gr.Markdown("### 👋 欢迎！")
                logout_btn = gr.Button("🚪 退出登录", size="sm", elem_classes="logout-btn")
            
            # 对话界面
            chatbot = gr.Chatbot(
                label="对话",
                height=500,
                show_copy_button=True,
                avatar_images=(None, "https://em-content.zobj.net/source/apple/391/cherry-blossom_1f338.png")
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="消息",
                    placeholder="和小虹说说你的心情吧...",
                    max_lines=3,
                    scale=9
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)
            
            # 快捷操作
            with gr.Row():
                gr.Examples(
                    examples=[
                        "你好，我今天心情不太好",
                        "最近工作压力很大，感觉很焦虑",
                        "我升职了！想和你分享这个好消息",
                        "有时候感觉很孤独",
                    ],
                    inputs=msg_input,
                    label="💡 试试这些话题"
                )
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ 清空对话")
        
        # ========== 事件绑定 ==========
        
        # 登录（禁用 API 避免 schema 问题）
        login_btn.click(
            fn=login,
            inputs=[username_input, code_input],
            outputs=[user_state, login_page, chat_page, login_error, login_error, welcome_msg],
            api_name=False
        )
        
        # 回车登录
        code_input.submit(
            fn=login,
            inputs=[username_input, code_input],
            outputs=[user_state, login_page, chat_page, login_error, login_error, welcome_msg],
            api_name=False
        )
        
        # 发送消息（流式）
        def respond(message, history, user_state_str):
            """处理消息并更新历史"""
            if not message.strip():
                return history, ""
            
            # 添加用户消息到历史
            history = history + [[message, ""]]
            
            # 流式获取回复
            for response in chat_stream(message, history, user_state_str):
                history[-1][1] = response
                yield history, ""
        
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, user_state],
            outputs=[chatbot, msg_input],
            api_name=False
        )
        
        send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, user_state],
            outputs=[chatbot, msg_input],
            api_name=False
        )
        
        # 清空对话
        clear_btn.click(
            fn=lambda: [],
            outputs=[chatbot],
            api_name=False
        )
        
        # 登出
        logout_btn.click(
            fn=logout,
            inputs=[user_state],
            outputs=[user_state, login_page, chat_page, chatbot],
            api_name=False
        )
    
    return demo


def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("启动情感机器人（多用户版）...")
    logger.info(f"环境: {config.env}")
    logger.info(f"LLM提供商: {config.llm_provider}")
    logger.info(f"Agent模式: {config.agent_mode}")
    logger.info(f"数据库路径: {config.database_path}")
    logger.info(f"日志路径: {config.log_path}")
    logger.info(f"激活码已配置: {'是' if config.activation_code else '否（允许所有用户）'}")
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
