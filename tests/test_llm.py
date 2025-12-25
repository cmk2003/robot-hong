"""
LLM客户端模块测试
TDD: 先写测试，测试应该失败
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json


class TestLLMClient:
    """LLM客户端测试"""
    
    @pytest.fixture
    def mock_config(self):
        """模拟配置"""
        from src.config import LLMProviderConfig
        return LLMProviderConfig(
            base_url="https://api.example.com/v1",
            model="test-model",
            api_key="test-api-key"
        )
    
    def test_llm_client_init(self, mock_config):
        """测试: LLM客户端初始化"""
        from src.llm.client import LLMClient
        
        client = LLMClient(mock_config)
        
        assert client.model == "test-model"
        assert client.base_url == "https://api.example.com/v1"
    
    def test_llm_client_build_messages(self, mock_config):
        """测试: 构建消息列表"""
        from src.llm.client import LLMClient
        
        client = LLMClient(mock_config)
        
        messages = client.build_messages(
            system_prompt="你是一个情感陪伴机器人",
            user_message="你好",
            history=[
                {"role": "user", "content": "之前的消息"},
                {"role": "assistant", "content": "之前的回复"}
            ]
        )
        
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "你好"
    
    @patch("src.llm.client.OpenAI")
    def test_llm_client_chat(self, mock_openai_class, mock_config):
        """测试: 发送聊天请求"""
        from src.llm.client import LLMClient
        
        # 模拟OpenAI响应
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="你好！有什么我可以帮助你的吗？",
                    tool_calls=None
                )
            )
        ]
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        client = LLMClient(mock_config)
        
        response = client.chat(
            messages=[
                {"role": "system", "content": "你是一个助手"},
                {"role": "user", "content": "你好"}
            ]
        )
        
        assert response["content"] == "你好！有什么我可以帮助你的吗？"
        assert response["tool_calls"] is None
    
    @patch("src.llm.client.OpenAI")
    def test_llm_client_chat_with_functions(self, mock_openai_class, mock_config):
        """测试: 带函数调用的聊天"""
        from src.llm.client import LLMClient
        
        # 模拟带tool_calls的响应
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "save_emotion"
        mock_tool_call.function.arguments = json.dumps({
            "emotion_type": "喜悦",
            "intensity": 0.8
        })
        
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="我感受到你很开心！",
                    tool_calls=[mock_tool_call]
                )
            )
        ]
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        client = LLMClient(mock_config)
        
        functions = [
            {
                "type": "function",
                "function": {
                    "name": "save_emotion",
                    "description": "保存情感记录",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "emotion_type": {"type": "string"},
                            "intensity": {"type": "number"}
                        }
                    }
                }
            }
        ]
        
        response = client.chat(
            messages=[{"role": "user", "content": "我今天很开心！"}],
            tools=functions
        )
        
        assert response["tool_calls"] is not None
        assert len(response["tool_calls"]) == 1
        assert response["tool_calls"][0]["function"]["name"] == "save_emotion"


class TestPrompts:
    """提示词模板测试"""
    
    def test_system_prompt_exists(self):
        """测试: 系统提示词存在"""
        from src.llm.prompts import SYSTEM_PROMPT
        
        assert SYSTEM_PROMPT is not None
        assert len(SYSTEM_PROMPT) > 100  # 应该有足够长的提示词
        assert "情感" in SYSTEM_PROMPT or "陪伴" in SYSTEM_PROMPT
    
    def test_working_context_template(self):
        """测试: 工作上下文模板"""
        from src.llm.prompts import format_working_context
        
        context = {
            "user_name": "小明",
            "current_emotion": "焦虑",
            "recent_events": ["工作压力大"]
        }
        
        formatted = format_working_context(context)
        
        assert "小明" in formatted
        assert "焦虑" in formatted


class TestFunctions:
    """函数定义测试"""
    
    def test_function_definitions_exist(self):
        """测试: 函数定义存在"""
        from src.llm.functions import TOOL_DEFINITIONS
        
        assert isinstance(TOOL_DEFINITIONS, list)
        assert len(TOOL_DEFINITIONS) > 0
    
    def test_save_emotion_function(self):
        """测试: save_emotion函数定义"""
        from src.llm.functions import TOOL_DEFINITIONS
        
        function_names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
        
        assert "save_emotion" in function_names
    
    def test_save_life_event_function(self):
        """测试: save_life_event函数定义"""
        from src.llm.functions import TOOL_DEFINITIONS
        
        function_names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
        
        assert "save_life_event" in function_names

