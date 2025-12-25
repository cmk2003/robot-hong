"""
核心Agent模块测试
TDD: 先写测试
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock


class TestEmotionalAgent:
    """情感Agent测试"""
    
    @pytest.fixture
    def temp_db_path(self):
        """创建临时数据库"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        if os.path.exists(f.name):
            os.remove(f.name)
    
    @pytest.fixture
    def mock_llm_client(self):
        """模拟LLM客户端"""
        mock = MagicMock()
        mock.chat.return_value = {
            "content": "你好！有什么我可以帮助你的吗？",
            "tool_calls": None
        }
        mock.build_messages.return_value = [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "你好"}
        ]
        return mock
    
    @pytest.fixture
    def agent(self, temp_db_path, mock_llm_client):
        """创建Agent实例"""
        from src.agent.emotional_agent import EmotionalAgent
        
        agent = EmotionalAgent(
            db_path=temp_db_path,
            user_id="test-user",
            llm_client=mock_llm_client
        )
        agent.init()
        yield agent
        agent.close()
    
    def test_agent_init(self, agent):
        """测试: Agent初始化"""
        assert agent is not None
        assert agent.user_id == "test-user"
        assert agent.memory is not None
        assert agent.emotion_analyzer is not None
    
    def test_agent_chat_basic(self, agent, mock_llm_client):
        """测试: 基本对话"""
        response = agent.chat("你好")
        
        assert response is not None
        assert "content" in response
        # LLM应该被调用
        mock_llm_client.chat.assert_called()
    
    def test_agent_saves_user_message(self, agent):
        """测试: 保存用户消息"""
        agent.chat("我今天很开心")
        
        messages = agent.memory.get_recent_messages(limit=5)
        
        # 应该有用户消息和助手回复
        assert len(messages) >= 1
    
    def test_agent_analyzes_emotion(self, agent):
        """测试: 分析用户情感"""
        agent.chat("我今天超级开心！太棒了！")
        
        # 工作上下文应该更新了情感
        context = agent.memory.working_context
        
        assert context.current_emotion is not None
    
    def test_agent_handles_tool_calls(self, agent, mock_llm_client):
        """测试: 处理工具调用"""
        # 模拟LLM返回工具调用
        mock_llm_client.chat.return_value = {
            "content": "我感受到你很开心！",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "save_emotion",
                        "arguments": {
                            "emotion_type": "喜悦",
                            "intensity": 0.8
                        }
                    }
                }
            ]
        }
        
        response = agent.chat("我今天升职了！太开心了！")
        
        # 情感应该被保存
        assert response is not None
    
    def test_agent_save_life_event_tool(self, agent, mock_llm_client):
        """测试: 生活事件工具调用"""
        mock_llm_client.chat.return_value = {
            "content": "恭喜你升职！这是个很棒的消息！",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "save_life_event",
                        "arguments": {
                            "event_type": "work",
                            "title": "升职加薪",
                            "importance": 5
                        }
                    }
                }
            ]
        }
        
        agent.chat("我升职了！")
        
        events = agent.memory.get_life_events()
        
        assert len(events) >= 1
    
    def test_agent_get_history(self, agent, mock_llm_client):
        """测试: 获取对话历史"""
        agent.chat("你好")
        agent.chat("我叫小明")
        
        history = agent.get_chat_history(limit=10)
        
        assert len(history) >= 2
    
    def test_agent_context_persistence(self, temp_db_path, mock_llm_client):
        """测试: 上下文持久化"""
        from src.agent.emotional_agent import EmotionalAgent
        
        # 第一次会话
        agent1 = EmotionalAgent(
            db_path=temp_db_path,
            user_id="test-user",
            llm_client=mock_llm_client
        )
        agent1.init()
        agent1.memory.working_context.set_user_info(name="小明")
        agent1.close()
        
        # 第二次会话
        agent2 = EmotionalAgent(
            db_path=temp_db_path,
            user_id="test-user",
            llm_client=mock_llm_client
        )
        agent2.init()
        
        # 应该记住用户名
        assert agent2.memory.working_context.user_name == "小明"
        
        agent2.close()


class TestFunctionExecutor:
    """函数执行器测试"""
    
    @pytest.fixture
    def temp_db_path(self):
        """创建临时数据库"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        if os.path.exists(f.name):
            os.remove(f.name)
    
    @pytest.fixture
    def executor(self, temp_db_path):
        """创建执行器实例"""
        from src.agent.emotional_agent import FunctionExecutor
        from src.agent.memory import MemoryManager
        
        memory = MemoryManager(temp_db_path, "test-user")
        memory.init()
        
        executor = FunctionExecutor(memory)
        yield executor
        
        memory.close()
    
    def test_execute_save_emotion(self, executor):
        """测试: 执行save_emotion"""
        result = executor.execute("save_emotion", {
            "emotion_type": "喜悦",
            "intensity": 0.8,
            "trigger": "升职"
        })
        
        assert result["success"] is True
    
    def test_execute_save_life_event(self, executor):
        """测试: 执行save_life_event"""
        result = executor.execute("save_life_event", {
            "event_type": "work",
            "title": "项目上线",
            "importance": 4
        })
        
        assert result["success"] is True
    
    def test_execute_update_user_profile(self, executor):
        """测试: 执行update_user_profile"""
        result = executor.execute("update_user_profile", {
            "field": "name",
            "value": "小红"
        })
        
        assert result["success"] is True
    
    def test_execute_unknown_function(self, executor):
        """测试: 执行未知函数"""
        result = executor.execute("unknown_function", {})
        
        assert result["success"] is False

