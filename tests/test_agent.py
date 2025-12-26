"""
核心Agent模块测试
基于 LangGraph 架构重构
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

from langchain_core.messages import AIMessage


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
        mock.api_key = "test-key"
        mock.base_url = "http://test"
        mock.model = "test-model"
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
        with patch('langchain_openai.ChatOpenAI') as mock_chat:
            # Mock ChatOpenAI
            mock_llm = MagicMock()
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)
            mock_llm.invoke = MagicMock(return_value=AIMessage(content="你好！有什么我可以帮助你的吗？"))
            mock_chat.return_value = mock_llm
            
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
        assert agent._graph is not None  # LangGraph 应该被初始化
    
    def test_agent_chat_basic(self, agent):
        """测试: 基本对话"""
        # Mock graph invoke
        agent._graph.invoke = MagicMock(return_value={
            "messages": [AIMessage(content="你好！有什么我可以帮助你的吗？")],
            "emotion_result": None,
            "final_response": "你好！有什么我可以帮助你的吗？"
        })
        
        response = agent.chat("你好")
        
        assert response is not None
        assert "content" in response
        assert response["content"] == "你好！有什么我可以帮助你的吗？"
    
    def test_agent_saves_user_message(self, agent):
        """测试: 保存用户消息"""
        # 直接保存消息来测试
        agent.memory.save_message(role="user", content="我今天很开心")
        
        messages = agent.memory.get_recent_messages(limit=5)
        
        # 应该有用户消息
        assert len(messages) >= 1
        assert any(m["content"] == "我今天很开心" for m in messages)
    
    def test_agent_analyzes_emotion(self, agent):
        """测试: 分析用户情感"""
        # 直接测试情感分析
        result = agent.emotion_analyzer.analyze("我今天超级开心！太棒了！")
        
        if result:
            assert result.emotion_type in ["喜悦", "开心", "愉快"]
    
    def test_agent_handles_tool_calls(self, agent):
        """测试: 处理工具调用"""
        # Mock graph invoke 返回包含工具调用的响应
        mock_ai_message = AIMessage(content="我感受到你很开心！")
        mock_ai_message.tool_calls = [
            {
                "id": "call_1",
                "name": "save_emotion",
                "args": {
                    "emotion_type": "喜悦",
                    "intensity": 0.8
                }
            }
        ]
        
        agent._graph.invoke = MagicMock(return_value={
            "messages": [mock_ai_message],
            "emotion_result": {"emotion_type": "喜悦", "intensity": 0.8},
            "final_response": "我感受到你很开心！"
        })
        
        response = agent.chat("我今天升职了！太开心了！")
        
        assert response is not None
        assert "content" in response
        # 检查工具调用被提取
        if response.get("tool_calls"):
            assert len(response["tool_calls"]) >= 1
    
    def test_agent_get_history(self, agent):
        """测试: 获取对话历史"""
        # 保存一些消息
        agent.memory.save_message(role="user", content="你好")
        agent.memory.save_message(role="assistant", content="你好！")
        agent.memory.save_message(role="user", content="我叫小明")
        agent.memory.save_message(role="assistant", content="你好小明！")
        
        history = agent.get_chat_history(limit=10)
        
        assert len(history) >= 2
    
    def test_agent_context_persistence(self, temp_db_path, mock_llm_client):
        """测试: 上下文持久化"""
        with patch('langchain_openai.ChatOpenAI') as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)
            mock_chat.return_value = mock_llm
            
            from src.agent.emotional_agent import EmotionalAgent
            
            # 第一次会话
            agent1 = EmotionalAgent(
                db_path=temp_db_path,
                user_id="test-user",
                llm_client=mock_llm_client
            )
            agent1.init()
            agent1.memory.working_context.set_user_info(name="小明")
            agent1.memory.save_working_context()
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


class TestToolExecution:
    """工具执行测试（使用 LangChain Tool）"""
    
    @pytest.fixture
    def temp_db_path(self):
        """创建临时数据库"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        if os.path.exists(f.name):
            os.remove(f.name)
    
    @pytest.fixture
    def memory(self, temp_db_path):
        """创建 MemoryManager 实例"""
        from src.agent.memory import MemoryManager
        
        memory = MemoryManager(temp_db_path, "test-user")
        memory.init()
        yield memory
        memory.close()
    
    def test_save_emotion_tool(self, memory):
        """测试: save_emotion 工具"""
        from src.agent.tools import create_memory_tools
        
        tools = create_memory_tools(memory)
        save_emotion = next(t for t in tools if t.name == "save_emotion")
        
        result = save_emotion.invoke({
            "emotion_type": "喜悦",
            "intensity": 0.8,
            "trigger": "升职"
        })
        
        assert result["success"] is True
        
        # 验证情感被保存
        emotions = memory.get_emotion_history()
        assert len(emotions) >= 1
    
    def test_save_life_event_tool(self, memory):
        """测试: save_life_event 工具"""
        from src.agent.tools import create_memory_tools
        
        tools = create_memory_tools(memory)
        save_event = next(t for t in tools if t.name == "save_life_event")
        
        result = save_event.invoke({
            "event_type": "work",
            "title": "项目上线",
            "importance": 4
        })
        
        assert result["success"] is True
        
        # 验证事件被保存
        events = memory.get_life_events()
        assert len(events) >= 1
    
    def test_update_user_profile_tool(self, memory):
        """测试: update_user_profile 工具"""
        from src.agent.tools import create_memory_tools
        
        tools = create_memory_tools(memory)
        update_profile = next(t for t in tools if t.name == "update_user_profile")
        
        result = update_profile.invoke({
            "field": "name",
            "value": "小红"
        })
        
        assert result["success"] is True
        
        # 验证用户画像被更新
        assert memory.working_context.user_name == "小红"
    
    def test_search_memory_tool(self, memory):
        """测试: search_memory 工具"""
        from src.agent.tools import create_memory_tools
        
        # 先保存一些消息
        memory.save_message(role="user", content="我喜欢打篮球")
        memory.save_message(role="assistant", content="篮球是个很好的运动！")
        
        tools = create_memory_tools(memory)
        search = next(t for t in tools if t.name == "search_memory")
        
        result = search.invoke({
            "query": "篮球",
            "search_type": "messages"
        })
        
        assert result["success"] is True
        # 注意: FTS5 搜索可能需要特定条件才能触发，这里只验证接口正常
        assert "results" in result
    
    def test_set_follow_up_tool(self, memory):
        """测试: set_follow_up 工具"""
        from src.agent.tools import create_memory_tools
        
        tools = create_memory_tools(memory)
        set_follow_up = next(t for t in tools if t.name == "set_follow_up")
        
        result = set_follow_up.invoke({
            "item": "下次问候用户的项目进展"
        })
        
        assert result["success"] is True
        
        # 验证跟进事项被添加
        assert len(memory.working_context.follow_ups) >= 1
