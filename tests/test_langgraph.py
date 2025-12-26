"""
LangGraph 相关测试
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestAgentTools:
    """工具定义测试"""
    
    def test_stateless_tools_exist(self):
        """测试无状态工具存在"""
        from src.agent.tools import STATELESS_TOOLS
        
        assert len(STATELESS_TOOLS) == 2
        tool_names = [t.name for t in STATELESS_TOOLS]
        assert "get_current_datetime" in tool_names
        assert "get_weather" in tool_names
    
    def test_get_current_datetime_tool(self):
        """测试时间工具"""
        from src.agent.tools import get_current_datetime
        
        result = get_current_datetime.invoke({})
        
        assert result["success"] is True
        assert "date" in result
        assert "time" in result
        assert "weekday" in result
    
    def test_get_weather_tool(self):
        """测试天气工具（可能因网络问题失败）"""
        from src.agent.tools import get_weather
        
        result = get_weather.invoke({"city": "深圳"})
        
        # 只检查返回结构，不检查成功与否（网络可能不可用）
        assert "city" in result or "error" in result
    
    def test_create_memory_tools(self):
        """测试创建记忆工具"""
        from src.agent.tools import create_memory_tools
        
        # Mock MemoryManager
        mock_memory = Mock()
        mock_memory.working_context = Mock()
        mock_memory.working_context.add_follow_up = Mock()
        mock_memory.save_emotion = Mock()
        mock_memory.save_life_event = Mock()
        mock_memory.update_user_profile = Mock()
        mock_memory.search_messages = Mock(return_value=[])
        mock_memory.get_life_events = Mock(return_value=[])
        mock_memory.get_emotion_history = Mock(return_value=[])
        
        tools = create_memory_tools(mock_memory)
        
        assert len(tools) == 5
        tool_names = [t.name for t in tools]
        assert "save_emotion" in tool_names
        assert "save_life_event" in tool_names
        assert "update_user_profile" in tool_names
        assert "search_memory" in tool_names
        assert "set_follow_up" in tool_names
    
    def test_save_emotion_tool_execution(self):
        """测试 save_emotion 工具执行"""
        from src.agent.tools import create_memory_tools
        
        # Mock MemoryManager
        mock_memory = Mock()
        mock_memory.save_emotion = Mock()
        
        tools = create_memory_tools(mock_memory)
        save_emotion_tool = next(t for t in tools if t.name == "save_emotion")
        
        result = save_emotion_tool.invoke({
            "emotion_type": "喜悦",
            "intensity": 0.8,
            "trigger": "收到好消息"
        })
        
        assert result["success"] is True
        mock_memory.save_emotion.assert_called_once_with(
            emotion_type="喜悦",
            intensity=0.8,
            trigger="收到好消息"
        )


class TestAgentState:
    """状态定义测试"""
    
    def test_agent_state_structure(self):
        """测试 AgentState 结构"""
        from src.agent.graph import AgentState
        
        # 验证字段存在
        annotations = AgentState.__annotations__
        assert "messages" in annotations
        assert "user_input" in annotations
        assert "emotion_result" in annotations
        assert "working_context" in annotations
        assert "final_response" in annotations


class TestCreateAgentGraph:
    """测试 Agent Graph 创建"""
    
    def test_create_graph_returns_compiled_graph(self):
        """测试创建图返回编译后的图"""
        from src.agent.graph import create_agent_graph
        
        # Mock 依赖
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        
        mock_memory = Mock()
        mock_memory.working_context = Mock()
        mock_memory.search_relevant_context = Mock(return_value="")
        mock_memory.save_message = Mock()
        mock_memory.get_context_for_llm = Mock(return_value="")
        mock_memory.get_messages_for_llm = Mock(return_value=[])
        
        graph = create_agent_graph(
            llm=mock_llm,
            memory=mock_memory,
            emotion_analyzer=None,
            system_prompt="测试系统提示词"
        )
        
        assert graph is not None
        # 检查图是可调用的
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")


class TestLangChainChatModel:
    """测试 LangChain ChatModel 工厂"""
    
    def test_create_langchain_chat_model(self):
        """测试创建 LangChain ChatModel"""
        from src.llm.client import create_langchain_chat_model
        from src.config import LLMProviderConfig
        
        # 使用真实的 LLMProviderConfig 创建一个测试配置
        mock_config = Mock(spec=LLMProviderConfig)
        mock_config.api_key = "test-key"
        mock_config.base_url = "http://test-url"
        mock_config.model = "test-model"
        
        # 实际调用工厂方法
        with patch('langchain_openai.ChatOpenAI') as mock_chat:
            from src.llm.client import create_langchain_chat_model
            create_langchain_chat_model(mock_config)
            
            mock_chat.assert_called_once_with(
                api_key="test-key",
                base_url="http://test-url",
                model="test-model",
                temperature=0.7,
                max_tokens=2000
            )


class TestEmotionalAgentLangGraph:
    """EmotionalAgent LangGraph 集成测试"""
    
    def test_agent_creation_with_llm_client(self):
        """测试使用 llm_client 创建 Agent"""
        # Mock LLMClient
        mock_llm_client = Mock()
        mock_llm_client.api_key = "test-key"
        mock_llm_client.base_url = "http://test"
        mock_llm_client.model = "test-model"
        
        with patch('langchain_openai.ChatOpenAI'):
            from src.agent.emotional_agent import EmotionalAgent
            
            agent = EmotionalAgent(
                db_path=":memory:",
                user_id="test_user",
                llm_client=mock_llm_client
            )
            
            assert agent is not None
            assert agent.user_id == "test_user"
            assert agent.llm_client == mock_llm_client
    
    def test_agent_requires_llm(self):
        """测试 Agent 必须提供 LLM"""
        from src.agent.emotional_agent import EmotionalAgent
        
        with pytest.raises(ValueError, match="必须提供"):
            EmotionalAgent(
                db_path=":memory:",
                user_id="test_user"
            )
    
    def test_chat_returns_expected_structure(self):
        """测试 chat 方法返回预期结构"""
        from langchain_core.messages import AIMessage
        
        # Mock LLMClient
        mock_llm_client = Mock()
        mock_llm_client.api_key = "test-key"
        mock_llm_client.base_url = "http://test"
        mock_llm_client.model = "test-model"
        
        with patch('langchain_openai.ChatOpenAI'):
            from src.agent.emotional_agent import EmotionalAgent
            
            agent = EmotionalAgent(
                db_path=":memory:",
                user_id="test_user",
                llm_client=mock_llm_client
            )
            
            # Mock graph
            mock_graph = Mock()
            mock_graph.invoke = Mock(return_value={
                "messages": [AIMessage(content="你好！我是小虹~")],
                "emotion_result": {"emotion_type": "平静", "intensity": 0.5},
                "final_response": "你好！我是小虹~"
            })
            
            # Mock memory init
            agent.memory = Mock()
            agent._graph = mock_graph
            
            result = agent.chat("你好")
            
            assert "content" in result
            assert "emotion" in result
            assert result["content"] == "你好！我是小虹~"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
