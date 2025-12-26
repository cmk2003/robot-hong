"""
多 Agent 架构测试
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch

# 导入测试目标
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.agents.base import BaseAgent
from src.agent.agents.emotion_agent import EmotionAgent
from src.agent.agents.memory_agent import MemoryRetrievalAgent
from src.agent.agents.response_agent import ResponseAgent
from src.agent.agents.save_agent import MemorySaveAgent
from src.agent.agents.review_agent import ReviewAgent


class TestBaseAgent:
    """BaseAgent 基类测试"""
    
    def test_parse_json_simple(self):
        """测试简单 JSON 解析"""
        mock_llm = Mock()
        
        # 创建一个具体子类用于测试
        class TestAgent(BaseAgent):
            def run(self, **kwargs):
                return {}
        
        agent = TestAgent(mock_llm, "test")
        
        result = agent._parse_json('{"key": "value"}')
        assert result == {"key": "value"}
    
    def test_parse_json_with_markdown(self):
        """测试从 markdown 代码块解析 JSON"""
        mock_llm = Mock()
        
        class TestAgent(BaseAgent):
            def run(self, **kwargs):
                return {}
        
        agent = TestAgent(mock_llm, "test")
        
        text = '''这是一些说明文字

```json
{"emotion": "happy", "score": 0.8}
```

这是结尾'''
        
        result = agent._parse_json(text)
        assert result["emotion"] == "happy"
        assert result["score"] == 0.8
    
    def test_parse_json_with_extra_text(self):
        """测试带有额外文字的 JSON 解析"""
        mock_llm = Mock()
        
        class TestAgent(BaseAgent):
            def run(self, **kwargs):
                return {}
        
        agent = TestAgent(mock_llm, "test")
        
        text = '根据分析，结果如下：{"result": true}'
        
        result = agent._parse_json(text)
        assert result["result"] == True


class TestEmotionAgent:
    """EmotionAgent 测试"""
    
    def test_run_returns_emotion_result(self):
        """测试情感分析返回结果"""
        mock_llm = Mock()
        mock_llm.chat.return_value = {
            "content": json.dumps({
                "emotion_type": "焦虑",
                "intensity": 0.7,
                "trigger": "工作压力",
                "needs": "倾诉"
            })
        }
        
        agent = EmotionAgent(mock_llm)
        result = agent.run("我最近工作压力好大啊")
        
        assert result["emotion_type"] == "焦虑"
        assert result["intensity"] == 0.7
        assert result["needs"] == "倾诉"
    
    def test_run_with_invalid_response(self):
        """测试无效响应时返回默认值"""
        mock_llm = Mock()
        mock_llm.chat.return_value = {"content": "这不是JSON"}
        
        agent = EmotionAgent(mock_llm)
        result = agent.run("你好")
        
        # 应该返回默认值
        assert result["emotion_type"] == "平静"
        assert result["intensity"] == 0.3
    
    def test_validate_intensity_range(self):
        """测试强度值范围验证"""
        mock_llm = Mock()
        mock_llm.chat.return_value = {
            "content": json.dumps({
                "emotion_type": "开心",
                "intensity": 1.5,  # 超出范围
            })
        }
        
        agent = EmotionAgent(mock_llm)
        result = agent.run("太开心了！")
        
        # intensity 应该被限制在 0-1
        assert 0.0 <= result["intensity"] <= 1.0


class TestMemoryRetrievalAgent:
    """MemoryRetrievalAgent 测试"""
    
    def test_run_no_search_needed(self):
        """测试不需要检索的情况"""
        mock_llm = Mock()
        mock_llm.chat.return_value = {
            "content": json.dumps({
                "should_search": False,
                "reasoning": "简单问候，不需要检索"
            })
        }
        
        mock_memory = Mock()
        
        agent = MemoryRetrievalAgent(mock_llm, mock_memory)
        result = agent.run("你好！")
        
        assert result["should_search"] == False
        assert result["retrieved_context"] == ""
    
    def test_run_with_search(self):
        """测试需要检索的情况"""
        mock_llm = Mock()
        mock_llm.chat.return_value = {
            "content": json.dumps({
                "should_search": True,
                "search_queries": ["工作", "压力"],
                "search_types": ["messages"],
                "reasoning": "用户提到工作压力"
            })
        }
        
        mock_memory = Mock()
        mock_memory.search_messages.return_value = [
            {"role": "user", "content": "我最近工作很忙"}
        ]
        
        agent = MemoryRetrievalAgent(mock_llm, mock_memory)
        result = agent.run("还是那个工作的事...")
        
        assert result["should_search"] == True
        assert len(result["retrieved_context"]) > 0


class TestReviewAgent:
    """ReviewAgent 测试"""
    
    def test_approve_good_response(self):
        """测试通过好的回复"""
        mock_llm = Mock()
        mock_llm.chat.return_value = {
            "content": json.dumps({
                "approved": True,
                "score": 8,
                "issues": [],
                "reasoning": "回复自然，符合人设"
            })
        }
        
        agent = ReviewAgent(mock_llm)
        result = agent.run(
            user_message="我今天心情不好",
            ai_response="哎呀，怎么啦？跟我说说嘛~"
        )
        
        assert result["approved"] == True
        assert result["score"] >= 7
    
    def test_reject_list_response(self):
        """测试拒绝包含列表的回复"""
        mock_llm = Mock()
        mock_llm.chat.return_value = {
            "content": json.dumps({
                "approved": False,
                "score": 4,
                "issues": ["使用了列表格式"],
                "suggestion": "去掉列表，用自然的语言表达"
            })
        }
        
        agent = ReviewAgent(mock_llm)
        result = agent.run(
            user_message="我该怎么办",
            ai_response="你可以：1. 休息一下 2. 找人聊聊 3. 出去走走"
        )
        
        assert result["approved"] == False
    
    def test_quick_check_detects_list(self):
        """测试快速检查能检测列表"""
        mock_llm = Mock()
        agent = ReviewAgent(mock_llm)
        
        # 包含列表的回复应该不通过
        assert agent._quick_check("1. 第一点 2. 第二点") == False
        assert agent._quick_check("• 项目一 • 项目二") == False
        
        # 正常回复应该通过
        assert agent._quick_check("哎呀，别难过啦~") == True
    
    def test_quick_check_detects_long_response(self):
        """测试快速检查能检测过长回复"""
        mock_llm = Mock()
        agent = ReviewAgent(mock_llm)
        
        long_response = "很长的回复" * 50
        assert agent._quick_check(long_response) == False


class TestIntegration:
    """集成测试"""
    
    def test_multi_agent_flow(self):
        """测试多 Agent 流程"""
        # 这个测试需要更复杂的 mock 设置
        # 实际运行时会测试完整流程
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

