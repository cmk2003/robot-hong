"""
多 Agent 架构测试
"""

import pytest
import json
import time
import threading
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


class TestParallelExecution:
    """并行执行优化测试"""
    
    def test_run_parallel_agents_executes_concurrently(self):
        """测试情感分析和记忆检索并行执行"""
        from src.agent.multi_agent_graph import run_parallel_agents
        
        # 记录执行时间和顺序
        execution_log = []
        
        def slow_emotion_run(user_message, context=None):
            execution_log.append(("emotion_start", time.time()))
            time.sleep(0.3)  # 模拟 LLM 调用耗时
            execution_log.append(("emotion_end", time.time()))
            return {"emotion_type": "开心", "intensity": 0.8}
        
        def slow_memory_run(user_message, emotion_result=None):
            execution_log.append(("memory_start", time.time()))
            time.sleep(0.3)  # 模拟 LLM 调用耗时
            execution_log.append(("memory_end", time.time()))
            return {"should_search": False, "retrieved_context": ""}
        
        mock_emotion_agent = Mock()
        mock_emotion_agent.run = slow_emotion_run
        
        mock_memory_agent = Mock()
        mock_memory_agent.run = slow_memory_run
        
        start_time = time.time()
        emotion_result, memory_result = run_parallel_agents(
            mock_emotion_agent,
            mock_memory_agent,
            "测试消息"
        )
        total_time = time.time() - start_time
        
        # 验证结果正确
        assert emotion_result["emotion_type"] == "开心"
        assert memory_result["should_search"] == False
        
        # 验证并行执行（总时间应该接近 0.3 秒，而不是 0.6 秒）
        assert total_time < 0.5, f"并行执行总时间应该 < 0.5 秒，实际: {total_time:.2f} 秒"
        
        # 验证两个任务确实同时启动（开始时间差应该很小）
        emotion_start = next(t for name, t in execution_log if name == "emotion_start")
        memory_start = next(t for name, t in execution_log if name == "memory_start")
        start_diff = abs(emotion_start - memory_start)
        assert start_diff < 0.1, f"两个任务启动时间差应该 < 0.1 秒，实际: {start_diff:.2f} 秒"
    
    def test_run_parallel_agents_handles_exception(self):
        """测试并行执行时异常处理"""
        from src.agent.multi_agent_graph import run_parallel_agents
        
        mock_emotion_agent = Mock()
        mock_emotion_agent.run.return_value = {"emotion_type": "平静", "intensity": 0.5}
        
        mock_memory_agent = Mock()
        mock_memory_agent.run.side_effect = Exception("模拟错误")
        
        # 应该抛出异常
        with pytest.raises(Exception):
            run_parallel_agents(mock_emotion_agent, mock_memory_agent, "测试")


class TestAsyncSaveMemory:
    """异步保存测试"""
    
    def test_async_save_does_not_block(self):
        """测试异步保存不阻塞主线程"""
        from src.agent.multi_agent_graph import async_save_memory
        
        save_started = threading.Event()
        save_completed = threading.Event()
        
        def slow_save(user_message, ai_response):
            save_started.set()
            time.sleep(0.5)  # 模拟慢保存
            save_completed.set()
            return {"saved_count": 1}
        
        mock_save_agent = Mock()
        mock_save_agent.run = slow_save
        
        start_time = time.time()
        async_save_memory(mock_save_agent, "用户消息", "AI回复")
        call_time = time.time() - start_time
        
        # 验证调用立即返回（不等待保存完成）
        assert call_time < 0.1, f"异步保存应该立即返回，实际: {call_time:.2f} 秒"
        
        # 等待后台任务开始
        assert save_started.wait(timeout=1.0), "保存任务应该在后台启动"
        
        # 等待后台任务完成
        assert save_completed.wait(timeout=1.0), "保存任务应该在后台完成"
    
    def test_async_save_handles_exception_gracefully(self):
        """测试异步保存处理异常"""
        from src.agent.multi_agent_graph import async_save_memory
        
        mock_save_agent = Mock()
        mock_save_agent.run.side_effect = Exception("保存失败")
        
        # 不应该抛出异常
        async_save_memory(mock_save_agent, "用户消息", "AI回复")
        
        # 等待后台线程执行
        time.sleep(0.1)
        
        # 验证 run 被调用了
        mock_save_agent.run.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

