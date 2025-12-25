"""
记忆管理模块测试
TDD: 先写测试
"""

import pytest
import os
import tempfile
from typing import Dict, Any


class TestWorkingContext:
    """工作上下文测试"""
    
    @pytest.fixture
    def context(self):
        """创建上下文实例"""
        from src.agent.context import WorkingContext
        return WorkingContext()
    
    def test_context_init(self, context):
        """测试: 上下文初始化"""
        assert context is not None
        assert context.user_name is None
        assert context.current_emotion is None
    
    def test_context_set_user_info(self, context):
        """测试: 设置用户信息"""
        context.set_user_info(name="小明", age=25, occupation="工程师")
        
        assert context.user_name == "小明"
        assert context.user_info["age"] == 25
    
    def test_context_update_emotion(self, context):
        """测试: 更新情感状态"""
        context.update_emotion("喜悦", 0.8)
        
        assert context.current_emotion == "喜悦"
        assert context.emotion_intensity == 0.8
    
    def test_context_add_recent_event(self, context):
        """测试: 添加最近事件"""
        context.add_recent_event("升职了")
        context.add_recent_event("搬新家")
        
        assert len(context.recent_events) == 2
        assert "升职了" in context.recent_events
    
    def test_context_recent_events_limit(self, context):
        """测试: 最近事件数量限制"""
        for i in range(15):
            context.add_recent_event(f"事件{i}")
        
        # 应该只保留最近的N个事件
        assert len(context.recent_events) <= 10
    
    def test_context_add_follow_up(self, context):
        """测试: 添加待跟进事项"""
        context.add_follow_up("询问面试结果")
        
        assert "询问面试结果" in context.follow_ups
    
    def test_context_to_dict(self, context):
        """测试: 转换为字典"""
        context.set_user_info(name="小明")
        context.update_emotion("平静", 0.5)
        
        d = context.to_dict()
        
        assert isinstance(d, dict)
        assert d["user_name"] == "小明"
        assert d["current_emotion"] == "平静"
    
    def test_context_from_dict(self, context):
        """测试: 从字典加载"""
        data = {
            "user_name": "小红",
            "current_emotion": "焦虑",
            "emotion_intensity": 0.7,
            "recent_events": ["考试", "面试"]
        }
        
        context.load_from_dict(data)
        
        assert context.user_name == "小红"
        assert context.current_emotion == "焦虑"
        assert len(context.recent_events) == 2
    
    def test_context_clear_emotion_history(self, context):
        """测试: 清除情感历史"""
        context.update_emotion("喜悦", 0.8)
        context.update_emotion("悲伤", 0.6)
        
        context.clear_emotion_history()
        
        assert len(context.emotion_history) == 0


class TestMemoryManager:
    """记忆管理器测试"""
    
    @pytest.fixture
    def temp_db_path(self):
        """创建临时数据库"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        if os.path.exists(f.name):
            os.remove(f.name)
    
    @pytest.fixture
    def memory_manager(self, temp_db_path):
        """创建记忆管理器实例"""
        from src.agent.memory import MemoryManager
        mm = MemoryManager(temp_db_path, user_id="test-user")
        mm.init()
        yield mm
        mm.close()
    
    def test_memory_manager_init(self, memory_manager):
        """测试: 记忆管理器初始化"""
        assert memory_manager is not None
        assert memory_manager.user_id == "test-user"
    
    def test_save_and_get_message(self, memory_manager):
        """测试: 保存和获取消息"""
        memory_manager.save_message(
            role="user",
            content="你好",
            emotion_type="平静"
        )
        
        messages = memory_manager.get_recent_messages(limit=5)
        
        assert len(messages) == 1
        assert messages[0]["content"] == "你好"
    
    def test_fifo_queue_management(self, memory_manager):
        """测试: FIFO队列管理"""
        # 添加多条消息
        for i in range(25):
            memory_manager.add_to_queue({"role": "user", "content": f"消息{i}"})
        
        queue = memory_manager.get_queue()
        
        # 队列应该有大小限制
        assert len(queue) <= 20
    
    def test_save_emotion_record(self, memory_manager):
        """测试: 保存情感记录"""
        memory_manager.save_emotion("喜悦", 0.8, trigger="完成项目")
        
        history = memory_manager.get_emotion_history(limit=5)
        
        assert len(history) == 1
        assert history[0]["emotion_type"] == "喜悦"
    
    def test_save_life_event(self, memory_manager):
        """测试: 保存生活事件"""
        memory_manager.save_life_event(
            event_type="work",
            title="升职加薪",
            description="终于升职了",
            importance=5
        )
        
        events = memory_manager.get_life_events()
        
        assert len(events) == 1
        assert events[0]["title"] == "升职加薪"
    
    def test_search_messages(self, memory_manager):
        """测试: 搜索消息"""
        memory_manager.save_message(role="user", content="Today I feel happy")
        memory_manager.save_message(role="user", content="Work is stressful")
        
        # 搜索（使用英文确保FTS正常工作）
        results = memory_manager.search_messages("happy")
        
        assert len(results) >= 1
    
    def test_working_context_persistence(self, memory_manager):
        """测试: 工作上下文持久化"""
        # 更新上下文
        memory_manager.working_context.set_user_info(name="小明")
        memory_manager.working_context.update_emotion("平静", 0.5)
        
        # 保存
        memory_manager.save_working_context()
        
        # 重新加载
        memory_manager.load_working_context()
        
        assert memory_manager.working_context.user_name == "小明"
    
    def test_get_context_for_llm(self, memory_manager):
        """测试: 获取LLM用的上下文字符串"""
        memory_manager.working_context.set_user_info(name="小明")
        memory_manager.working_context.update_emotion("喜悦", 0.8)
        
        context_str = memory_manager.get_context_for_llm()
        
        assert "小明" in context_str
        assert "喜悦" in context_str

