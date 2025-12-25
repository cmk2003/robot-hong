"""
Repository模块测试
TDD: 先写测试，测试应该失败
"""

import pytest
import os
import tempfile
from datetime import datetime


class TestRepository:
    """数据仓库测试"""
    
    @pytest.fixture
    def temp_db_path(self):
        """创建临时数据库文件路径"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        if os.path.exists(f.name):
            os.remove(f.name)
    
    @pytest.fixture
    def repo(self, temp_db_path):
        """创建Repository实例"""
        from src.storage.repository import Repository
        repo = Repository(temp_db_path)
        repo.init()
        yield repo
        repo.close()
    
    # ============ 用户相关测试 ============
    
    def test_create_user(self, repo):
        """测试: 创建用户"""
        user = repo.create_user("test-user-1", "小明")
        
        assert user["id"] == "test-user-1"
        assert user["name"] == "小明"
    
    def test_get_user(self, repo):
        """测试: 获取用户"""
        repo.create_user("test-user-1", "小明")
        user = repo.get_user("test-user-1")
        
        assert user is not None
        assert user["name"] == "小明"
    
    def test_get_or_create_user(self, repo):
        """测试: 获取或创建用户"""
        # 第一次创建
        user1 = repo.get_or_create_user("test-user-1", "小明")
        # 第二次获取
        user2 = repo.get_or_create_user("test-user-1", "小明")
        
        assert user1["id"] == user2["id"]
    
    def test_update_user_profile(self, repo):
        """测试: 更新用户画像"""
        repo.create_user("test-user-1", "小明")
        
        profile = {"age": 25, "occupation": "工程师"}
        repo.update_user_profile("test-user-1", profile)
        
        user = repo.get_user("test-user-1")
        assert user["profile_data"] is not None
    
    # ============ 消息相关测试 ============
    
    def test_save_message(self, repo):
        """测试: 保存消息"""
        repo.create_user("test-user-1", "小明")
        
        msg = repo.save_message(
            user_id="test-user-1",
            session_id="session-1",
            role="user",
            content="你好，我今天心情不太好",
            emotion_type="悲伤",
            emotion_intensity=0.6
        )
        
        assert msg["id"] is not None
        assert msg["content"] == "你好，我今天心情不太好"
        assert msg["emotion_type"] == "悲伤"
    
    def test_get_recent_messages(self, repo):
        """测试: 获取最近消息"""
        repo.create_user("test-user-1", "小明")
        
        # 保存多条消息
        for i in range(5):
            repo.save_message(
                user_id="test-user-1",
                session_id="session-1",
                role="user",
                content=f"消息{i}"
            )
        
        # 获取最近3条
        messages = repo.get_recent_messages("test-user-1", limit=3)
        
        assert len(messages) == 3
        assert messages[0]["content"] == "消息4"  # 最新的在前
    
    def test_search_messages(self, repo):
        """测试: 搜索消息"""
        repo.create_user("test-user-1", "小明")
        
        repo.save_message(
            user_id="test-user-1",
            session_id="session-1",
            role="user",
            content="Today work is very hard pressure"
        )
        repo.save_message(
            user_id="test-user-1",
            session_id="session-1",
            role="user",
            content="Weekend hiking relaxation"
        )
        
        # 搜索包含"work"的消息（使用英文确保FTS分词正常）
        results = repo.search_messages("test-user-1", "work")
        
        assert len(results) >= 1
        assert "work" in results[0]["content"]
    
    # ============ 情感记录测试 ============
    
    def test_save_emotion_record(self, repo):
        """测试: 保存情感记录"""
        repo.create_user("test-user-1", "小明")
        
        record = repo.save_emotion_record(
            user_id="test-user-1",
            emotion_type="焦虑",
            intensity=0.7,
            trigger="工作压力",
            context="项目deadline临近"
        )
        
        assert record["emotion_type"] == "焦虑"
        assert record["intensity"] == 0.7
    
    def test_get_emotion_history(self, repo):
        """测试: 获取情感历史"""
        repo.create_user("test-user-1", "小明")
        
        repo.save_emotion_record("test-user-1", "喜悦", 0.8)
        repo.save_emotion_record("test-user-1", "悲伤", 0.5)
        
        history = repo.get_emotion_history("test-user-1", limit=10)
        
        assert len(history) == 2
    
    # ============ 生活事件测试 ============
    
    def test_save_life_event(self, repo):
        """测试: 保存生活事件"""
        repo.create_user("test-user-1", "小明")
        
        event = repo.save_life_event(
            user_id="test-user-1",
            event_type="work",
            title="升职加薪",
            description="终于升职了，薪资涨了30%",
            importance=5,
            emotion_impact="喜悦"
        )
        
        assert event["title"] == "升职加薪"
        assert event["importance"] == 5
    
    def test_get_life_events(self, repo):
        """测试: 获取生活事件"""
        repo.create_user("test-user-1", "小明")
        
        repo.save_life_event("test-user-1", "work", "项目上线")
        repo.save_life_event("test-user-1", "relationship", "和朋友聚会")
        
        events = repo.get_life_events("test-user-1")
        
        assert len(events) == 2
    
    # ============ 会话相关测试 ============
    
    def test_create_session(self, repo):
        """测试: 创建会话"""
        repo.create_user("test-user-1", "小明")
        
        session = repo.create_session("test-user-1")
        
        assert session["id"] is not None
        assert session["user_id"] == "test-user-1"
    
    def test_end_session(self, repo):
        """测试: 结束会话"""
        repo.create_user("test-user-1", "小明")
        session = repo.create_session("test-user-1")
        
        repo.end_session(session["id"], summary="聊了工作压力问题")
        
        ended = repo.get_session(session["id"])
        assert ended["ended_at"] is not None
        assert ended["summary"] == "聊了工作压力问题"
    
    # ============ 工作上下文测试 ============
    
    def test_save_working_context(self, repo):
        """测试: 保存工作上下文"""
        repo.create_user("test-user-1", "小明")
        
        context = {
            "current_emotion": "平静",
            "recent_topics": ["工作", "健康"],
            "user_preferences": {"communication_style": "温和"}
        }
        
        repo.save_working_context("test-user-1", context)
        
        loaded = repo.get_working_context("test-user-1")
        assert loaded is not None
        assert loaded["current_emotion"] == "平静"
    
    def test_update_working_context(self, repo):
        """测试: 更新工作上下文"""
        repo.create_user("test-user-1", "小明")
        
        # 第一次保存
        repo.save_working_context("test-user-1", {"emotion": "平静"})
        
        # 更新
        repo.save_working_context("test-user-1", {"emotion": "焦虑"})
        
        loaded = repo.get_working_context("test-user-1")
        assert loaded["emotion"] == "焦虑"

