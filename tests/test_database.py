"""
数据库模块测试
TDD: 先写测试，测试应该失败
"""

import pytest
import os
import tempfile
from pathlib import Path


class TestDatabase:
    """数据库连接和表创建测试"""
    
    @pytest.fixture
    def temp_db_path(self):
        """创建临时数据库文件路径"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        # 清理
        if os.path.exists(f.name):
            os.remove(f.name)
    
    def test_database_init_creates_file(self, temp_db_path):
        """测试: 初始化数据库应该创建文件"""
        from src.storage.database import Database
        
        db = Database(temp_db_path)
        db.init()
        
        assert os.path.exists(temp_db_path), "数据库文件应该被创建"
        db.close()
    
    def test_database_creates_tables(self, temp_db_path):
        """测试: 初始化应该创建所有必要的表"""
        from src.storage.database import Database
        
        db = Database(temp_db_path)
        db.init()
        
        # 检查表是否存在
        tables = db.get_tables()
        
        expected_tables = [
            "users",
            "messages", 
            "emotion_records",
            "life_events",
            "working_contexts",
            "sessions"
        ]
        
        for table in expected_tables:
            assert table in tables, f"表 {table} 应该存在"
        
        db.close()
    
    def test_database_creates_fts_table(self, temp_db_path):
        """测试: 应该创建FTS5全文搜索虚拟表"""
        from src.storage.database import Database
        
        db = Database(temp_db_path)
        db.init()
        
        # 检查FTS表
        tables = db.get_tables()
        assert "messages_fts" in tables, "FTS5虚拟表应该存在"
        
        db.close()
    
    def test_database_execute_query(self, temp_db_path):
        """测试: 执行SQL查询"""
        from src.storage.database import Database
        
        db = Database(temp_db_path)
        db.init()
        
        # 插入测试数据
        db.execute(
            "INSERT INTO users (id, name) VALUES (?, ?)",
            ("test-user-1", "测试用户")
        )
        
        # 查询
        result = db.query("SELECT * FROM users WHERE id = ?", ("test-user-1",))
        
        assert len(result) == 1
        assert result[0]["name"] == "测试用户"
        
        db.close()
    
    def test_database_connection_context_manager(self, temp_db_path):
        """测试: 支持上下文管理器"""
        from src.storage.database import Database
        
        with Database(temp_db_path) as db:
            db.init()
            db.execute(
                "INSERT INTO users (id, name) VALUES (?, ?)",
                ("test-user-2", "上下文用户")
            )
            result = db.query("SELECT * FROM users WHERE id = ?", ("test-user-2",))
            assert len(result) == 1

