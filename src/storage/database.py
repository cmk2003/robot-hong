"""
数据库模块 - SQLite + FTS5
提供数据库连接和基础操作
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager


class Database:
    """SQLite数据库管理类"""
    
    def __init__(self, db_path: str):
        """
        初始化数据库
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
    
    def __enter__(self) -> "Database":
        """上下文管理器入口"""
        self._connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器退出"""
        self.close()
    
    def _connect(self) -> None:
        """建立数据库连接"""
        if self.conn is None:
            # 确保目录存在
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # check_same_thread=False 允许跨线程使用（Gradio多线程需要）
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # 返回字典形式的结果
            # 启用外键约束
            self.conn.execute("PRAGMA foreign_keys = ON")
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def init(self) -> None:
        """初始化数据库，创建所有表"""
        self._connect()
        self._create_tables()
        self._create_fts_table()
        self._create_indexes()
    
    def _create_tables(self) -> None:
        """创建数据库表"""
        cursor = self.conn.cursor()
        
        # 用户表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                profile_data JSON
            )
        """)
        
        # 对话消息表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                emotion_type TEXT,
                emotion_intensity REAL,
                metadata JSON,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # 情感记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emotion_records (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                emotion_type TEXT NOT NULL,
                intensity REAL NOT NULL,
                trigger TEXT,
                context TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # 生活事件表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS life_events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                importance INTEGER DEFAULT 3,
                emotion_impact TEXT,
                event_date DATE,
                embedding TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # 尝试添加 embedding 列（如果表已存在但没有该列）
        try:
            cursor.execute("ALTER TABLE life_events ADD COLUMN embedding TEXT")
        except sqlite3.OperationalError:
            pass  # 列已存在
        
        # 工作上下文表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS working_contexts (
                id TEXT PRIMARY KEY,
                user_id TEXT UNIQUE NOT NULL,
                content JSON NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # 会话表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                ended_at DATETIME,
                summary TEXT,
                main_topics JSON,
                emotional_arc JSON,
                message_count INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        self.conn.commit()
    
    def _create_fts_table(self) -> None:
        """创建FTS5全文搜索虚拟表"""
        cursor = self.conn.cursor()
        
        # 检查表是否存在
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='messages_fts'
        """)
        
        if not cursor.fetchone():
            cursor.execute("""
                CREATE VIRTUAL TABLE messages_fts USING fts5(
                    content,
                    emotion_type,
                    content=messages,
                    content_rowid=rowid,
                    tokenize='unicode61'
                )
            """)
            
            # 创建触发器以保持FTS表同步
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(rowid, content, emotion_type) 
                    VALUES (NEW.rowid, NEW.content, NEW.emotion_type);
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content, emotion_type) 
                    VALUES ('delete', OLD.rowid, OLD.content, OLD.emotion_type);
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content, emotion_type) 
                    VALUES ('delete', OLD.rowid, OLD.content, OLD.emotion_type);
                    INSERT INTO messages_fts(rowid, content, emotion_type) 
                    VALUES (NEW.rowid, NEW.content, NEW.emotion_type);
                END
            """)
        
        self.conn.commit()
    
    def _create_indexes(self) -> None:
        """创建索引"""
        cursor = self.conn.cursor()
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_messages_user_time ON messages(user_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_emotion ON messages(emotion_type)",
            "CREATE INDEX IF NOT EXISTS idx_emotion_user_time ON emotion_records(user_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_emotion_type ON emotion_records(emotion_type)",
            "CREATE INDEX IF NOT EXISTS idx_events_user_date ON life_events(user_id, event_date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_events_type ON life_events(event_type)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_time ON sessions(user_id, started_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_users_name ON users(name)",
        ]
        
        for idx in indexes:
            cursor.execute(idx)
        
        self.conn.commit()
    
    def get_tables(self) -> List[str]:
        """获取所有表名"""
        self._connect()
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type IN ('table', 'view') 
            AND name NOT LIKE 'sqlite_%'
        """)
        return [row[0] for row in cursor.fetchall()]
    
    def execute(self, sql: str, params: Tuple = ()) -> int:
        """
        执行SQL语句
        
        Args:
            sql: SQL语句
            params: 参数元组
        
        Returns:
            受影响的行数
        """
        self._connect()
        cursor = self.conn.cursor()
        cursor.execute(sql, params)
        self.conn.commit()
        return cursor.rowcount
    
    def execute_many(self, sql: str, params_list: List[Tuple]) -> int:
        """
        批量执行SQL语句
        
        Args:
            sql: SQL语句
            params_list: 参数列表
        
        Returns:
            受影响的行数
        """
        self._connect()
        cursor = self.conn.cursor()
        cursor.executemany(sql, params_list)
        self.conn.commit()
        return cursor.rowcount
    
    def query(self, sql: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """
        执行查询并返回结果
        
        Args:
            sql: SQL查询语句
            params: 参数元组
        
        Returns:
            查询结果列表（字典形式）
        """
        self._connect()
        cursor = self.conn.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def query_one(self, sql: str, params: Tuple = ()) -> Optional[Dict[str, Any]]:
        """
        执行查询并返回单条结果
        
        Args:
            sql: SQL查询语句
            params: 参数元组
        
        Returns:
            单条结果（字典形式）或 None
        """
        results = self.query(sql, params)
        return results[0] if results else None
    
    def search_fts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        全文搜索消息
        
        Args:
            query: 搜索关键词
            limit: 返回数量限制
        
        Returns:
            匹配的消息列表
        """
        self._connect()
        sql = """
            SELECT m.* FROM messages m
            JOIN messages_fts fts ON m.rowid = fts.rowid
            WHERE messages_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """
        return self.query(sql, (query, limit))

