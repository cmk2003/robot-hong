"""
数据仓库模块
提供业务级别的数据访问接口
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from .database import Database


class Repository:
    """数据仓库 - 封装所有数据库操作"""
    
    def __init__(self, db_path: str):
        """
        初始化仓库
        
        Args:
            db_path: 数据库文件路径
        """
        self.db = Database(db_path)
    
    def init(self) -> None:
        """初始化数据库"""
        self.db.init()
    
    def close(self) -> None:
        """关闭数据库连接"""
        self.db.close()
    
    def _generate_id(self) -> str:
        """生成唯一ID"""
        return str(uuid.uuid4())
    
    # ============ 用户操作 ============
    
    def create_user(self, user_id: str, name: str, profile_data: Dict = None) -> Dict[str, Any]:
        """
        创建用户
        
        Args:
            user_id: 用户ID
            name: 用户名
            profile_data: 用户画像数据
        
        Returns:
            创建的用户信息
        """
        profile_json = json.dumps(profile_data, ensure_ascii=False) if profile_data else None
        
        self.db.execute(
            """INSERT INTO users (id, name, profile_data) VALUES (?, ?, ?)""",
            (user_id, name, profile_json)
        )
        
        return self.get_user(user_id)
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        获取用户信息
        
        Args:
            user_id: 用户ID
        
        Returns:
            用户信息或 None
        """
        user = self.db.query_one(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        )
        
        if user and user.get("profile_data"):
            user["profile_data"] = json.loads(user["profile_data"])
        
        return user
    
    def get_or_create_user(self, user_id: str, name: str) -> Dict[str, Any]:
        """
        获取或创建用户
        
        Args:
            user_id: 用户ID
            name: 用户名
        
        Returns:
            用户信息
        """
        user = self.get_user(user_id)
        if user:
            return user
        return self.create_user(user_id, name)
    
    def update_user_profile(self, user_id: str, profile_data: Dict) -> None:
        """
        更新用户画像
        
        Args:
            user_id: 用户ID
            profile_data: 用户画像数据
        """
        profile_json = json.dumps(profile_data, ensure_ascii=False)
        
        self.db.execute(
            """UPDATE users SET profile_data = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
            (profile_json, user_id)
        )
    
    # ============ 消息操作 ============
    
    def save_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        emotion_type: str = None,
        emotion_intensity: float = None,
        metadata: Dict = None
    ) -> Dict[str, Any]:
        """
        保存消息
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            role: 角色 (user/assistant/system)
            content: 消息内容
            emotion_type: 情感类型
            emotion_intensity: 情感强度 0.0-1.0
            metadata: 扩展数据
        
        Returns:
            保存的消息
        """
        msg_id = self._generate_id()
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        
        self.db.execute(
            """INSERT INTO messages 
            (id, user_id, session_id, role, content, emotion_type, emotion_intensity, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (msg_id, user_id, session_id, role, content, emotion_type, emotion_intensity, metadata_json)
        )
        
        # 更新会话消息计数
        self.db.execute(
            "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
            (session_id,)
        )
        
        return self.db.query_one("SELECT * FROM messages WHERE id = ?", (msg_id,))
    
    def get_recent_messages(
        self,
        user_id: str,
        limit: int = 20,
        session_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        获取最近的消息
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            session_id: 可选，指定会话ID
        
        Returns:
            消息列表（最新的在前）
        """
        if session_id:
            return self.db.query(
                """SELECT * FROM messages 
                WHERE user_id = ? AND session_id = ?
                ORDER BY created_at DESC, rowid DESC LIMIT ?""",
                (user_id, session_id, limit)
            )
        else:
            return self.db.query(
                """SELECT * FROM messages 
                WHERE user_id = ?
                ORDER BY created_at DESC, rowid DESC LIMIT ?""",
                (user_id, limit)
            )
    
    def search_messages(self, user_id: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        全文搜索消息
        
        Args:
            user_id: 用户ID
            query: 搜索关键词
            limit: 返回数量限制
        
        Returns:
            匹配的消息列表
        """
        try:
            # 清理查询字符串，移除 FTS5 特殊字符
            clean_query = query.replace('"', '').replace("'", '').replace('*', '').replace('-', ' ')
            if not clean_query.strip():
                return []
            
            # 使用FTS5搜索
            return self.db.query(
                """SELECT m.* FROM messages m
                JOIN messages_fts fts ON m.rowid = fts.rowid
                WHERE messages_fts MATCH ? AND m.user_id = ?
                ORDER BY rank
                LIMIT ?""",
                (clean_query, user_id, limit)
            )
        except Exception:
            # FTS5 查询失败时，回退到 LIKE 搜索
            return self.db.query(
                """SELECT * FROM messages 
                WHERE user_id = ? AND content LIKE ?
                ORDER BY created_at DESC
                LIMIT ?""",
                (user_id, f"%{query}%", limit)
            )
    
    # ============ 情感记录操作 ============
    
    def save_emotion_record(
        self,
        user_id: str,
        emotion_type: str,
        intensity: float,
        trigger: str = None,
        context: str = None
    ) -> Dict[str, Any]:
        """
        保存情感记录
        
        Args:
            user_id: 用户ID
            emotion_type: 情感类型
            intensity: 情感强度 0.0-1.0
            trigger: 触发因素
            context: 上下文
        
        Returns:
            保存的记录
        """
        record_id = self._generate_id()
        
        self.db.execute(
            """INSERT INTO emotion_records 
            (id, user_id, emotion_type, intensity, trigger, context)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (record_id, user_id, emotion_type, intensity, trigger, context)
        )
        
        return self.db.query_one("SELECT * FROM emotion_records WHERE id = ?", (record_id,))
    
    def get_emotion_history(
        self,
        user_id: str,
        limit: int = 20,
        emotion_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        获取情感历史
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            emotion_type: 可选，过滤特定情感类型
        
        Returns:
            情感记录列表
        """
        if emotion_type:
            return self.db.query(
                """SELECT * FROM emotion_records 
                WHERE user_id = ? AND emotion_type = ?
                ORDER BY created_at DESC LIMIT ?""",
                (user_id, emotion_type, limit)
            )
        else:
            return self.db.query(
                """SELECT * FROM emotion_records 
                WHERE user_id = ?
                ORDER BY created_at DESC LIMIT ?""",
                (user_id, limit)
            )
    
    # ============ 生活事件操作 ============
    
    def save_life_event(
        self,
        user_id: str,
        event_type: str,
        title: str,
        description: str = None,
        importance: int = 3,
        emotion_impact: str = None,
        event_date: str = None,
        embedding: str = None
    ) -> Dict[str, Any]:
        """
        保存生活事件
        
        Args:
            user_id: 用户ID
            event_type: 事件类型 (work/relationship/health/life)
            title: 事件标题
            description: 事件描述
            importance: 重要程度 1-5
            emotion_impact: 情感影响
            event_date: 事件日期
            embedding: 标题的向量嵌入（JSON字符串）
        
        Returns:
            保存的事件
        """
        event_id = self._generate_id()
        
        self.db.execute(
            """INSERT INTO life_events 
            (id, user_id, event_type, title, description, importance, emotion_impact, event_date, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event_id, user_id, event_type, title, description, importance, emotion_impact, event_date, embedding)
        )
        
        return self.db.query_one("SELECT * FROM life_events WHERE id = ?", (event_id,))
    
    def get_life_events(
        self,
        user_id: str,
        limit: int = 20,
        event_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        获取生活事件
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            event_type: 可选，过滤特定事件类型
        
        Returns:
            事件列表
        """
        if event_type:
            return self.db.query(
                """SELECT * FROM life_events 
                WHERE user_id = ? AND event_type = ?
                ORDER BY created_at DESC LIMIT ?""",
                (user_id, event_type, limit)
            )
        else:
            return self.db.query(
                """SELECT * FROM life_events 
                WHERE user_id = ?
                ORDER BY created_at DESC LIMIT ?""",
                (user_id, limit)
            )
    
    # ============ 会话操作 ============
    
    def create_session(self, user_id: str) -> Dict[str, Any]:
        """
        创建新会话
        
        Args:
            user_id: 用户ID
        
        Returns:
            创建的会话
        """
        session_id = self._generate_id()
        
        self.db.execute(
            "INSERT INTO sessions (id, user_id) VALUES (?, ?)",
            (session_id, user_id)
        )
        
        return self.db.query_one("SELECT * FROM sessions WHERE id = ?", (session_id,))
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话信息
        
        Args:
            session_id: 会话ID
        
        Returns:
            会话信息或 None
        """
        return self.db.query_one("SELECT * FROM sessions WHERE id = ?", (session_id,))
    
    def end_session(
        self,
        session_id: str,
        summary: str = None,
        main_topics: List[str] = None,
        emotional_arc: List[Dict] = None
    ) -> None:
        """
        结束会话
        
        Args:
            session_id: 会话ID
            summary: 会话摘要
            main_topics: 主要话题
            emotional_arc: 情感变化轨迹
        """
        topics_json = json.dumps(main_topics, ensure_ascii=False) if main_topics else None
        arc_json = json.dumps(emotional_arc, ensure_ascii=False) if emotional_arc else None
        
        self.db.execute(
            """UPDATE sessions SET 
            ended_at = CURRENT_TIMESTAMP,
            summary = ?,
            main_topics = ?,
            emotional_arc = ?
            WHERE id = ?""",
            (summary, topics_json, arc_json, session_id)
        )
    
    def get_recent_sessions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的会话
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
        
        Returns:
            会话列表
        """
        return self.db.query(
            """SELECT * FROM sessions 
            WHERE user_id = ?
            ORDER BY started_at DESC LIMIT ?""",
            (user_id, limit)
        )
    
    # ============ 工作上下文操作 ============
    
    def save_working_context(self, user_id: str, content: Dict) -> None:
        """
        保存工作上下文（upsert）
        
        Args:
            user_id: 用户ID
            content: 上下文内容
        """
        content_json = json.dumps(content, ensure_ascii=False)
        
        # 先尝试更新
        rows = self.db.execute(
            """UPDATE working_contexts SET 
            content = ?, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?""",
            (content_json, user_id)
        )
        
        # 如果没有更新到行，则插入
        if rows == 0:
            context_id = self._generate_id()
            self.db.execute(
                "INSERT INTO working_contexts (id, user_id, content) VALUES (?, ?, ?)",
                (context_id, user_id, content_json)
            )
    
    def get_working_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        获取工作上下文
        
        Args:
            user_id: 用户ID
        
        Returns:
            上下文内容或 None
        """
        result = self.db.query_one(
            "SELECT content FROM working_contexts WHERE user_id = ?",
            (user_id,)
        )
        
        if result and result.get("content"):
            return json.loads(result["content"])
        return None

