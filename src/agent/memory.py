"""
记忆管理模块
管理对话历史、情感记录、生活事件等
"""

import uuid
from typing import List, Dict, Any, Optional
from collections import deque

from .context import WorkingContext
from ..storage.repository import Repository
from ..llm.prompts import format_working_context


class MemoryManager:
    """
    记忆管理器 - MemGPT 核心
    管理分层记忆：Working Context + FIFO Queue + Persistent Storage
    """
    
    # 配置
    MAX_QUEUE_SIZE: int = 10  # FIFO队列最大长度（方案B减少，依靠智能搜索补充）
    SUMMARY_TRIGGER: int = 8   # 触发摘要的消息数
    
    def __init__(self, db_path: str, user_id: str):
        """
        初始化记忆管理器
        
        Args:
            db_path: 数据库路径
            user_id: 用户ID
        """
        self.db_path = db_path
        self.user_id = user_id
        self.repository = Repository(db_path)
        
        # 工作上下文（RAM）
        self.working_context = WorkingContext()
        
        # FIFO消息队列
        self._queue: deque = deque(maxlen=self.MAX_QUEUE_SIZE)
        
        # 当前会话ID
        self.session_id: Optional[str] = None
    
    def init(self) -> None:
        """初始化（创建表、加载上下文）"""
        self.repository.init()
        
        # 确保用户存在
        self.repository.get_or_create_user(self.user_id, "默认用户")
        
        # 创建新会话
        session = self.repository.create_session(self.user_id)
        self.session_id = session["id"]
        
        # 加载工作上下文
        self.load_working_context()
        
        # 加载最近的对话历史到队列（实现跨会话记忆）
        self._load_recent_history()
    
    def close(self) -> None:
        """关闭（保存上下文、结束会话）"""
        self.save_working_context()
        if self.session_id:
            self.repository.end_session(self.session_id)
        self.repository.close()
    
    def _load_recent_history(self) -> None:
        """
        加载最近的对话历史到队列
        实现跨会话的记忆连续性
        """
        # 从数据库获取最近的消息（按时间倒序）
        recent_messages = self.repository.get_recent_messages(
            self.user_id, 
            limit=self.MAX_QUEUE_SIZE
        )
        
        # 倒序添加到队列（让最新的在最后）
        for msg in reversed(recent_messages):
            self._queue.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # ============ 消息队列管理 ============
    
    def add_to_queue(self, message: Dict[str, Any]) -> None:
        """
        添加消息到队列
        
        Args:
            message: 消息字典 {"role": str, "content": str, ...}
        """
        self._queue.append(message)
    
    def get_queue(self) -> List[Dict[str, Any]]:
        """
        获取队列中的所有消息
        
        Returns:
            消息列表
        """
        return list(self._queue)
    
    def clear_queue(self) -> None:
        """清空队列"""
        self._queue.clear()
    
    def should_summarize(self) -> bool:
        """
        是否应该触发摘要
        
        Returns:
            是否需要摘要
        """
        return len(self._queue) >= self.SUMMARY_TRIGGER
    
    # ============ 消息持久化 ============
    
    def save_message(
        self,
        role: str,
        content: str,
        emotion_type: str = None,
        emotion_intensity: float = None
    ) -> Dict[str, Any]:
        """
        保存消息（同时添加到队列和持久化存储）
        
        Args:
            role: 角色 (user/assistant/system)
            content: 消息内容
            emotion_type: 情感类型
            emotion_intensity: 情感强度
        
        Returns:
            保存的消息
        """
        # 添加到队列
        queue_msg = {
            "role": role,
            "content": content
        }
        self.add_to_queue(queue_msg)
        
        # 持久化到数据库
        msg = self.repository.save_message(
            user_id=self.user_id,
            session_id=self.session_id,
            role=role,
            content=content,
            emotion_type=emotion_type,
            emotion_intensity=emotion_intensity
        )
        
        # 更新交互计数
        if role == "user":
            self.working_context.increment_interaction()
        
        return msg
    
    def get_recent_messages(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取最近的消息
        
        Args:
            limit: 返回数量
        
        Returns:
            消息列表
        """
        return self.repository.get_recent_messages(self.user_id, limit=limit)
    
    def search_messages(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        搜索消息
        
        Args:
            query: 搜索关键词
            limit: 返回数量
        
        Returns:
            匹配的消息列表
        """
        return self.repository.search_messages(self.user_id, query, limit=limit)
    
    def search_relevant_context(self, user_message: str) -> str:
        """
        智能搜索相关历史上下文
        根据用户消息提取关键词，搜索历史记录
        
        Args:
            user_message: 用户当前消息
        
        Returns:
            格式化的相关历史信息字符串
        """
        # 提取搜索关键词（简单分词，取主要词汇）
        keywords = self._extract_keywords(user_message)
        
        if not keywords:
            return ""
        
        parts = []
        
        # 搜索相关消息
        for keyword in keywords[:3]:  # 最多用3个关键词搜索
            messages = self.search_messages(keyword, limit=3)
            if messages:
                for msg in messages:
                    # 避免重复添加最近的消息
                    if msg["content"] not in [m.get("content") for m in self._queue]:
                        parts.append(f"[历史对话] {msg['role']}: {msg['content'][:100]}")
        
        # 搜索相关事件
        events = self.get_life_events(limit=10)
        for keyword in keywords[:3]:
            for event in events:
                title = event.get("title") or ""
                description = event.get("description") or ""
                if keyword in title or keyword in description:
                    parts.append(f"[历史事件] {title}")
                    break
        
        # 去重并限制数量
        unique_parts = list(dict.fromkeys(parts))[:5]
        
        if unique_parts:
            return "\n📚 **相关历史记忆**:\n" + "\n".join(unique_parts)
        return ""
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        从文本中提取关键词
        简单实现：过滤停用词，提取有意义的词
        
        Args:
            text: 输入文本
        
        Returns:
            关键词列表
        """
        # 中文停用词
        stop_words = {
            "的", "了", "是", "我", "你", "他", "她", "它", "们", "这", "那",
            "吗", "呢", "吧", "啊", "哦", "呀", "嗯", "好", "在", "有", "和",
            "也", "都", "就", "不", "很", "到", "说", "要", "会", "去", "能",
            "还", "可以", "一个", "什么", "怎么", "为什么", "哪", "谁", "最近"
        }
        
        # 简单分词（按标点和空格分割）
        import re
        words = re.split(r'[，。！？、\s]+', text)
        
        # 过滤停用词和短词
        keywords = [
            w.strip() for w in words 
            if w.strip() and len(w.strip()) >= 2 and w.strip() not in stop_words
        ]
        
        return keywords
    
    # ============ 情感记录 ============
    
    def save_emotion(
        self,
        emotion_type: str,
        intensity: float,
        trigger: str = None,
        context: str = None
    ) -> Dict[str, Any]:
        """
        保存情感记录
        
        Args:
            emotion_type: 情感类型
            intensity: 情感强度
            trigger: 触发因素
            context: 上下文
        
        Returns:
            保存的记录
        """
        # 更新工作上下文
        self.working_context.update_emotion(emotion_type, intensity)
        
        # 持久化
        return self.repository.save_emotion_record(
            user_id=self.user_id,
            emotion_type=emotion_type,
            intensity=intensity,
            trigger=trigger,
            context=context
        )
    
    def get_emotion_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取情感历史
        
        Args:
            limit: 返回数量
        
        Returns:
            情感记录列表
        """
        return self.repository.get_emotion_history(self.user_id, limit=limit)
    
    # ============ 生活事件 ============
    
    def save_life_event(
        self,
        event_type: str,
        title: str,
        description: str = None,
        importance: int = 3,
        emotion_impact: str = None
    ) -> Dict[str, Any]:
        """
        保存生活事件
        
        Args:
            event_type: 事件类型
            title: 事件标题
            description: 事件描述
            importance: 重要程度
            emotion_impact: 情感影响
        
        Returns:
            保存的事件
        """
        # 更新工作上下文
        self.working_context.add_recent_event(title)
        
        # 持久化
        return self.repository.save_life_event(
            user_id=self.user_id,
            event_type=event_type,
            title=title,
            description=description,
            importance=importance,
            emotion_impact=emotion_impact
        )
    
    def get_life_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取生活事件
        
        Args:
            limit: 返回数量
        
        Returns:
            事件列表
        """
        return self.repository.get_life_events(self.user_id, limit=limit)
    
    # ============ 工作上下文持久化 ============
    
    def save_working_context(self) -> None:
        """保存工作上下文到数据库"""
        self.repository.save_working_context(
            self.user_id,
            self.working_context.to_dict()
        )
    
    def load_working_context(self) -> None:
        """从数据库加载工作上下文"""
        # 1. 加载 working_contexts 表
        data = self.repository.get_working_context(self.user_id)
        if data:
            self.working_context.load_from_dict(data)
        
        # 2. 如果用户名为空，尝试从 users 表的 profile_data 加载
        if not self.working_context.user_name:
            user = self.repository.get_user(self.user_id)
            if user and user.get("profile_data"):
                profile = user["profile_data"]
                if profile.get("name"):
                    self.working_context.set_user_info(name=profile["name"])
                # 加载其他 profile 信息
                for key in ["age", "occupation", "personality", "interests"]:
                    if profile.get(key):
                        self.working_context.user_info[key] = profile[key]
    
    # ============ LLM 接口 ============
    
    def get_context_for_llm(self) -> str:
        """
        获取格式化的上下文字符串，用于LLM
        包含用户画像和近期事件
        
        Returns:
            格式化的上下文
        """
        parts = []
        
        # 1. 基础上下文（用户名、情感等）
        base_context = self.working_context.format_for_llm()
        if base_context:
            parts.append(base_context)
        
        # 2. 用户画像详情
        user_info = self.working_context.user_info
        if user_info:
            info_parts = []
            if user_info.get('location'):
                info_parts.append(f"住在{user_info['location']}")
            if user_info.get('birthday'):
                info_parts.append(f"生日{user_info['birthday']}")
            if user_info.get('occupation') and user_info.get('occupation') != '未提及':
                info_parts.append(f"职业是{user_info['occupation']}")
            if info_parts:
                parts.append(f"**用户信息**：{', '.join(info_parts)}")
        
        # 3. 近期事件（让模型可以主动关心）
        recent_events = self.get_life_events(limit=5)
        if recent_events:
            event_strs = []
            for e in recent_events[:3]:  # 最多显示3个
                event_strs.append(f"{e.get('title', '未知事件')}")
            parts.append(f"**用户近期经历**：{', '.join(event_strs)}（可以适时关心）")
        
        return "\n".join(parts) if parts else ""
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """
        获取用于LLM的消息列表
        
        Returns:
            消息列表（仅包含role和content）
        """
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.get_queue()
        ]
    
    # ============ 用户画像 ============
    
    def update_user_profile(self, field: str, value: str) -> None:
        """
        更新用户画像
        
        Args:
            field: 字段名
            value: 字段值
        """
        if field == "name":
            self.working_context.set_user_info(name=value)
        elif field in ["age", "birthday", "location", "occupation", "personality", 
                       "interests", "communication_style", "sensitive_topics"]:
            self.working_context.user_info[field] = value
        
        # 保存到 users 表
        self.repository.update_user_profile(
            self.user_id,
            self.working_context.user_info
        )
        
        # 同时保存到 working_contexts 表（确保重启后能加载）
        self.save_working_context()

