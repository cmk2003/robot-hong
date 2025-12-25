"""
工作上下文模块
管理当前会话的上下文信息
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class WorkingContext:
    """
    工作上下文 - 存储当前会话的关键信息
    类比 MemGPT 的 Working Context 区域
    """
    
    # 用户信息
    user_name: Optional[str] = None
    user_info: Dict[str, Any] = field(default_factory=dict)
    
    # 情感状态
    current_emotion: Optional[str] = None
    emotion_intensity: float = 0.0
    emotion_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # 最近事件
    recent_events: List[str] = field(default_factory=list)
    
    # 用户偏好
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    # 待跟进事项
    follow_ups: List[str] = field(default_factory=list)
    
    # 关系进展
    trust_level: float = 0.5  # 信任度 0.0-1.0
    interaction_count: int = 0
    
    # 配置
    MAX_RECENT_EVENTS: int = 10
    MAX_EMOTION_HISTORY: int = 20
    MAX_FOLLOW_UPS: int = 5
    
    def set_user_info(
        self,
        name: str = None,
        age: int = None,
        occupation: str = None,
        **kwargs
    ) -> None:
        """
        设置用户信息
        
        Args:
            name: 用户名
            age: 年龄
            occupation: 职业
            **kwargs: 其他信息
        """
        if name:
            self.user_name = name
            self.user_info["name"] = name
        if age:
            self.user_info["age"] = age
        if occupation:
            self.user_info["occupation"] = occupation
        
        for key, value in kwargs.items():
            self.user_info[key] = value
    
    def update_emotion(self, emotion_type: str, intensity: float) -> None:
        """
        更新当前情感状态
        
        Args:
            emotion_type: 情感类型
            intensity: 情感强度 0.0-1.0
        """
        # 保存到历史
        if self.current_emotion:
            self.emotion_history.append({
                "type": self.current_emotion,
                "intensity": self.emotion_intensity,
                "timestamp": datetime.now().isoformat()
            })
            
            # 限制历史长度
            if len(self.emotion_history) > self.MAX_EMOTION_HISTORY:
                self.emotion_history = self.emotion_history[-self.MAX_EMOTION_HISTORY:]
        
        # 更新当前状态
        self.current_emotion = emotion_type
        self.emotion_intensity = intensity
    
    def add_recent_event(self, event: str) -> None:
        """
        添加最近事件
        
        Args:
            event: 事件描述
        """
        if event not in self.recent_events:
            self.recent_events.append(event)
        
        # 限制数量
        if len(self.recent_events) > self.MAX_RECENT_EVENTS:
            self.recent_events = self.recent_events[-self.MAX_RECENT_EVENTS:]
    
    def add_follow_up(self, item: str) -> None:
        """
        添加待跟进事项
        
        Args:
            item: 待跟进事项
        """
        if item not in self.follow_ups:
            self.follow_ups.append(item)
        
        # 限制数量
        if len(self.follow_ups) > self.MAX_FOLLOW_UPS:
            self.follow_ups = self.follow_ups[-self.MAX_FOLLOW_UPS:]
    
    def remove_follow_up(self, item: str) -> None:
        """
        移除待跟进事项
        
        Args:
            item: 待跟进事项
        """
        if item in self.follow_ups:
            self.follow_ups.remove(item)
    
    def update_preference(self, key: str, value: Any) -> None:
        """
        更新用户偏好
        
        Args:
            key: 偏好键
            value: 偏好值
        """
        self.preferences[key] = value
    
    def increment_interaction(self) -> None:
        """增加交互计数"""
        self.interaction_count += 1
        # 随着交互增加信任度
        if self.trust_level < 0.95:
            self.trust_level = min(0.95, self.trust_level + 0.01)
    
    def clear_emotion_history(self) -> None:
        """清除情感历史"""
        self.emotion_history = []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            上下文字典
        """
        return {
            "user_name": self.user_name,
            "user_info": self.user_info,
            "current_emotion": self.current_emotion,
            "emotion_intensity": self.emotion_intensity,
            "emotion_history": self.emotion_history,
            "recent_events": self.recent_events,
            "preferences": self.preferences,
            "follow_ups": self.follow_ups,
            "trust_level": self.trust_level,
            "interaction_count": self.interaction_count
        }
    
    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """
        从字典加载
        
        Args:
            data: 上下文字典
        """
        self.user_name = data.get("user_name")
        self.user_info = data.get("user_info", {})
        self.current_emotion = data.get("current_emotion")
        self.emotion_intensity = data.get("emotion_intensity", 0.0)
        self.emotion_history = data.get("emotion_history", [])
        self.recent_events = data.get("recent_events", [])
        self.preferences = data.get("preferences", {})
        self.follow_ups = data.get("follow_ups", [])
        self.trust_level = data.get("trust_level", 0.5)
        self.interaction_count = data.get("interaction_count", 0)
    
    def format_for_llm(self) -> str:
        """
        格式化为LLM可读的字符串
        只传递核心信息，避免每次都提及历史事件导致对话不自然
        
        Returns:
            格式化的上下文字符串
        """
        parts = []
        
        # 核心信息：用户名
        if self.user_name:
            parts.append(f"**用户名称**：{self.user_name}")
        
        # 核心信息：当前情感状态（不包括历史）
        if self.current_emotion:
            intensity_desc = "轻微" if self.emotion_intensity < 0.4 else "中等" if self.emotion_intensity < 0.7 else "强烈"
            parts.append(f"**当前情感**：{self.current_emotion}（{intensity_desc}）")
        
        # 核心信息：待跟进事项（这些确实需要主动关注）
        if self.follow_ups:
            parts.append(f"**待跟进**：{', '.join(self.follow_ups)}")
        
        # 不再自动传递以下信息，让 LLM 通过 search_memory 按需查询：
        # - emotion_history（情感变化历史）
        # - recent_events（近期事件）
        # - preferences（用户偏好）
        # - trust_level / interaction_count
        
        return "\n".join(parts) if parts else ""

