"""
LangChain 工具定义模块
将现有工具函数包装为 LangChain Tool
"""

from typing import Optional
from langchain_core.tools import tool

from ..tools.realtime import get_current_datetime as _get_datetime, get_weather as _get_weather


# ============ 记忆相关工具（需要 MemoryManager 实例）============

def create_memory_tools(memory_manager):
    """
    创建依赖 MemoryManager 的工具
    
    Args:
        memory_manager: MemoryManager 实例
    
    Returns:
        工具列表
    """
    
    @tool
    def save_emotion(emotion_type: str, intensity: float, trigger: Optional[str] = None) -> dict:
        """保存用户的情感记录。当识别到用户明确表达情感状态时调用。
        
        Args:
            emotion_type: 情感类型，如：喜悦、悲伤、愤怒、焦虑等
            intensity: 情感强度，0.0-1.0之间的数值
            trigger: 触发情感的原因或事件
        """
        memory_manager.save_emotion(
            emotion_type=emotion_type,
            intensity=intensity,
            trigger=trigger
        )
        return {"success": True, "message": "情感记录已保存"}
    
    @tool
    def save_life_event(
        event_type: str, 
        title: str, 
        description: Optional[str] = None,
        importance: int = 3,
        emotion_impact: Optional[str] = None
    ) -> dict:
        """保存用户的重要生活事件。当用户提到重要的工作、关系、健康或生活事件时调用。
        
        Args:
            event_type: 事件类型：work(工作)、relationship(关系)、health(健康)、life(生活)
            title: 事件标题，简短描述
            description: 事件详细描述
            importance: 重要程度，1-5分，5分最重要
            emotion_impact: 此事件对用户情感的影响
        """
        memory_manager.save_life_event(
            event_type=event_type,
            title=title,
            description=description,
            importance=importance,
            emotion_impact=emotion_impact
        )
        return {"success": True, "message": "生活事件已保存"}
    
    @tool
    def update_user_profile(field: str, value: str) -> dict:
        """更新用户画像信息。当用户告诉你任何个人信息时必须调用此函数保存。
        
        Args:
            field: 要更新的字段：name, age, birthday, location, occupation, personality, interests, communication_style, sensitive_topics
            value: 字段的值
        """
        memory_manager.update_user_profile(field=field, value=value)
        return {"success": True, "message": "用户画像已更新"}
    
    @tool
    def search_memory(query: str, search_type: str = "messages") -> dict:
        """搜索历史记忆。当需要回忆之前的对话内容或用户提到的事件时调用。
        
        Args:
            query: 搜索关键词
            search_type: 搜索类型：messages(对话)、events(事件)、emotions(情感记录)
        """
        if search_type == "messages":
            results = memory_manager.search_messages(query)
        elif search_type == "events":
            results = memory_manager.get_life_events()
            results = [e for e in results if query.lower() in e.get("title", "").lower()]
        elif search_type == "emotions":
            results = memory_manager.get_emotion_history()
            results = [e for e in results if query.lower() in e.get("emotion_type", "").lower()]
        else:
            results = []
        
        return {"success": True, "results": results[:5]}
    
    @tool
    def set_follow_up(item: str) -> dict:
        """设置待跟进事项。当用户提到需要后续关注的事情时调用。
        
        Args:
            item: 待跟进的事项描述
        """
        memory_manager.working_context.add_follow_up(item)
        return {"success": True, "message": "待跟进事项已添加"}
    
    return [save_emotion, save_life_event, update_user_profile, search_memory, set_follow_up]


# ============ 实时信息工具（无状态）============

@tool
def get_current_datetime() -> dict:
    """获取当前的日期和时间。当用户问现在几点、今天几号、星期几时调用。"""
    return _get_datetime()


@tool
def get_weather(city: str = "深圳") -> dict:
    """获取指定城市的天气信息。当用户问天气、温度、是否下雨时调用。
    
    Args:
        city: 城市名称，如：北京、上海、广州、深圳。默认深圳
    """
    return _get_weather(city)


# 无状态工具列表
STATELESS_TOOLS = [get_current_datetime, get_weather]

