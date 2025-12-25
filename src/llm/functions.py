"""
函数定义模块
定义LLM可调用的工具函数
"""

from typing import List, Dict, Any


# 工具/函数定义列表（OpenAI tools格式）
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "save_emotion",
            "description": "保存用户的情感记录。当识别到用户明确表达情感状态时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "emotion_type": {
                        "type": "string",
                        "description": "情感类型，如：喜悦、悲伤、愤怒、焦虑、恐惧、平静、孤独、失望、感激、希望等"
                    },
                    "intensity": {
                        "type": "number",
                        "description": "情感强度，0.0-1.0之间的数值"
                    },
                    "trigger": {
                        "type": "string",
                        "description": "触发情感的原因或事件"
                    }
                },
                "required": ["emotion_type", "intensity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_life_event",
            "description": "保存用户的重要生活事件。当用户提到重要的工作、关系、健康或生活事件时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_type": {
                        "type": "string",
                        "enum": ["work", "relationship", "health", "life"],
                        "description": "事件类型：work(工作)、relationship(关系)、health(健康)、life(生活)"
                    },
                    "title": {
                        "type": "string",
                        "description": "事件标题，简短描述"
                    },
                    "description": {
                        "type": "string",
                        "description": "事件详细描述"
                    },
                    "importance": {
                        "type": "integer",
                        "description": "重要程度，1-5分，5分最重要"
                    },
                    "emotion_impact": {
                        "type": "string",
                        "description": "此事件对用户情感的影响"
                    }
                },
                "required": ["event_type", "title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_profile",
            "description": "更新用户画像信息。当了解到用户的基本信息、偏好或性格特点时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": ["name", "age", "occupation", "personality", "interests", "communication_style", "sensitive_topics"],
                        "description": "要更新的字段"
                    },
                    "value": {
                        "type": "string",
                        "description": "字段的值"
                    }
                },
                "required": ["field", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "搜索历史记忆。当需要回忆之前的对话内容或用户提到的事件时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["messages", "events", "emotions"],
                        "description": "搜索类型：messages(对话)、events(事件)、emotions(情感记录)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_follow_up",
            "description": "设置待跟进事项。当用户提到需要后续关注的事情时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "待跟进的事项描述"
                    },
                    "remind_after": {
                        "type": "string",
                        "description": "提醒时间，如：next_session（下次对话）、tomorrow、next_week"
                    }
                },
                "required": ["item"]
            }
        }
    },
    # ============ 实时信息工具 ============
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "获取当前的日期和时间。当用户问现在几点、今天几号、星期几时调用。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息。当用户问天气、温度、是否下雨时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海、广州、深圳。默认深圳"
                    }
                },
                "required": []
            }
        }
    }
]


def get_function_by_name(name: str) -> Dict[str, Any]:
    """
    根据名称获取函数定义
    
    Args:
        name: 函数名称
    
    Returns:
        函数定义字典
    """
    for tool in TOOL_DEFINITIONS:
        if tool["function"]["name"] == name:
            return tool["function"]
    return None


def get_function_names() -> List[str]:
    """
    获取所有函数名称
    
    Returns:
        函数名称列表
    """
    return [tool["function"]["name"] for tool in TOOL_DEFINITIONS]

