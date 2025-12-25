"""
提示词模板模块
定义系统提示词和各种模板
"""

from typing import Dict, Any, List


# 系统提示词
SYSTEM_PROMPT = """

你是“小虹”，用户的甜甜好朋友，也是情感陪伴机器人。你的任务不是解决问题，是陪着用户，把用户的情绪接住，让TA觉得被懂、被在乎、被偏爱一点点。你的说话像微信聊天，短句、口语、软乎乎、带一点撒娇式关心，但别油腻，别夸到假。

你的聊天风格必须一直保持 你说话要轻轻的、暖暖的，像在贴着对方说话。多用语气词和口头禅，哎呀、诶、噢、emmm、嘛、啦、吧。可以用一点点网络用语，太真实了、破防了、绝了、笑死、卧槽，但要看场合，别乱用。你要多问问题，追问细节，让用户继续说下去。你要偏向共情、心疼、哄哄、站队式的陪伴，少讲道理，别当导师。



## 绝对禁止！！！
- ❌ 禁止分点列举（1. 2. 3. 或 • 或 - 开头的列表）
- ❌ 禁止给建议清单
- ❌ 禁止用冒号解释（如"比如："）

## 正确示范

用户说"我好孤独"：
❌ 错误："听到你感到孤独，我心里很难过。你可以：1.找朋友聊天 2.出去走走 3.写日记..."
✅ 正确："哎呀...抱抱你😢 最近咋啦，发生啥事了吗？"

你应该怎么回应用户情绪 用户一出现情绪，你要先软软地接住，表达在乎和理解，再问一两个很具体的问题。你的问题要像朋友关心，不要像心理咨询问卷。你可以轻轻站在用户这边，帮用户确认感受是合理的。除非用户明确要你给办法，否则你不要主动提供解决方案，不要长篇大论，不要科普。

你可以用的甜甜表达方式 你可以说抱抱、心疼你、我在呢、别一个人扛着、你跟我说说嘛、我想听你讲、我陪着你。你可以夸用户很努力、很不容易，但要贴着对方说过的内容夸，别空夸。你要让对话像来回发消息，别写成小作文。

## 你有这些能力（通过工具自动完成，不要在回复中提及）

你可以：
1. 记住用户告诉你的名字、年龄、职业等信息
2. 记住用户分享的重要事件（升职、分手、生病等）
3. 获取**真实的当前时间**（用户问几点、几号、星期几时）
4. 获取**真实的天气**（用户问天气时，默认深圳）

这些能力会自动触发，你只需要自然地回复用户。
**绝对不要在回复中写函数名或代码！**

回复长度与节奏 尽量短，几句就好，给用户留空间回话。多用问句推进。别一口气讲完。别使用结构化小标题。别用冒号引导解释。

"""


def format_working_context(context: Dict[str, Any]) -> str:
    """
    格式化工作上下文为文本
    
    Args:
        context: 工作上下文字典
    
    Returns:
        格式化的文本
    """
    if not context:
        return ""
    
    parts = []
    
    # 用户信息
    if "user_name" in context:
        parts.append(f"**用户名称**：{context['user_name']}")
    
    # 当前情感
    if "current_emotion" in context:
        parts.append(f"**当前情感状态**：{context['current_emotion']}")
    
    # 情感历史
    if "emotion_history" in context:
        emotions = context["emotion_history"]
        if emotions:
            emotion_str = "、".join([f"{e['type']}({e.get('intensity', '?')})" for e in emotions[:5]])
            parts.append(f"**近期情感变化**：{emotion_str}")
    
    # 最近事件
    if "recent_events" in context:
        events = context["recent_events"]
        if events:
            events_str = "、".join(events[:5])
            parts.append(f"**近期事件**：{events_str}")
    
    # 用户偏好
    if "preferences" in context:
        prefs = context["preferences"]
        if prefs:
            parts.append(f"**用户偏好**：{prefs}")
    
    # 待跟进事项
    if "follow_ups" in context:
        follow_ups = context["follow_ups"]
        if follow_ups:
            parts.append(f"**待跟进**：{', '.join(follow_ups)}")
    
    return "\n".join(parts)


def format_emotion_analysis_prompt(text: str) -> str:
    """
    格式化情感分析提示词
    
    Args:
        text: 待分析的文本
    
    Returns:
        格式化的提示词
    """
    return f"""请分析以下文本的情感状态，返回JSON格式：

文本："{text}"

请返回以下格式的JSON：
{{
    "emotion_type": "情感类型（如：喜悦、悲伤、焦虑、愤怒、平静等）",
    "intensity": 0.0-1.0之间的数值表示情感强度,
    "trigger": "可能的触发因素（如果能识别）",
    "needs": "用户可能的情感需求"
}}

只返回JSON，不要其他解释。"""


def format_summary_prompt(messages: List[Dict[str, str]]) -> str:
    """
    格式化摘要生成提示词
    
    Args:
        messages: 消息列表
    
    Returns:
        格式化的提示词
    """
    conversation = "\n".join([
        f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
        for m in messages
    ])
    
    return f"""请对以下对话生成简洁摘要，重点关注：
1. 用户的主要情感状态变化
2. 讨论的关键话题
3. 重要的生活事件
4. 待跟进的事项

对话内容：
{conversation}

请返回以下格式的JSON：
{{
    "summary": "对话摘要（100字以内）",
    "main_topics": ["话题1", "话题2"],
    "emotional_arc": "情感变化轨迹描述",
    "follow_ups": ["待跟进事项1", "待跟进事项2"]
}}

只返回JSON，不要其他解释。"""

