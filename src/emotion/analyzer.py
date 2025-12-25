"""
情感分析模块
混合策略：规则优先，LLM兜底
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class EmotionResult:
    """情感分析结果"""
    emotion_type: str
    intensity: float  # 0.0-1.0
    confidence: float = 0.0  # 分析置信度
    trigger: Optional[str] = None  # 触发因素
    needs: Optional[str] = None  # 情感需求
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class EmotionAnalyzer:
    """
    混合情感分析器
    - 规则层：关键词匹配 + 情感词典（快速、低成本）
    - LLM层：复杂情况调用LLM分析（准确、高成本）
    """
    
    # 情感关键词字典
    emotion_keywords: Dict[str, List[str]] = {
        "喜悦": [
            "开心", "高兴", "快乐", "幸福", "棒", "太好了", "兴奋", "激动",
            "满足", "愉快", "欣喜", "喜悦", "欢喜", "乐", "爽", "赞", "好开心",
            "超级开心", "非常开心", "特别开心", "真开心", "太开心"
        ],
        "悲伤": [
            "难过", "伤心", "悲伤", "哭", "泪", "心痛", "痛苦", "绝望",
            "沮丧", "低落", "失落", "郁闷", "忧伤", "心碎", "难受", "不开心"
        ],
        "愤怒": [
            "生气", "愤怒", "气死", "气人", "火大", "烦躁", "恼火", "暴躁",
            "讨厌", "恨", "可恶", "过分", "太过分", "气愤", "愤恨"
        ],
        "焦虑": [
            "焦虑", "紧张", "担心", "害怕", "不安", "忐忑", "着急", "急",
            "慌", "恐慌", "压力", "压力大", "焦急", "心烦", "烦"
        ],
        "恐惧": [
            "害怕", "恐惧", "可怕", "吓", "恐怖", "惊恐", "惧怕", "畏惧"
        ],
        "惊讶": [
            "惊讶", "震惊", "意外", "没想到", "居然", "竟然", "天啊", "哇"
        ],
        "厌恶": [
            "恶心", "讨厌", "厌恶", "反感", "烦", "无语", "受不了"
        ],
        "平静": [
            "平静", "安静", "放松", "轻松", "淡定", "冷静", "安心"
        ],
        "孤独": [
            "孤独", "寂寞", "一个人", "没人理", "孤单", "冷清"
        ],
        "感激": [
            "感谢", "感激", "谢谢", "多谢", "感恩", "谢", "太感谢"
        ],
        "希望": [
            "希望", "期待", "盼望", "憧憬", "相信", "会好的", "加油"
        ],
        "困惑": [
            "困惑", "迷茫", "迷惑", "不懂", "不理解", "不明白", "纠结"
        ],
        "失望": [
            "失望", "遗憾", "可惜", "落空", "不如意", "白费"
        ]
    }
    
    # 强度修饰词
    intensity_modifiers: Dict[str, float] = {
        # 增强词
        "非常": 0.2,
        "特别": 0.2,
        "超级": 0.25,
        "极其": 0.25,
        "太": 0.15,
        "真的": 0.1,
        "好": 0.1,
        "很": 0.15,
        "十分": 0.2,
        # 减弱词
        "有点": -0.15,
        "有些": -0.1,
        "稍微": -0.2,
        "略微": -0.15,
    }
    
    def __init__(self, llm_client=None):
        """
        初始化分析器
        
        Args:
            llm_client: LLM客户端（可选，用于复杂情况）
        """
        self.llm_client = llm_client
    
    def get_supported_emotions(self) -> List[str]:
        """获取支持的情感类型列表"""
        return list(self.emotion_keywords.keys())
    
    def analyze(self, text: str) -> Optional[EmotionResult]:
        """
        分析文本情感（混合策略）
        
        Args:
            text: 待分析文本
        
        Returns:
            情感分析结果
        """
        # 第一步：规则分析
        rule_result = self.analyze_rule_based(text)
        
        if rule_result and rule_result.get("confidence", 0) >= 0.7:
            # 高置信度，直接返回规则结果
            return EmotionResult(
                emotion_type=rule_result["emotion_type"],
                intensity=rule_result["intensity"],
                confidence=rule_result["confidence"],
                trigger=rule_result.get("trigger")
            )
        
        # 第二步：如果有LLM客户端且规则置信度低，尝试LLM分析
        if self.llm_client and (not rule_result or rule_result.get("confidence", 0) < 0.5):
            llm_result = self._llm_analyze(text)
            if llm_result:
                return EmotionResult(
                    emotion_type=llm_result["emotion_type"],
                    intensity=llm_result["intensity"],
                    confidence=llm_result.get("confidence", 0.8),
                    trigger=llm_result.get("trigger"),
                    needs=llm_result.get("needs")
                )
        
        # 返回规则结果（即使置信度不高）
        if rule_result:
            return EmotionResult(
                emotion_type=rule_result["emotion_type"],
                intensity=rule_result["intensity"],
                confidence=rule_result["confidence"],
                trigger=rule_result.get("trigger")
            )
        
        return None
    
    def analyze_rule_based(self, text: str) -> Optional[Dict[str, Any]]:
        """
        基于规则的情感分析
        
        Args:
            text: 待分析文本
        
        Returns:
            分析结果字典或 None
        """
        if not text or not text.strip():
            return None
        
        text_lower = text.lower()
        
        # 统计各情感的匹配分数
        emotion_scores: Dict[str, float] = {}
        emotion_matches: Dict[str, List[str]] = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            matches = []
            score = 0.0
            
            for keyword in keywords:
                if keyword in text_lower:
                    matches.append(keyword)
                    # 基础分数
                    score += 1.0
                    # 关键词长度加分（长词更精确）
                    score += len(keyword) * 0.05
            
            if matches:
                emotion_scores[emotion] = score
                emotion_matches[emotion] = matches
        
        if not emotion_scores:
            return None
        
        # 找出最高分的情感
        best_emotion = max(emotion_scores, key=emotion_scores.get)
        best_score = emotion_scores[best_emotion]
        
        # 计算基础强度（0.4-0.9之间）
        base_intensity = min(0.4 + best_score * 0.1, 0.9)
        
        # 应用强度修饰词
        intensity_modifier = 0.0
        for modifier, value in self.intensity_modifiers.items():
            if modifier in text_lower:
                intensity_modifier += value
        
        # 感叹号增强
        exclamation_count = text.count("！") + text.count("!")
        intensity_modifier += min(exclamation_count * 0.05, 0.15)
        
        # 计算最终强度
        final_intensity = max(0.1, min(1.0, base_intensity + intensity_modifier))
        
        # 计算置信度（基于匹配数量和强度）
        match_count = len(emotion_matches[best_emotion])
        confidence = min(0.4 + match_count * 0.15 + intensity_modifier * 0.5, 0.95)
        
        return {
            "emotion_type": best_emotion,
            "intensity": round(final_intensity, 2),
            "confidence": round(confidence, 2),
            "matches": emotion_matches[best_emotion]
        }
    
    def _llm_analyze(self, text: str) -> Optional[Dict[str, Any]]:
        """
        使用LLM进行情感分析
        
        Args:
            text: 待分析文本
        
        Returns:
            分析结果字典或 None
        """
        if not self.llm_client:
            return None
        
        try:
            from ..llm.prompts import format_emotion_analysis_prompt
            
            prompt = format_emotion_analysis_prompt(text)
            
            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            content = response.get("content", "")
            if content:
                import json
                # 尝试解析JSON
                result = json.loads(content)
                return result
        except Exception:
            pass
        
        return None

