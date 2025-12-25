"""
情感分析模块测试
TDD: 先写测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.emotion.analyzer import EmotionAnalyzer, EmotionResult


class TestEmotionAnalyzer:
    """情感分析器测试"""
    
    @pytest.fixture
    def analyzer(self):
        """创建分析器实例"""
        from src.emotion.analyzer import EmotionAnalyzer
        return EmotionAnalyzer()
    
    # ============ 规则层测试 ============
    
    def test_analyze_happy_text(self, analyzer):
        """测试: 分析开心的文本"""
        result = analyzer.analyze_rule_based("今天太开心了！终于完成了项目！")
        
        assert result is not None
        assert result["emotion_type"] in ["喜悦", "开心", "高兴"]
        assert result["intensity"] > 0.5
    
    def test_analyze_sad_text(self, analyzer):
        """测试: 分析悲伤的文本"""
        result = analyzer.analyze_rule_based("我很难过，心情很低落")
        
        assert result is not None
        assert result["emotion_type"] in ["悲伤", "难过", "低落"]
        assert result["intensity"] > 0.3
    
    def test_analyze_anxious_text(self, analyzer):
        """测试: 分析焦虑的文本"""
        result = analyzer.analyze_rule_based("明天的面试让我非常焦虑和紧张")
        
        assert result is not None
        assert result["emotion_type"] in ["焦虑", "紧张"]
        assert result["intensity"] > 0.5
    
    def test_analyze_angry_text(self, analyzer):
        """测试: 分析愤怒的文本"""
        result = analyzer.analyze_rule_based("太气人了！这也太过分了！")
        
        assert result is not None
        assert result["emotion_type"] in ["愤怒", "生气"]
        assert result["intensity"] > 0.5
    
    def test_analyze_neutral_text(self, analyzer):
        """测试: 分析中性文本"""
        result = analyzer.analyze_rule_based("今天天气不错")
        
        # 中性文本可能返回None或低置信度
        if result:
            assert result["confidence"] < 0.6
    
    def test_rule_based_returns_confidence(self, analyzer):
        """测试: 规则分析返回置信度"""
        result = analyzer.analyze_rule_based("非常非常开心！太棒了！")
        
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
    
    # ============ 混合分析测试 ============
    
    def test_analyze_uses_rule_when_confident(self, analyzer):
        """测试: 高置信度时使用规则结果"""
        # 明显的情感表达应该直接使用规则结果
        result = analyzer.analyze("我今天超级超级开心！！！")
        
        assert result is not None
        assert result.emotion_type in ["喜悦", "开心", "高兴"]
    
    def test_analyze_returns_none_for_neutral(self, analyzer):
        """测试: 对于无情感文本返回None"""
        # 完全中性的文本
        result = analyzer.analyze("1加1等于2")
        
        # 无法识别情感时可能返回None
        # 这是正常行为
        assert result is None or result.confidence < 0.5
    
    # ============ 情感词典测试 ============
    
    def test_emotion_keywords_exist(self, analyzer):
        """测试: 情感关键词字典存在"""
        assert hasattr(analyzer, "emotion_keywords")
        assert "喜悦" in analyzer.emotion_keywords or "开心" in [v for vals in analyzer.emotion_keywords.values() for v in vals]
    
    def test_emotion_types(self, analyzer):
        """测试: 支持的情感类型"""
        supported_types = analyzer.get_supported_emotions()
        
        assert "喜悦" in supported_types
        assert "悲伤" in supported_types
        assert "焦虑" in supported_types
        assert "愤怒" in supported_types


class TestEmotionResult:
    """情感分析结果测试"""
    
    def test_emotion_result_structure(self):
        """测试: 结果结构完整"""
        result = EmotionResult(
            emotion_type="喜悦",
            intensity=0.8,
            confidence=0.9,
            trigger="完成项目"
        )
        
        assert result.emotion_type == "喜悦"
        assert result.intensity == 0.8
        assert result.confidence == 0.9
        assert result.trigger == "完成项目"
    
    def test_emotion_result_to_dict(self):
        """测试: 转换为字典"""
        result = EmotionResult(
            emotion_type="悲伤",
            intensity=0.6
        )
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert d["emotion_type"] == "悲伤"
        assert d["intensity"] == 0.6

