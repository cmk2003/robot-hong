"""
配置管理模块
支持从环境变量读取配置，区分开发/生产环境
支持多 Agent 模式下每个 Agent 独立配置模型
"""

import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict
from pathlib import Path

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class LLMProviderConfig:
    """LLM提供商配置"""
    base_url: str
    model: str
    api_key: str


@dataclass
class MultiAgentConfig:
    """
    多 Agent 模型配置
    
    每个 Agent 可以独立配置模型，如果不配置则使用默认模型
    
    环境变量示例：
    - AGENT_EMOTION_MODEL=qwen-turbo  # 情感分析用小模型
    - AGENT_MEMORY_MODEL=qwen-turbo   # 记忆检索用小模型
    - AGENT_RESPONSE_MODEL=qwen-max   # 回复生成用大模型
    - AGENT_SAVE_MODEL=qwen-turbo     # 记忆保存用小模型
    - AGENT_REVIEW_MODEL=qwen-plus    # 质量评审用中等模型
    """
    # 默认模型（所有 Agent 的 fallback）
    default_model: str = field(
        default_factory=lambda: os.getenv("AGENT_DEFAULT_MODEL", "")
    )
    
    # 各 Agent 的模型配置（为空则使用默认模型）
    emotion_model: str = field(
        default_factory=lambda: os.getenv("AGENT_EMOTION_MODEL", "")
    )
    memory_model: str = field(
        default_factory=lambda: os.getenv("AGENT_MEMORY_MODEL", "")
    )
    response_model: str = field(
        default_factory=lambda: os.getenv("AGENT_RESPONSE_MODEL", "")
    )
    save_model: str = field(
        default_factory=lambda: os.getenv("AGENT_SAVE_MODEL", "")
    )
    review_model: str = field(
        default_factory=lambda: os.getenv("AGENT_REVIEW_MODEL", "")
    )
    
    def get_model_for_agent(self, agent_name: str, fallback_model: str) -> str:
        """
        获取指定 Agent 的模型
        
        Args:
            agent_name: Agent 名称 (emotion, memory, response, save, review)
            fallback_model: 回退模型（如果没有配置）
        
        Returns:
            模型名称
        """
        model_map = {
            "emotion": self.emotion_model,
            "memory": self.memory_model,
            "response": self.response_model,
            "save": self.save_model,
            "review": self.review_model,
        }
        
        # 优先使用 Agent 专属模型
        model = model_map.get(agent_name, "")
        if model:
            return model
        
        # 其次使用默认模型
        if self.default_model:
            return self.default_model
        
        # 最后使用 fallback
        return fallback_model
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典"""
        return {
            "default": self.default_model or "(使用主配置)",
            "emotion": self.emotion_model or "(使用默认)",
            "memory": self.memory_model or "(使用默认)",
            "response": self.response_model or "(使用默认)",
            "save": self.save_model or "(使用默认)",
            "review": self.review_model or "(使用默认)",
        }


@dataclass
class Config:
    """应用配置"""
    
    # 环境
    env: Literal["development", "production"] = field(
        default_factory=lambda: os.getenv("ENV", "development")
    )
    
    # LLM提供商
    llm_provider: Literal["qwen", "deepseek"] = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "qwen")
    )
    
    # Agent 模式
    agent_mode: Literal["single", "multi"] = field(
        default_factory=lambda: os.getenv("AGENT_MODE", "single")
    )
    
    # 多 Agent 模型配置
    multi_agent_config: MultiAgentConfig = field(
        default_factory=MultiAgentConfig
    )
    
    # 数据库路径
    database_path: str = field(
        default_factory=lambda: os.getenv("DATABASE_PATH", "./data/emotional_bot.db")
    )
    
    # 日志级别
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    
    # 日志路径
    log_path: str = field(
        default_factory=lambda: os.getenv("LOG_PATH", "./logs")
    )
    
    # Gradio配置
    gradio_server_port: int = field(
        default_factory=lambda: int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    )
    
    # 激活码（用于用户验证）
    activation_code: str = field(
        default_factory=lambda: os.getenv("ACTIVATION_CODE", "")
    )
    
    # LLM提供商配置映射
    LLM_PROVIDERS: dict = field(default_factory=lambda: {
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-max",
            "env_key": "DASHSCOPE_API_KEY"
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "env_key": "DEEPSEEK_API_KEY"
        }
    })
    
    @property
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.env == "development"
    
    @property
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.env == "production"
    
    @property
    def gradio_server_name(self) -> str:
        """Gradio服务器地址"""
        return "127.0.0.1" if self.is_development else "0.0.0.0"
    
    def get_llm_config(self) -> LLMProviderConfig:
        """获取当前LLM提供商配置"""
        provider_info = self.LLM_PROVIDERS.get(self.llm_provider)
        if not provider_info:
            raise ValueError(f"不支持的LLM提供商: {self.llm_provider}")
        
        api_key = os.getenv(provider_info["env_key"], "")
        if not api_key:
            raise ValueError(f"未设置API密钥: {provider_info['env_key']}")
        
        return LLMProviderConfig(
            base_url=provider_info["base_url"],
            model=provider_info["model"],
            api_key=api_key
        )
    
    def get_agent_llm_config(self, agent_name: str) -> LLMProviderConfig:
        """
        获取指定 Agent 的 LLM 配置
        
        Args:
            agent_name: Agent 名称 (emotion, memory, response, save, review)
        
        Returns:
            LLMProviderConfig 实例
        """
        base_config = self.get_llm_config()
        
        # 获取该 Agent 的模型（可能与默认不同）
        model = self.multi_agent_config.get_model_for_agent(
            agent_name, 
            base_config.model
        )
        
        return LLMProviderConfig(
            base_url=base_config.base_url,
            model=model,
            api_key=base_config.api_key
        )
    
    def get_all_agent_configs(self) -> Dict[str, LLMProviderConfig]:
        """
        获取所有 Agent 的 LLM 配置
        
        Returns:
            字典 {agent_name: LLMProviderConfig}
        """
        agent_names = ["emotion", "memory", "response", "save", "review"]
        return {name: self.get_agent_llm_config(name) for name in agent_names}
    
    def ensure_data_dir(self) -> None:
        """确保数据目录存在"""
        data_dir = Path(self.database_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)


# 全局配置实例
config = Config()

