"""
配置管理模块
支持从环境变量读取配置，区分开发/生产环境
"""

import os
from dataclasses import dataclass, field
from typing import Literal
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
    
    # 数据库路径
    database_path: str = field(
        default_factory=lambda: os.getenv("DATABASE_PATH", "./data/emotional_bot.db")
    )
    
    # 日志级别
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    
    # Gradio配置
    gradio_server_port: int = field(
        default_factory=lambda: int(os.getenv("GRADIO_SERVER_PORT", "7860"))
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
    
    def ensure_data_dir(self) -> None:
        """确保数据目录存在"""
        data_dir = Path(self.database_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)


# 全局配置实例
config = Config()

