"""
日志工具模块
"""

import logging
import sys
from typing import Optional


def get_logger(name: Optional[str] = None, level: Optional[str] = None) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称，默认为 'emotional_bot'
        level: 日志级别，默认从环境变量 LOG_LEVEL 读取
    
    Returns:
        配置好的日志记录器
    """
    import os
    
    logger_name = name or "emotional_bot"
    log_level = level or os.getenv("LOG_LEVEL", "INFO")
    
    logger = logging.getLogger(logger_name)
    
    # 避免重复添加handler
    if not logger.handlers:
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # 控制台输出
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        
        # 格式化
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

