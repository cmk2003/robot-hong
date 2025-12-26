"""
日志工具模块
支持控制台和文件双输出
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler


# 全局日志目录（延迟初始化）
_log_dir: Optional[Path] = None
_file_handler_added: bool = False


def _get_log_dir() -> Path:
    """获取日志目录"""
    global _log_dir
    if _log_dir is None:
        log_path = os.getenv("LOG_PATH", "./logs")
        _log_dir = Path(log_path)
        _log_dir.mkdir(parents=True, exist_ok=True)
    return _log_dir


def _setup_file_handler(logger: logging.Logger) -> None:
    """为 logger 添加文件处理器"""
    global _file_handler_added
    
    # 检查是否已经有文件处理器
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            return
    
    try:
        log_dir = _get_log_dir()
        log_file = log_dir / "robot-hong.log"
        
        # 使用 RotatingFileHandler，单文件最大 10MB，保留 5 个备份
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        _file_handler_added = True
    except Exception as e:
        # 文件日志失败不应影响程序运行
        print(f"警告: 无法创建日志文件: {e}", file=sys.stderr)


def get_logger(name: Optional[str] = None, level: Optional[str] = None) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称，默认为 'emotional_bot'
        level: 日志级别，默认从环境变量 LOG_LEVEL 读取
    
    Returns:
        配置好的日志记录器
    """
    logger_name = name or "emotional_bot"
    log_level = level or os.getenv("LOG_LEVEL", "INFO")
    
    logger = logging.getLogger(logger_name)
    
    # 避免重复添加handler
    if not logger.handlers:
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # 控制台输出
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        # 格式化
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 添加文件处理器
        _setup_file_handler(logger)
    
    return logger

