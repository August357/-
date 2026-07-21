"""
统一日志配置：控制台 + 文件双输出。

在应用入口（backend/main.py）调用一次 setup_logging()；
各模块通过 logging.getLogger(__name__) 获取 logger。
"""

import logging
import os
from logging.handlers import RotatingFileHandler

_configured = False


def setup_logging(level: str = "INFO", log_file: str = None):
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    _configured = True
