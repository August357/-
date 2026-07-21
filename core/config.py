"""
集中配置管理（pydantic-settings + .env）

所有阈值/权重/限额常量统一在此定义，可用环境变量或项目根目录 .env 覆盖。
示例见 .env.example。
"""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- JWT 签名密钥（必填，未设置时后端拒绝启动）----
    JWT_SECRET: str | None = None

    # ---- Reranker 阈值 ----
    # BGE-Reranker 分数低于此阈值视为与知识库无关（cross-encoder logits）
    # 实测：高度相关约 4~6，主题相近但编号不符约 1~2，完全无关常为负数
    RERANK_MIN_SCORE: float = 2.5
    # 低于此分数且非「实验编号不匹配」→ 走通用 LLM（闲聊/编程等）
    GENERAL_LLM_MAX_SCORE: float = 1.0

    # ---- LLM 推理限额（8GB 显存安全上限）----
    MAX_INPUT_CHARS: int = 2500
    MAX_NEW_TOKENS: int = 512
    MAX_TOTAL_TOKENS: int = 2048

    # ---- 多轮对话 ----
    # 拼入 history 的最近轮数
    HISTORY_ROUNDS: int = 3
    MAX_HISTORY_CHARS: int = 800

    # ---- 检索与融合权重 ----
    RETRIEVAL_CANDIDATE_K: int = 10
    RETRIEVAL_VECTOR_WEIGHT: float = 0.7
    RETRIEVAL_BM25_WEIGHT: float = 0.3
    BM25_TOP_N: int = 10
    # 向量 L2 距离过滤阈值（BGE 归一化向量，越小越相似）
    MAX_L2_DISTANCE: float = 1.2

    # ---- 上传限制 ----
    MAX_UPLOAD_MB: int = 50

    # ---- CORS（逗号分隔）----
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ---- 日志 ----
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = os.path.join(BASE_DIR, "logs", "app.log")

    @property
    def cors_origin_list(self) -> list:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
