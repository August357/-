"""下载项目所需的 HuggingFace 模型到 models/ 目录"""

import os
import logging
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download

from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODELS = [
    ("THUDM/chatglm3-6b", MODELS_DIR / "chatglm3-6b"),
    ("BAAI/bge-large-zh-v1.5", MODELS_DIR / "bge-large-zh-v1.5"),
    ("BAAI/bge-reranker-base", MODELS_DIR / "bge-reranker-base"),
]


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for repo_id, local_dir in MODELS:
        logger.info("正在下载 %s -> %s", repo_id, local_dir)
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir), resume_download=True)
        logger.info("  完成: %s", local_dir.name)
    logger.info("全部模型下载完成！")


if __name__ == "__main__":
    main()
