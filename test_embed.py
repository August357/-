"""验证本地 Embedding 模型可加载的冒烟脚本：python test_embed.py"""

import os
import logging

from sentence_transformers import SentenceTransformer

from core.config import BASE_DIR
from core.logging_config import setup_logging
from core.vector_store import EMBEDDING_MODEL_PATH

setup_logging()
logger = logging.getLogger(__name__)

logger.info("模型路径: %s", EMBEDDING_MODEL_PATH)
logger.info("路径存在: %s", os.path.exists(EMBEDDING_MODEL_PATH))

model = SentenceTransformer(EMBEDDING_MODEL_PATH, device="cpu")
logger.info("加载成功")
