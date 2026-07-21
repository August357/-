"""仅下载 ChatGLM3-6B 基础模型的便捷脚本（完整模型列表请用 download_models.py）"""

import os
import logging

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # 下载完整6B模型
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "chatglm3-6b")
    snapshot_download(
        repo_id="THUDM/chatglm3-6b",
        local_dir=target,
        resume_download=True
    )
    logger.info("基础模型下载完成！")
