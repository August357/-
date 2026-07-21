"""独立脚本：重建向量库

统一入口，等价于调用 core.vector_store.build_vector_db()。
用法：python rebuild_db_v3.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.logging_config import setup_logging
from core.vector_store import build_vector_db

setup_logging()


if __name__ == "__main__":
    success = build_vector_db()
    sys.exit(0 if success else 1)
