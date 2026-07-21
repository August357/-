import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "rag.db"
)

def get_db():
    logger.debug("当前数据库: %s", os.path.abspath(DB_PATH))

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """初始化数据库表"""
    conn = get_db()
    cursor = conn.cursor()

    # 用户表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            create_time TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 聊天历史表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_id TEXT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT,
            create_time TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # 旧库迁移：补充 session_id 列
    cursor.execute("PRAGMA table_info(chat_history)")
    columns = {row[1] for row in cursor.fetchall()}
    if "session_id" not in columns:
        cursor.execute("ALTER TABLE chat_history ADD COLUMN session_id TEXT")

    conn.commit()
    conn.close()

# 初始化数据库
init_db()