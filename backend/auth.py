from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from backend.database import get_db
import os
import sqlite3
import json

# JWT配置：密钥必须从环境变量读取，禁止硬编码默认值
SECRET_KEY = os.environ.get("JWT_SECRET")
if not SECRET_KEY:
    raise RuntimeError(
        "环境变量 JWT_SECRET 未设置，服务拒绝启动。"
        "请先执行：export JWT_SECRET='<随机长字符串>' 后再启动后端。"
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24小时

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8') if isinstance(hashed_password, str) else hashed_password
    )


def get_password_hash(password: str) -> str:
    """加密密码"""
    return bcrypt.hashpw(
        password.encode('utf-8'),
        bcrypt.gensalt()
    ).decode('utf-8')


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """创建JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """验证token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        if username is None or user_id is None:
            return None
        return {"username": username, "user_id": user_id}
    except JWTError:
        return None


def get_current_user(token: str = Depends(oauth2_scheme)):
    """获取当前用户"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )

    user_info = verify_token(token)
    if user_info is None:
        raise credentials_exception

    return user_info


def register_user(username: str, password: str) -> bool:
    """注册用户"""
    conn = get_db()
    cursor = conn.cursor()

    try:
        # 检查用户名是否已存在
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            return False

        # 创建用户
        hashed_password = get_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed_password)
        )
        conn.commit()
        return True
    except Exception as e:
        print("注册异常:", e)
        conn.rollback()
        raise
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """验证用户登录"""
    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT id, username, password FROM users WHERE username = ?",
            (username,)
        )
        user = cursor.fetchone()

        if not user:
            return None

        if not verify_password(password, user["password"]):
            return None

        return {
            "user_id": user["id"],
            "username": user["username"]
        }
    finally:
        conn.close()


def save_chat_history(user_id: int, question: str, answer: str, sources: list = None, session_id: str = None) -> bool:
    """保存聊天记录"""
    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """INSERT INTO chat_history (user_id, session_id, question, answer, sources)
               VALUES (?, ?, ?, ?, ?)""",
            (
                user_id,
                session_id,
                question,
                answer,
                json.dumps(sources, ensure_ascii=False) if sources is not None else None
            )
        )
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        return False
    finally:
        conn.close()


def get_session_history(user_id: int, session_id: str, limit: int = 3) -> list:
    """
    获取指定会话最近 limit 轮对话，按时间正序返回，
    用于拼入 LLM 的 history（多轮对话）。
    """
    if not session_id:
        return []
    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """SELECT question, answer
               FROM chat_history
               WHERE user_id = ? AND session_id = ?
               ORDER BY create_time DESC, id DESC
               LIMIT ?""",
            (user_id, session_id, limit)
        )
        rows = cursor.fetchall()
        rows.reverse()  # 恢复时间正序
        return [{"question": row["question"], "answer": row["answer"]} for row in rows]
    finally:
        conn.close()


def get_chat_history(user_id: int, limit: int = 50) -> list:
    """获取聊天历史"""
    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """SELECT question, answer, sources, create_time
               FROM chat_history
               WHERE user_id = ?
               ORDER BY create_time DESC
               LIMIT ?""",
            (user_id, limit)
        )
        rows = cursor.fetchall()

        return [
            {
                "question": row["question"],
                "answer": row["answer"],
                "sources": row["sources"],
                "create_time": row["create_time"]
            }
            for row in rows
        ]
    finally:
        conn.close()