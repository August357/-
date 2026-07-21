from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Depends,
    HTTPException
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import shutil
import os

from backend.schemas import AskRequest, BuildDBRequest, RegisterRequest, LoginRequest
from backend.system_api import get_system_info
from backend.auth import register_user, authenticate_user, create_access_token, get_current_user, save_chat_history, get_chat_history, get_session_history
from backend.database import get_db
from core.qa_chain import ask, ask_stream
from core.llm import load_model
from core.vector_store import (
    build_vector_db,
    list_knowledge_files,
    preview_file,
    delete_file,
    load_config,
    parse_chunk_info_file,
    EMBEDDING_MODEL_NAME
)

app = FastAPI()

_llm_ready = False
_llm_load_error = ""


@app.on_event("startup")
def preload_llm():
    """启动时预加载 LLM；失败时不终止整个服务，便于前端仍能访问其他接口"""
    global _llm_ready, _llm_load_error
    print("正在预加载 ChatGLM3-6B 模型...")
    try:
        load_model()
        _llm_ready = True
        _llm_load_error = ""
        print("ChatGLM3-6B 预加载完成")
    except Exception as e:
        _llm_ready = False
        _llm_load_error = str(e)
        print(f"ChatGLM3-6B 预加载失败（服务仍运行）: {e}")
        print("问答接口将返回错误提示，请检查 models/chatglm3-6b 与 GPU 显存")


@app.get("/api/health")
def health():
    """健康检查：前端/代理报错时可先访问此接口确认后端是否存活"""
    return {
        "status": "ok",
        "llm_ready": _llm_ready,
        "llm_error": _llm_load_error or None,
    }


# 仅允许前端开发服务器来源，不再使用通配符
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# 文档目录
DOCS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "docs"
)

# 上传限制
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB


def safe_join(base_dir: str, filename: str) -> str:
    """
    校验文件名并拼接为 base_dir 内的绝对路径，防止路径穿越。
    非法文件名直接抛出 400。
    """
    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="非法文件名")
    # 拒绝任何路径分隔符，只允许纯文件名
    if "/" in filename or "\\" in filename or "\x00" in filename:
        raise HTTPException(status_code=400, detail="非法文件名")

    base_real = os.path.realpath(base_dir)
    target = os.path.realpath(os.path.join(base_real, filename))
    if os.path.commonpath([base_real, target]) != base_real:
        raise HTTPException(status_code=400, detail="非法文件名")
    return target


# ==========================
# 问答接口
# ==========================

@app.post("/api/ask")
def ask_api(
    req: AskRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        history = get_session_history(
            current_user["user_id"], req.session_id, limit=3
        ) if req.session_id else []

        result = ask(req.query, history=history)

        save_chat_history(
            user_id=current_user["user_id"],
            question=req.query,
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            session_id=req.session_id
        )

        return result

    except Exception as e:
        import traceback

        print("\n")
        print("=" * 80)
        print("ASK接口异常")
        traceback.print_exc()
        print("=" * 80)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================
# 流式问答接口（SSE）
# ==========================

@app.post("/api/ask/stream")
def ask_stream_api(
    req: AskRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    SSE 流式问答：逐 token 推送 {"type":"token","text":...}，
    末尾推送 {"type":"end","sources":...} 事件后发送 [DONE]。
    """
    history = get_session_history(
        current_user["user_id"], req.session_id, limit=3
    ) if req.session_id else []

    def event_gen():
        full_answer = ""
        final_sources = []
        try:
            for event in ask_stream(req.query, history=history):
                if event.get("type") == "token":
                    full_answer += event.get("text", "")
                elif event.get("type") == "end":
                    final_sources = event.get("sources", [])
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            save_chat_history(
                user_id=current_user["user_id"],
                question=req.query,
                answer=full_answer,
                sources=final_sources,
                session_id=req.session_id
            )
            yield "data: [DONE]\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ==========================
# 用户注册接口
# ==========================

@app.post("/api/register")
def register(req: RegisterRequest):
    success = register_user(req.username, req.password)
    if success:
        return {"msg": "注册成功"}
    else:
        raise HTTPException(status_code=400, detail="用户名已存在")


# ==========================
# 用户登录接口
# ==========================

@app.post("/api/login")
def login(req: LoginRequest):
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    access_token = create_access_token(
        data={"sub": user["username"], "user_id": user["user_id"]}
    )

    return {
        "token": access_token,
        "username": user["username"]
    }


# ==========================
# 保存聊天记录接口
# ==========================

@app.post("/api/chat/save")
def save_chat(req: dict, current_user: dict = Depends(get_current_user)):
    success = save_chat_history(
        user_id=current_user["user_id"],
        question=req.get("question"),
        answer=req.get("answer"),
        sources=req.get("sources", [])
    )
    return {"success": success}


# ==========================
# 获取聊天历史接口
# ==========================

@app.get("/api/chat/history")
def get_history(current_user: dict = Depends(get_current_user)):
    history = get_chat_history(current_user["user_id"])
    return {"history": history}


# ==========================
# 配置接口
# ==========================

@app.get("/api/config")
def get_config(current_user: dict = Depends(get_current_user)):
    return load_config()


# 全局变量跟踪构建进度
build_progress = {"progress": 0, "status": "idle"}

# ==========================
# 修改Chunk配置接口
# ==========================

@app.post("/api/build-db")
def build_db(req: BuildDBRequest, current_user: dict = Depends(get_current_user)):
    global build_progress

    # 重置进度
    build_progress = {"progress": 0, "status": "building"}

    try:
        success = build_vector_db(
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            split_mode=req.split_mode,
            top_k=req.top_k
        )

        build_progress = {"progress": 100, "status": "completed"}

        return {
            "success": success
        }
    except Exception as e:
        build_progress = {"progress": 0, "status": "failed"}
        raise HTTPException(status_code=500, detail=str(e))


# ==========================
# 获取构建进度接口
# ==========================

@app.get("/api/build-status")
def get_build_status(current_user: dict = Depends(get_current_user)):
    return build_progress


# ==========================
# 文件列表接口
# ==========================

@app.get("/api/files")
def get_files(current_user: dict = Depends(get_current_user)):
    return {
        "files": list_knowledge_files()
    }


# ==========================
# 文件预览接口
# ==========================

@app.get("/api/preview/{filename}")
def preview(filename: str, current_user: dict = Depends(get_current_user)):
    safe_join(DOCS_DIR, filename)
    return {
        "content": preview_file(filename)
    }


# ==========================
# 删除文件接口
# ==========================

@app.delete("/api/file/{filename}")
def delete(filename: str, current_user: dict = Depends(get_current_user)):
    safe_join(DOCS_DIR, filename)
    result = delete_file(filename)
    return {
        "msg": result
    }


# ==========================
# 上传文件接口
# ==========================

@app.post("/api/upload")
async def upload(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {ext or '(无扩展名)'}，仅允许: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    path = safe_join(DOCS_DIR, file.filename)
    os.makedirs(DOCS_DIR, exist_ok=True)

    # 流式写入并统计大小，超过 50MB 即中止并清理残留文件
    written = 0
    try:
        with open(path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_SIZE:
                    raise HTTPException(status_code=413, detail="文件超过 50MB 大小限制")
                f.write(chunk)
    except Exception:
        if os.path.exists(path):
            os.remove(path)
        raise

    return {
        "msg": "上传成功"
    }


# ==========================
# Chunk可视化接口
# ==========================

@app.get("/api/chunks")
def get_chunks(source: str = None, current_user: dict = Depends(get_current_user)):
    """获取 Chunk 信息，按文档分组；可选 source 参数只返回指定文档"""
    from core.vector_store import list_chunks_grouped_by_source

    data = list_chunks_grouped_by_source()
    if not data["documents"]:
        return {
            "documents": [],
            "total": 0,
            "document_count": 0,
            "message": "暂无 Chunk 信息。请先在「配置」页重建向量库。",
        }

    if source:
        matched = [d for d in data["documents"] if d["source"] == source]
        return {
            "documents": matched,
            "total": sum(d["chunk_count"] for d in matched),
            "document_count": len(matched),
        }

    return data


# ==========================
# 系统监控接口
# ==========================

@app.get("/api/system")
def system_info(current_user: dict = Depends(get_current_user)):
    return get_system_info()


# ==========================
# 首页统计接口
# ==========================

@app.get("/api/dashboard")
def dashboard(current_user: dict = Depends(get_current_user)):
    docs_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "docs"
    )

    file_count = 0

    if os.path.exists(docs_dir):
        file_count = len([
            f for f in os.listdir(docs_dir)
            if os.path.isfile(os.path.join(docs_dir, f))
        ])

    # 使用 chunk_info 的真实解析结果统计，而非文本计数
    chunk_count = len(parse_chunk_info_file())

    vector_db_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "vector_db",
        "index.faiss"
    )

    return {
        "files": file_count,
        "chunks": chunk_count,
        "model": "ChatGLM3-6B",
        "embedding": EMBEDDING_MODEL_NAME,
        "vector_db": os.path.exists(vector_db_path)
    }


# ==========================
# 首页详情接口（加分项）
# ==========================

@app.get("/api/dashboard-detail")
def dashboard_detail(current_user: dict = Depends(get_current_user)):
    docs_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "docs"
    )

    recent_files = []
    if os.path.exists(docs_dir):
        files = os.listdir(docs_dir)
        files.sort(key=lambda x: os.path.getmtime(os.path.join(docs_dir, x)), reverse=True)
        recent_files = files[:5]

    faiss_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "vector_db",
        "index.faiss"
    )

    faiss_size = "0MB"
    if os.path.exists(faiss_path):
        size = os.path.getsize(faiss_path)
        faiss_size = f"{round(size / 1024**2, 2)}MB"

    return {
        "recent_files": recent_files,
        "faiss_size": faiss_size,
        "questions": 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
