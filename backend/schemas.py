from pydantic import BaseModel
from typing import List, Optional


class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class BuildDBRequest(BaseModel):
    chunk_size: int
    chunk_overlap: int
    split_mode: str
    top_k: int


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    token: str
    username: str


# ==========================
# 新增 Schemas
# ==========================

class ChatRequest(BaseModel):
    query: str
    use_reranker: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    context_length: int
    retrieved_docs: int


class UploadResponse(BaseModel):
    message: str
    success: bool
    files_uploaded: int


class RebuildRequest(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 100
    split_mode: str = "zh"
    top_k: int = 5


class RebuildResponse(BaseModel):
    message: str
    success: bool
    total_chunks: int


class FileListResponse(BaseModel):
    files: List[str]
    total: int


class DeleteResponse(BaseModel):
    message: str
    success: bool


class PreviewResponse(BaseModel):
    content: str
    filename: str
    length: int


class ChunkInfo(BaseModel):
    chunk_id: int
    source: str
    page: int
    length: int
    preview: str


class ChunksResponse(BaseModel):
    chunks: List[ChunkInfo]
    total: int


class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    vector_db_exists: bool
    total_chunks: int
    total_files: int


class SystemInfoResponse(BaseModel):
    model_name: str
    embedding_model: str
    device: str
    python_version: str
    pytorch_version: str


class MessageResponse(BaseModel):
    message: str
