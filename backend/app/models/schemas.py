from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    history: Optional[List[ChatMessage]] = Field(default_factory=list, description="对话历史")
    stream: bool = Field(default=False, description="是否流式返回")
    session_id: Optional[str] = Field(default=None, description="会话ID")


class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument] = Field(default_factory=list)
    agent_trace: List[str] = Field(default_factory=list, description="Agent执行路径")
    session_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class KnowledgeBaseStatus(BaseModel):
    total_documents: int
    collection_name: str
    status: str = "ready"


class DataImportRequest(BaseModel):
    file_path: Optional[str] = None
    reset: bool = Field(default=False, description="是否重置知识库")


class DataImportResponse(BaseModel):
    imported_count: int
    status: str
    message: str
