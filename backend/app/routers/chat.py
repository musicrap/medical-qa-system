import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest, ChatResponse, SourceDocument
from app.agents.medical_agent import get_agent

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        agent = get_agent()
        result = agent.run(
            query=request.query,
            history=request.history,
            session_id=session_id,
        )
        return ChatResponse(
            answer=result["answer"],
            sources=[SourceDocument(**s) for s in result.get("sources", [])],
            agent_trace=result.get("agent_trace", []),
            session_id=session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求出错: {str(e)}")


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    agent = get_agent()

    async def event_generator():
        async for chunk in agent.run_stream(query=request.query, session_id=session_id):
            # Token 级流式中 tokens 极小，直接原样输出（换行符由前端 markdown 渲染处理）
            # 转义换行符，避免破坏 SSE 格式，前端负责反转义
            safe_chunk = chunk.replace('\n', '__NL__').replace('\r', '__CR__')
            yield f"data: {safe_chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id,
        }
    )
