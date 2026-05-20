import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json as _json
from app.models.schemas import KnowledgeBaseStatus, DataImportRequest, DataImportResponse
from app.rag.pipeline import get_rag

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.get("/status", response_model=KnowledgeBaseStatus)
async def knowledge_status():
    rag = get_rag()
    status = rag.get_status()
    return KnowledgeBaseStatus(**status)


@router.post("/import", response_model=DataImportResponse)
async def import_data(request: DataImportRequest):
    try:
        rag = get_rag()
        count = rag.build_knowledge_base(
            data_path=request.file_path,
            reset=request.reset,
        )
        return DataImportResponse(
            imported_count=count,
            status="success",
            message=f"成功导入 {count} 条向量记录到知识库",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导入数据失败: {str(e)}")


@router.post("/load")
async def load_data():
    import asyncio, concurrent.futures
    rag = get_rag()
    loop = asyncio.get_running_loop()

    async def event_stream():
        queue = asyncio.Queue()

        def run_sync():
            try:
                for event in rag.load_once_stream():
                    loop.call_soon_threadsafe(queue.put_nowait, event)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, {"error": str(e)})
                loop.call_soon_threadsafe(queue.put_nowait, None)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop.run_in_executor(executor, run_sync)

        while True:
            event = await queue.get()
            if event is None:
                break
            line = "data: " + _json.dumps(event, ensure_ascii=False) + "\n\n"
            yield line

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@router.get("/search")
async def search_knowledge(query: str, top_k: int = 5):
    rag = get_rag()
    results = rag.search(query, top_k)
    return {"query": query, "results": results, "count": len(results)}

