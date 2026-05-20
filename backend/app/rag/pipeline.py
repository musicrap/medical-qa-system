import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""RAG 模块 - 混合检索（向量+BM25关键词）+ Cross-Encoder 重排序"""
import json
import os as _os
import pickle
import threading
import numpy as np
from typing import List, Optional, Dict, Any, Generator
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, DashScopeEmbeddings
from langchain_core.documents import Document

from app.config import settings


def _tokenize(text: str) -> List[str]:
    """字符级分词，适配中文 BM25"""
    return list(text)


class RAGPipeline:
    def __init__(self):
        self.persist_dir = settings.CHROMA_PERSIST_DIR
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        self.embeddings = self._init_embeddings()
        self.vectorstore: Optional[Chroma] = None
        # BM25 关键词检索
        self.bm25_docs: List[Document] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_ready = threading.Event()
        self._bm25_cache_path = _os.path.join(self.persist_dir, "bm25_index.pkl")
        # Cross-Encoder 重排序（延迟加载）
        self.reranker: Optional[CrossEncoder] = None
        self._load_or_init_vectorstore()

    def _init_embeddings(self):
        provider = settings.EMBEDDING_PROVIDER
        if provider == "local":
            print(f"[RAG] Using local Embedding: {settings.LOCAL_EMBEDDING_MODEL}")
            return HuggingFaceEmbeddings(
                model_name=settings.LOCAL_EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        elif provider == "dashscope":
            print(f"[RAG] Using DashScope Embedding: {settings.EMBEDDING_MODEL}")
            return DashScopeEmbeddings(
                model=settings.EMBEDDING_MODEL
            )
        else:
            print(f"[RAG] Using OpenAI Embedding: {settings.EMBEDDING_MODEL}")
            return OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                openai_api_base=settings.OPENAI_API_BASE,
            )

    def _load_or_init_vectorstore(self):
        if _os.path.exists(self.persist_dir) and _os.listdir(self.persist_dir):
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
            count = self.vectorstore._collection.count()
            print(f"[RAG] 已加载向量库，文档数: {count}")
            self._start_async_bm25_build()
        else:
            self.vectorstore = None
            print("[RAG] 向量库为空，等待数据导入")

    def _start_async_bm25_build(self):
        """后台线程异步构建 BM25 索引，优先尝试加载磁盘缓存"""
        if _os.path.exists(self._bm25_cache_path):
            try:
                print("[RAG] 发现 BM25 缓存，尝试加载...")
                with open(self._bm25_cache_path, "rb") as f:
                    data = pickle.load(f)
                self.bm25_docs = data["docs"]
                self.bm25_index = BM25Okapi(data["tokenized_corpus"])
                self.bm25_ready.set()
                print(f"[RAG] BM25 缓存加载完成，文档数: {len(self.bm25_docs)}")
                return
            except Exception as e:
                print(f"[RAG] BM25 缓存加载失败: {e}，将在后台重新构建")
        threading.Thread(target=self._build_bm25_from_chroma, daemon=True).start()

    def _build_bm25_from_chroma(self):
        """从已有的 ChromaDB 中分批提取所有文档构建 BM25 索引（后台运行）"""
        if self.vectorstore is None:
            return
        try:
            print("[RAG] 后台开始构建 BM25 索引...")
            collection = self.vectorstore._collection
            total = collection.count()
            if total == 0:
                return
            batch_size = 5000
            docs = []
            for offset in range(0, total, batch_size):
                limit = min(batch_size, total - offset)
                result = collection.get(
                    include=["documents", "metadatas"],
                    limit=limit,
                    offset=offset
                )
                if result and result.get("documents"):
                    for text, meta in zip(result["documents"], result["metadatas"]):
                        doc = Document(page_content=text, metadata=meta or {})
                        docs.append(doc)
                if (offset + limit) % 20000 == 0 or (offset + limit) >= total:
                    print(f"[RAG] BM25 读取进度: {offset + limit}/{total}")
            if docs:
                self._build_bm25_index(docs)
                # 写入磁盘缓存
                tokenized_corpus = [_tokenize(doc.page_content) for doc in docs]
                cache_data = {"docs": docs, "tokenized_corpus": tokenized_corpus}
                with open(self._bm25_cache_path, "wb") as f:
                    pickle.dump(cache_data, f)
                print(f"[RAG] BM25 缓存已保存: {self._bm25_cache_path}")
            self.bm25_ready.set()
            print(f"[RAG] BM25 索引就绪，文档数: {len(docs)}")
        except Exception as e:
            print(f"[RAG] BM25 索引构建失败: {e}")

    def _build_bm25_index(self, docs: List[Document]):
        """构建 BM25 关键词索引"""
        self.bm25_docs = docs
        tokenized_corpus = [_tokenize(doc.page_content) for doc in docs]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.bm25_ready.set()
        print(f"[RAG] BM25 关键词索引构建完成，文档数: {len(docs)}")

    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """BM25 关键词检索"""
        if not self.bm25_ready.is_set() or self.bm25_index is None:
            return []
        tokenized_query = _tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.bm25_docs[idx]
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": min(1.0, float(scores[idx]) / 20.0),
                    "source": "keyword"
                })
        return results

    def _init_reranker(self):
        """后台加载 Cross-Encoder 重排序模型（非阻塞）"""
        if self.reranker is None and not getattr(self, "_reranker_loading", False):
            self._reranker_loading = True
            print(f"[RAG] 后台加载重排序模型: {settings.RERANKER_MODEL}")
            def _load():
                try:
                    self.reranker = CrossEncoder(
                        settings.RERANKER_MODEL,
                        max_length=512,
                        device="cpu"
                    )
                    print(f"[RAG] 重排序模型加载完成")
                except Exception as e:
                    print(f"[RAG] 重排序模型加载失败: {e}")
            threading.Thread(target=_load, daemon=True).start()

    def _rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """使用 Cross-Encoder 对候选文档重排序，未就绪时跳过"""
        if not candidates:
            return []
        try:
            self._init_reranker()
        except Exception:
            pass
        if self.reranker is None:
            candidates.sort(key=lambda x: x["score"], reverse=True)
            return candidates[:top_k]
        pairs = [(query, c["content"]) for c in candidates]
        raw = self.reranker.predict(pairs)
        scores = 1.0 / (1.0 + np.exp(-np.array(raw)))
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]

    def load_data(self, data_path: Optional[str] = None) -> List[Document]:
        path = data_path or settings.DATA_PATH
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")
        documents = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    src = record.get("src", [])
                    tgt = record.get("tgt", [])
                    if src and tgt:
                        question = "".join(src) if isinstance(src, list) else src
                        answer = "".join(tgt) if isinstance(tgt, list) else tgt
                        content = f"问题: {question}\n回答: {answer}"
                        doc = Document(
                            page_content=content,
                            metadata={"id": idx, "question": question, "answer": answer}
                        )
                        documents.append(doc)
                except json.JSONDecodeError:
                    continue
        print(f"[RAG] 从数据文件加载了 {len(documents)} 条记录")
        return documents

    def build_knowledge_base(self, data_path: Optional[str] = None, reset: bool = False) -> int:
        docs = self.load_data(data_path)
        if not docs:
            raise ValueError("无有效数据可用于构建知识库")
        print(f"[RAG] 直接存储 {len(docs)} 条记录（不分割）")
        if reset and _os.path.exists(self.persist_dir):
            import shutil
            shutil.rmtree(self.persist_dir)
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
        )
        count = self.vectorstore._collection.count()
        print(f"[RAG] 知识库构建完成，向量数: {count}")
        self._build_bm25_index(docs)
        # 保存 BM25 磁盘缓存
        tokenized_corpus = [_tokenize(doc.page_content) for doc in docs]
        cache_data = {"docs": docs, "tokenized_corpus": tokenized_corpus}
        with open(self._bm25_cache_path, "wb") as f:
            pickle.dump(cache_data, f)
        print(f"[RAG] BM25 缓存已保存")
        return count

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        混合检索：向量语义检索 + BM25 关键词检索 → 合并去重 → Cross-Encoder 重排序 → Top-3
        流程：
        1. 向量检索 top-10（语义匹配）
        2. BM25 关键词检索 top-10（精确关键词匹配）
        3. 合并去重（最多 20 条候选）
        4. Cross-Encoder 重排序
        5. 返回最终 top-3 结果
        """
        if self.vectorstore is None:
            return []

        vector_k = settings.HYBRID_VECTOR_K
        keyword_k = settings.HYBRID_KEYWORD_K
        final_k = top_k or settings.RERANK_TOP_K

        # 1. 向量检索（语义匹配）
        vector_results = []
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=vector_k)
        for doc, score in docs_with_scores:
            vector_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": 1.0 / (1.0 + max(float(score), 1e-8)),
                "source": "vector"
            })

        # 2. BM25 关键词检索（精确匹配）
        keyword_results = self._keyword_search(query, top_k=keyword_k)

        # 3. 合并去重（基于 content 去重，优先保留向量检索结果）
        merged = {r["content"]: r for r in keyword_results}
        for r in vector_results:
            if r["content"] not in merged:
                merged[r["content"]] = r
            else:
                merged[r["content"]]["source"] = "hybrid"
                merged[r["content"]]["score"] = max(merged[r["content"]]["score"], r["score"])

        candidates = list(merged.values())
        print(f"[RAG] 混合检索: 向量命中{len(vector_results)} + BM25命中{len(keyword_results)} → 去重合入{len(candidates)}条候选")

        # 4. Cross-Encoder 重排序
        if len(candidates) > final_k:
            candidates = self._rerank(query, candidates, final_k)
            # 将 rerank_score 同步到 score 字段（CrossEncoder 输出已在 0-1 范围）
            for c in candidates:
                c["score"] = c.get("rerank_score", c["score"])
        else:
            candidates.sort(key=lambda x: x["score"], reverse=True)
            candidates = candidates[:final_k]
        # 5. 对最终结果做 min-max 归一化，使相关性分数分布更合理
        if len(candidates) > 1:
            scores = [c["score"] for c in candidates]
            min_s, max_s = min(scores), max(scores)
            if max_s > min_s:
                for c in candidates:
                    c["score"] = 0.3 + 0.7 * (c["score"] - min_s) / (max_s - min_s)
            else:
                for c in candidates:
                    c["score"] = 0.5
        elif len(candidates) == 1:
            candidates[0]["score"] = 0.5

        print(f"[RAG] 重排序后返回 Top-{len(candidates)} 结果")
        return candidates

    def retrieve_vector_only(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """????????? BM25?????????????"""
        if self.vectorstore is None:
            return []
        results = []
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
        for doc, score in docs_with_scores:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": 1.0 / (1.0 + max(float(score), 1e-8)),
                "source": "vector_only"
            })
        return results

    def retrieve_hybrid_no_rerank(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """?? + BM25 ?????? Cross-Encoder ???????????"""
        if self.vectorstore is None:
            return []

        vector_k = settings.HYBRID_VECTOR_K
        keyword_k = settings.HYBRID_KEYWORD_K

        # ????
        vector_results = []
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=vector_k)
        for doc, score in docs_with_scores:
            vector_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": 1.0 / (1.0 + max(float(score), 1e-8)),
                "source": "vector"
            })

        # BM25 ?????
        keyword_results = self._keyword_search(query, top_k=keyword_k)

        # ????
        merged = {r["content"]: r for r in keyword_results}
        for r in vector_results:
            if r["content"] not in merged:
                merged[r["content"]] = r
            else:
                merged[r["content"]]["source"] = "hybrid"
                merged[r["content"]]["score"] = max(merged[r["content"]]["score"], r["score"])

        candidates = list(merged.values())
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def load_once(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """仅当向量库为空时加载数据，已存在则跳过"""
        if self.vectorstore is not None:
            count = self.vectorstore._collection.count()
            if count > 0:
                return {"imported_count": count, "status": "skipped", "message": f"知识库已有 {count} 条记录，无需重复加载"}
        count = self.build_knowledge_base(data_path, reset=False)
        return {"imported_count": count, "status": "success", "message": f"成功加载 {count} 条向量记录到知识库"}

    def load_once_stream(self, data_path=None):
        if self.vectorstore is not None:
            count = self.vectorstore._collection.count()
            if count > 0:
                yield {"progress": 100, "stage": "skipped", "imported_count": count, "message": f"知识库已有 {count} 条记录，无需重复加载"}
                return

        path = data_path or settings.DATA_PATH
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")

        yield {"progress": 1, "stage": "loading", "message": "正在统计数据..."}
        with open(path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        documents = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx % 1000 == 0:
                    pct = min(15, 5 + int(idx / max(total_lines, 1) * 10))
                    yield {"progress": pct, "stage": "loading", "message": f"正在读取数据... ({idx}/{total_lines})"}
                try:
                    record = json.loads(line.strip())
                    src = record.get("src", [])
                    tgt = record.get("tgt", [])
                    if src and tgt:
                        question = "".join(src) if isinstance(src, list) else src
                        answer = "".join(tgt) if isinstance(tgt, list) else tgt
                        content = f"问题: {question}\n回答: {answer}"


                        doc = Document(
                            page_content=content,
                            metadata={"id": idx, "question": question, "answer": answer}
                        )
                        documents.append(doc)
                except json.JSONDecodeError:
                    continue

        if not documents:
            raise ValueError("无有效数据")

        yield {"progress": 18, "stage": "embedding", "message": f"数据加载完成，共 {len(documents)} 条记录，开始向量化..."}

        import shutil
        if _os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)

        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )

        batch_size = 200
        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            done = min(i + batch_size, total)
            pct = min(95, 18 + int(done / total * 77))
            yield {
                "progress": pct,
                "stage": "embedding",
                "message": f"正在向量化... ({done}/{total})"
            }

        count = self.vectorstore._collection.count()
        self._build_bm25_index(documents)
        tokenized_corpus = [_tokenize(doc.page_content) for doc in documents]
        cache_data = {"docs": documents, "tokenized_corpus": tokenized_corpus}
        with open(self._bm25_cache_path, "wb") as f:
            pickle.dump(cache_data, f)
        yield {
            "progress": 100,
            "stage": "done",
            "imported_count": count,
            "message": f"加载完成，共 {count} 条向量记录"
        }

    def get_status(self) -> Dict[str, Any]:
        import os as _os2
        data_name = _os2.path.basename(settings.DATA_PATH)
        if self.vectorstore is None:
            return {"total_documents": 0, "collection_name": self.collection_name, "status": "empty", "data_file": data_name}
        count = self.vectorstore._collection.count()
        return {"total_documents": count, "collection_name": self.collection_name, "status": "ready", "data_file": data_name}

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return self.retrieve(query, top_k)


rag_pipeline: Optional[RAGPipeline] = None


def get_rag() -> RAGPipeline:
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
    return rag_pipeline
