# 医疗知识问答系统

基于 **RAG（检索增强生成）+ 多 Agent 协作** 的智能医疗知识问答平台。采用 LangGraph 编排 5 个专业 Agent，结合混合检索策略，提供精准、安全的医疗知识问答服务。

## 系统架构

```
用户提问
   │
   ▼
┌──────────────────────────────────────────────┐
│              Multi-Agent Pipeline             │
│                                               │
│  ① SafetyGuard    →  安全守门，拦截危险查询    │
│  ② IntentRouter   →  意图路由，识别问题类型    │
│  ③ MedicalRetriever → 医学检索，混合策略召回   │
│  ④ MedicalExpert  →  医学专家，生成专业回答    │
│  ⑤ QualityReviewer → 质量审核，事实准确性校验   │
│                                               │
└──────────────────────────────────────────────┘
   │                         │
   ▼                         ▼
┌──────────┐          ┌──────────────┐
│ ChromaDB │          │  BM25 关键词  │
│ 向量检索  │          │   检索        │
└──────────┘          └──────────────┘
```

## 功能特性

- **5 Agent 协作**：安全审核 → 意图路由 → 智能检索 → 专业回答 → 质量审核，全链路自动化
- **混合检索**：向量检索（ChromaDB）+ BM25 关键词检索 + BGE Reranker 重排序
- **安全优先**：第一道防线自动拦截自残、自杀、危险用药等高风险查询
- **流式输出**：支持 SSE 流式响应，逐 Token 返回回答
- **语义缓存**：相同问题命中缓存，秒级响应
- **Docker 一键部署**：前后端容器化，`docker-compose up -d` 即可启动

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| Agent 编排 | LangGraph |
| LLM 调用 | LangChain + OpenAI API |
| 向量数据库 | ChromaDB |
| Embedding | Sentence-Transformers / text2vec-base-chinese |
| 关键词检索 | rank-bm25 |
| 重排序 | BGE-Reranker-v2-m3 |
| 前端 | Vue 3 + Vite |
| 部署 | Docker + Docker Compose |

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/musicrap/medical-qa-system.git
cd medical-qa-system
```

### 2. 配置环境变量

复制 `.env.example`（或自行创建 `.env`）：

```env
OPENAI_API_KEY=your-api-key
OPENAI_API_BASE=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=shibing624/text2vec-base-chinese

CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=medical_knowledge
DATA_PATH=./data/train-sft.jsonl
```

### 3. Docker 部署（推荐）

```bash
docker-compose up -d
```

启动后访问：
- 前端界面：http://localhost:5173
- API 文档：http://localhost:8000/docs

### 4. 本地开发

**后端：**

```bash
cd backend
pip install -r requirements.txt
python -m app.main
```

**前端：**

```bash
cd frontend
npm install
npm run dev
```

## 知识库初始化

系统首次启动会自动加载 `data/` 目录下的 JSONL 数据，构建向量索引存入 `chroma_db/`。

数据格式（每行一条 JSON）：

```json
{"id": "001", "title": "高血压常见症状", "content": "高血压的典型症状包括..."}
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 系统信息 |
| GET | `/api/health` | 健康检查 |
| POST | `/api/chat` | 问答对话（SSE 流式） |
| GET | `/api/knowledge/status` | 知识库状态 |
| POST | `/api/knowledge/reload` | 重建知识库 |

## 项目结构

```
medical-qa-system/
├── backend/
│   ├── app/
│   │   ├── agents/         # 5 个 Agent 定义
│   │   ├── rag/             # RAG 检索管道
│   │   ├── routers/         # API 路由
│   │   ├── models/          # 数据模型
│   │   ├── config.py        # 配置管理
│   │   └── main.py          # 应用入口
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── views/           # 页面组件
│   │   ├── api/             # API 调用
│   │   ├── stores/          # 状态管理
│   │   └── router/          # 路由配置
│   ├── Dockerfile
│   └── package.json
├── data/                    # 知识库数据
├── chroma_db/               # 向量数据库（自动生成）
├── docker-compose.yml
└── .env                     # 环境配置
```
