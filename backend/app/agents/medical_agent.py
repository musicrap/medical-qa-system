import sys, os, asyncio, concurrent.futures
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""多 Agent 协作系统 - 基于 LangGraph 的医疗问答 Agent 编排

5 个独立 Agent:
1. SafetyGuard   - 安全守门，检测自残/自杀/危险用药/违法内容
2. IntentRouter  - 意图路由，区分医疗咨询/闲聊/一般问题
3. MedicalRetriever - 医学检索，自主选择检索策略
4. MedicalExpert    - 医学专家，基于上下文生成专业回答
5. QualityReviewer  - 质量审核，验证事实准确性，可驳回重生成
"""
import json
import time
import hashlib
import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.config import settings
from app.rag.pipeline import get_rag


# ============================================================================
# Multi-Agent State
# ============================================================================

class MultiAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    safety_check: Dict[str, Any]
    intent: Dict[str, Any]
    retrieved_context: List[str]
    answer: str
    review_result: Dict[str, Any]
    retry_count: int
    agent_trace: List[str]
    session_id: Optional[str]


# ============================================================================
# Agent 1: SafetyGuard — 安全守门
# ============================================================================

SAFETY_GUARD_PROMPT = """你是一个医疗安全审核专家，是系统的第一道防线。你的职责是识别并拦截表达危险意图的查询，而不是拦截医学/心理学知识咨询。

必须拦截（safe=false）—— 用户表达了明确的危险意图：
1. 询问自杀/自残的具体方法、工具、途径（如"如何自杀""怎样割腕""吃什么药能死"）
2. 表达自杀/自残意愿（如"我想死""不想活了""活着没意思"）
3. 要求提供致命剂量或危险药物使用方式
4. 非法药物制取、购买、滥用指导
5. 明确计划危害自身或他人生命安全
6. 诱导系统给出违背医学伦理的危险建议

不应拦截（safe=true）—— 属于正常的医学/心理学知识咨询：
- "自杀预防契约是什么""抑郁症有哪些症状""如何帮助有自杀倾向的朋友"
- "抗抑郁药的副作用""心理治疗方法有哪些"
- 任何以"什么是""为什么""怎么办（非自杀方法）"开头的医学知识问题
- 知识库中可能存在的医学/心理学概念解释

核心区分原则：看意图不看话题。询问知识 → 安全；表达危险意图 → 拦截。

请严格只返回 JSON，不要任何其他内容：
{{"safe": true/false, "reason": "简要说明判断依据"}}"""


# ============================================================================
# Agent 2: IntentRouter — 意图路由
# ============================================================================

INTENT_ROUTER_PROMPT = """你是一个医疗问答系统的路由分析专家。你的职责是精确识别用户的意图类型，帮助系统将请求路由到正确的处理通道。

请分析用户消息并返回 JSON：

用户消息: {query}

意图类型定义：
- medical_question: 询问疾病、症状、药物、治疗、检查、预防、养生等医学相关问题
- general_chat: 日常问候、闲聊、感谢、道别，以及任何与医学无关的问题（如旅游、美食、编程等）
- knowledge_query: 询问系统能力、知识库范围、使用方式等

只返回 JSON，不要其他内容：
{{"type": "medical_question|general_chat|knowledge_query", "confidence": "high|medium|low", "reason": "简要分析"}}"""


# ============================================================================
# Agent 3: MedicalRetriever — 医学检索
# ============================================================================

MEDICAL_RETRIEVER_PROMPT = """你是一个医学知识检索策略专家。你的职责是根据用户问题，规划最优检索策略并返回结构化结果。

用户问题: {query}

检索到的相关文档：
{raw_results}

请完成以下工作：
1. 评估上述文档与问题的相关性
2. 筛选出真正有用的文档（去重、去噪）
3. 按相关性排序并摘要
4. 判断是否需要更精确的检索关键词

返回 JSON：
{{
  "relevant_docs": [{{"content": "...", "relevance": "high|medium|low", "reason": "..."}}],
  "need_refine": true/false,
  "refine_keywords": ["关键词1", "关键词2"],
  "summary": "检索结果总结"
}}"""


# ============================================================================
# Agent 4: MedicalExpert — 医学专家
# ============================================================================

MEDICAL_EXPERT_PROMPT = """你是一位资深医学专家，拥有丰富的临床经验和医学知识。你的职责是基于参考资料为用户提供专业、准确、负责任的医疗建议。

核心原则：
1. **基于证据**：回答必须立足于提供的参考资料，不可凭空编造
2. **诚实透明**：如果参考资料不足以回答问题，如实说明并给出通用建议
3. **风险提示**：每个回答必须包含免责声明
4. **专业易懂**：用专业但通俗的语言解释医学术语
5. **安全第一**：绝不推荐未经临床验证的疗法或危险行为

回答结构要求：
- 先给出核心答案
- 如有参考资料，标注来源
- 如有不确定之处，明确指出
- 末尾必须包含：⚠️ 以上内容仅供参考，不能替代专业医疗诊断。如有健康问题，请及时就医。

参考资料：
{context}

审核反馈（如有）：
{review_feedback}"""


# ============================================================================
# Agent 5: QualityReviewer — 质量审核
# ============================================================================

QUALITY_REVIEWER_PROMPT = """你是一位严格的医学内容质量审核专家。你的职责是审核 AI 生成的医疗回答，确保其专业、准确、安全。

审核维度：
1. **事实准确性**：回答与参考资料是否一致？有无编造内容？
2. **完整性**：是否遗漏了关键信息？是否充分回答了用户问题？
3. **安全性**：是否包含免责声明？有无危险建议？
4. **可读性**：表述是否清晰？专业术语有无解释？
5. **伦理性**：是否避免歧视性表述？是否尊重患者隐私？

用户问题: {query}
参考资料: {context}
待审核回答: {answer}

请严格只返回 JSON：
{{
  "valid": true/false,
  "score": 1-10,
  "issues": ["问题1", "问题2"],
  "critical_failure": true/false,
  "suggestion": "具体修改建议（如果 answer 不合格，给出明确的改进方向）"
}}

判定标准：
- score >= 7 且无关键失败 → valid=true，直接通过
- score < 7 或有任何关键失败 → valid=false，需重生成
- 关键失败包括：编造医学事实、缺少免责声明、有危险建议"""


# ============================================================================
# LLM 工厂 — 每个 Agent 使用独立实例
# ============================================================================

class AgentLLMFactory:
    """为每个 Agent 创建独立的 LLM 实例，可配置不同参数"""

    @staticmethod
    def create(temperature: float = 0.3, max_tokens: int = 2048) -> ChatOpenAI:
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
        )

    @staticmethod
    def safety_guard() -> ChatOpenAI:
        return AgentLLMFactory.create(temperature=0.0, max_tokens=256)  # ???????JSON

    @staticmethod
    def intent_router() -> ChatOpenAI:
        return AgentLLMFactory.create(temperature=0.1, max_tokens=256)  # ???JSON

    @staticmethod
    def medical_retriever() -> ChatOpenAI:
        return AgentLLMFactory.create(temperature=0.1, max_tokens=512)  # ????JSON

    @staticmethod
    def medical_expert() -> ChatOpenAI:
        return AgentLLMFactory.create(temperature=0.5)  # 稍高温度增加回答多样性

    @staticmethod
    def quality_reviewer() -> ChatOpenAI:
        return AgentLLMFactory.create(temperature=0.0)  # 审核要严格一致

    @staticmethod
    def general_chat() -> ChatOpenAI:
        return AgentLLMFactory.create(temperature=0.7)  # 通用闲聊


# ============================================================================
# JSON 解析辅助
# ============================================================================

def _parse_json(text: str) -> Dict[str, Any]:
    """从 LLM 响应中提取 JSON"""
    content = text.strip()
    if "`" in content:
        content = content.split("`")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


# ============================================================================
# Multi-Agent 系统
# ============================================================================

MAX_RETRY = 2  # Expert-Reviewer 循环最大重试次数
MAX_HISTORY_TOKENS = 3000  # 上下文压缩阈值（token 估算：1 中文字符 ≈ 2 tokens）
CACHE_TTL_SECONDS = 1800  # 记忆缓存过期时间（30分钟）
CACHE_MAX_SIZE = 200  # 最大缓存条目数 循环最大重试次数


class MultiAgentSystem:
    def __init__(self):
        self.rag = get_rag()
        # 每个 Agent 使用独立 LLM 实例
        self.safety_llm = AgentLLMFactory.safety_guard()
        self.router_llm = AgentLLMFactory.intent_router()
        self.retriever_llm = AgentLLMFactory.medical_retriever()
        self.expert_llm = AgentLLMFactory.medical_expert()
        self.reviewer_llm = AgentLLMFactory.quality_reviewer()
        self.general_llm = AgentLLMFactory.general_chat()
        # 流式 LLM 实例（temperature 与对应 Agent 一致）
        self.stream_expert_llm = AgentLLMFactory.medical_expert()
        self.stream_general_llm = AgentLLMFactory.general_chat()
        self.graph = self._build_graph()
        # 记忆缓存：{question_hash: {answer, sources, timestamp}}
        self._answer_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = __import__('threading').Lock()
        # retrieval_mode: full | vector_only | hybrid_no_rerank
        self.retrieval_mode = 'full'

    # ========================================================================
    # Graph 构建
    # ========================================================================

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(MultiAgentState)

        workflow.add_node("safety_guard", self._safety_guard_node)
        workflow.add_node("intent_router", self._intent_router_node)
        workflow.add_node("medical_retriever", self._medical_retriever_node)
        workflow.add_node("medical_expert", self._medical_expert_node)
        workflow.add_node("quality_reviewer", self._quality_reviewer_node)
        workflow.add_node("blocked_response", self._blocked_response_node)
        workflow.add_node("fallback_response", self._fallback_response_node)
        workflow.add_node("general_chat", self._general_chat_node)

        workflow.set_entry_point("safety_guard")

        # SafetyGuard → 安全则路由，危险则拦截
        workflow.add_conditional_edges(
            "safety_guard", self._after_safety,
            {"router": "intent_router", "blocked": "blocked_response"}
        )

        # IntentRouter → 医疗/闲聊/其他
        workflow.add_conditional_edges(
            "intent_router", self._after_router,
            {"retrieve": "medical_retriever", "general_chat": "general_chat", "fallback": "fallback_response"}
        )

        workflow.add_edge("medical_retriever", "medical_expert")

        # Expert → Reviewer（循环入口）
        workflow.add_conditional_edges(
            "medical_expert", self._after_expert,
            {"review": "quality_reviewer", "done": END}
        )

        # Reviewer → 通过则结束，驳回则回到 Expert（带反馈）
        workflow.add_conditional_edges(
            "quality_reviewer", self._after_reviewer,
            {"regenerate": "medical_expert", "done": END}
        )

        workflow.add_edge("blocked_response", END)
        workflow.add_edge("fallback_response", END)
        workflow.add_edge("general_chat", END)

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    # ========================================================================
    # Node: SafetyGuard
    # ========================================================================

    def _safety_guard_node(self, state: MultiAgentState) -> MultiAgentState:
        trace = list(state.get("agent_trace", []))
        trace.append("[SafetyGuard] 开始安全检查")
        query = state["query"]

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", SAFETY_GUARD_PROMPT),
                ("human", "请检查以下用户输入: {query}"),
            ])
            chain = prompt | self.safety_llm
            response = chain.invoke({"query": query})
            result = _parse_json(response.content)
        except Exception as e:
            result = {"safe": False, "reason": f"安全检查异常，出于安全考虑拦截: {e}"}

        is_safe = result.get("safe", False)
        trace.append(f"[SafetyGuard] 结果: {'安全' if is_safe else '危险'} - {result.get('reason', '')}")

        return {
            **state,
            "safety_check": result,
            "agent_trace": trace,
        }

    def _after_safety(self, state: MultiAgentState) -> str:
        if state.get("safety_check", {}).get("safe", True):
            return "router"
        return "blocked"

    # ========================================================================
    # Node: IntentRouter
    # ========================================================================

    def _intent_router_node(self, state: MultiAgentState) -> MultiAgentState:
        trace = list(state.get("agent_trace", []))
        trace.append("[IntentRouter] 开始意图分析")
        query = state["query"]

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", INTENT_ROUTER_PROMPT),
            ])
            chain = prompt | self.router_llm
            response = chain.invoke({"query": query})
            result = _parse_json(response.content)
        except Exception as e:
            result = {"type": "medical_question", "confidence": "low", "reason": f"解析失败: {e}"}

        intent_type = result.get("type", "medical_question")
        trace.append(f"[IntentRouter] 分类: {intent_type} (置信度: {result.get('confidence', 'unknown')})")

        return {
            **state,
            "intent": result,
            "agent_trace": trace,
        }

    def _after_router(self, state: MultiAgentState) -> str:
        intent_type = state.get("intent", {}).get("type", "")
        if intent_type == "medical_question":
            return "retrieve"
        elif intent_type == "general_chat":
            return "general_chat"
        else:
            return "fallback"

    # ========================================================================
    # Node: MedicalRetriever
    # ========================================================================

    def _medical_retriever_node(self, state: MultiAgentState) -> MultiAgentState:
        trace = list(state.get("agent_trace", []))
        trace.append("[MedicalRetriever] 开始知识库检索")
        query = state["query"]

        try:
            if self.retrieval_mode == "vector_only":
                raw_results = self.rag.retrieve_vector_only(query, top_k=settings.TOP_K_RETRIEVAL)
            elif self.retrieval_mode == "hybrid_no_rerank":
                raw_results = self.rag.retrieve_hybrid_no_rerank(query, top_k=settings.TOP_K_RETRIEVAL)
            else:
                raw_results = self.rag.retrieve(query, top_k=settings.TOP_K_RETRIEVAL)
            raw_texts = [r["content"] for r in raw_results]
            trace.append(f"[MedicalRetriever] 原始检索到 {len(raw_texts)} 条")

            # 直接使用 Cross-Encoder 已排序的 RAG 结果，跳过冗余 LLM 精排
            context = raw_texts
            trace.append(f"[MedicalRetriever] 直接使用检索结果 {len(context)} 条")
        except Exception as e:
            trace.append(f"[MedicalRetriever] 检索出错: {e}")
            context = []

        return {
            **state,
            "retrieved_context": context,
            "agent_trace": trace,
        }

    # ========================================================================
    # Node: MedicalExpert
    # ========================================================================

    def _medical_expert_node(self, state: MultiAgentState) -> MultiAgentState:
        trace = list(state.get("agent_trace", []))
        retry = state.get("retry_count", 0)

        if retry > 0:
            trace.append(f"[MedicalExpert] 第 {retry} 次重生成（根据审核反馈修正）")
        else:
            trace.append("[MedicalExpert] 开始生成医学回答")

        query = state["query"]
        context = state.get("retrieved_context", [])
        review = state.get("review_result", {})

        context_text = "\n\n".join([
            f"「来源 {i+1}」\n{c}" for i, c in enumerate(context)
        ]) if context else "暂无相关医学参考资料"

        review_feedback = review.get("suggestion", "") if review else ""

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", MEDICAL_EXPERT_PROMPT),
                ("human", "用户问题: {query}"),
            ])
            chain = prompt | self.expert_llm
            response = chain.invoke({
                "context": context_text,
                "review_feedback": review_feedback or "无",
                "query": query,
            })
            answer = response.content
            trace.append("[MedicalExpert] 回答生成完成")
        except Exception as e:
            trace.append(f"[MedicalExpert] 生成出错: {e}")
            answer = f"抱歉，生成回答时出现错误：{e}"

        # 如果有审核反馈，清除旧的审核结果，避免死循环
        new_state = {**state, "answer": answer, "agent_trace": trace}
        if review:
            new_state["review_result"] = {}
        return new_state

    def _after_expert(self, state: MultiAgentState) -> str:
        intent_type = state.get("intent", {}).get("type", "")
        # 医疗问题需要审核，闲聊直接结束
        answer = state.get("answer", "")
        # 非医疗问题直接结束
        if intent_type != "medical_question":
            return "done"
        # 短回答且包含免责声明，跳过审核以加速
        if len(answer) < 500 and "仅供参考" in answer:
            return "done"
        return "review"

    # ========================================================================
    # Node: QualityReviewer
    # ========================================================================

    def _quality_reviewer_node(self, state: MultiAgentState) -> MultiAgentState:
        trace = list(state.get("agent_trace", []))
        trace.append("[QualityReviewer] 开始质量审核")

        query = state["query"]
        answer = state.get("answer", "")
        context = state.get("retrieved_context", [])

        context_text = "\n---\n".join(context) if context else "无参考资料"

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", QUALITY_REVIEWER_PROMPT),
            ])
            chain = prompt | self.reviewer_llm
            response = chain.invoke({
                "query": query,
                "context": context_text,
                "answer": answer,
            })
            result = _parse_json(response.content)
        except Exception as e:
            result = {"valid": True, "score": 7, "issues": [], "critical_failure": False, "suggestion": f"解析失败: {e}"}

        valid = result.get("valid", True)
        score = result.get("score", 7)
        trace.append(f"[QualityReviewer] 审核{'通过' if valid else '未通过'} (分数: {score}/10)")
        if not valid:
            trace.append(f"[QualityReviewer] 问题: {result.get('issues', [])}")

        return {
            **state,
            "review_result": result,
            "agent_trace": trace,
        }

    def _after_reviewer(self, state: MultiAgentState) -> str:
        review = state.get("review_result", {})
        retry = state.get("retry_count", 0)

        if review.get("valid", True):
            return "done"

        if retry >= MAX_RETRY:
            # 达到最大重试，直接输出（带警告）
            return "done"

        # 驳回重生成，增加重试计数
        state["retry_count"] = retry + 1
        return "regenerate"

    # ========================================================================
    # Node: BlockedResponse（安全拦截）
    # ========================================================================

    def _blocked_response_node(self, state: MultiAgentState) -> MultiAgentState:
        trace = list(state.get("agent_trace", []))
        trace.append("[Blocked] 内容被安全拦截")
        reason = state.get("safety_check", {}).get("reason", "")

        answer = f"""⚠️ 您的请求因安全原因被拦截。

原因: {reason}

如果您正在经历困难或危机，请立即寻求帮助：
- 全国心理援助热线：400-161-9995
- 北京心理危机研究与干预中心：010-82951332
- 或直接拨打 120 / 110

您不是一个人，有人愿意帮助您。"""

        return {
            **state,
            "answer": answer,
            "agent_trace": trace,
        }

    # ========================================================================
    # Node: GeneralChat（通用闲聊）
    # ========================================================================

    def _general_chat_node(self, state: MultiAgentState) -> MultiAgentState:
        trace = list(state.get("agent_trace", []))
        trace.append("[GeneralChat]")
        query = state["query"]
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是友好有帮助的AI助手，用中文自然回答用户问题。"),
                ("human", "{query}"),
            ])
            chain = prompt | self.general_llm
            response = chain.invoke({"query": query})
            answer = response.content
        except Exception as e:
            trace.append(f"[GeneralChat] Error: {e}")
            answer = f": {e}"
        return {**state, "answer": answer, "agent_trace": trace}

    # ========================================================================
    # Node: FallbackResponse（兜底回复）
    # ========================================================================

    def _fallback_response_node(self, state: MultiAgentState) -> MultiAgentState:
        trace = list(state.get("agent_trace", []))
        trace.append("[Fallback] 使用兜底回复")
        query = state["query"]

        answer = f"""您好！我是医疗知识问答助手。

关于您的问题「{query}」，建议如下：
- 如需了解医学知识，请明确描述您的症状或疑问
- 如需查询系统功能，可以说"你能做什么"
- 如需闲聊，欢迎随时交流

⚠️ 本系统仅供参考，不能替代专业医疗诊断。如有健康问题，请及时就医。"""

        return {
            **state,
            "answer": answer,
            "agent_trace": trace,
        }

    # ========================================================================
    # 公共 API
    # ========================================================================

    # ========================================================================
    # 记忆缓存
    # ========================================================================

    def _cache_key(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode('utf-8')).hexdigest()

    def _cache_get(self, query: str) -> Optional[Dict[str, Any]]:
        key = self._cache_key(query)
        with self._cache_lock:
            if key in self._answer_cache:
                entry = self._answer_cache[key]
                if time.time() - entry["timestamp"] < CACHE_TTL_SECONDS:
                    return entry["data"]
                else:
                    del self._answer_cache[key]
        return None

    def _cache_set(self, query: str, data: Dict[str, Any]):
        key = self._cache_key(query)
        with self._cache_lock:
            if len(self._answer_cache) >= CACHE_MAX_SIZE:
                oldest = min(self._answer_cache, key=lambda k: self._answer_cache[k]["timestamp"])
                del self._answer_cache[oldest]
            self._answer_cache[key] = {"data": data, "timestamp": time.time()}

    # ========================================================================
    # 上下文压缩
    # ========================================================================

    def _estimate_tokens(self, text: str) -> int:
        return len(text) * 2

    def _get_history_context(self, session_id: str) -> str:
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = self.graph.get_state(config)
            if state and state.values:
                msgs = state.values.get("messages", [])
                if msgs:
                    return self._summarize_history(msgs)
        except Exception:
            pass
        return ""

    def _summarize_history(self, messages: list) -> str:
        total_tokens = sum(self._estimate_tokens(m.content if hasattr(m, 'content') else str(m)) for m in messages)
        if total_tokens <= MAX_HISTORY_TOKENS:
            return "\n".join([
                f"{'用户' if getattr(m, 'type', '') == 'human' else '助手'}: {m.content if hasattr(m, 'content') else str(m)}"
                for m in messages[-10:]
            ])
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "请用中文将以下对话历史压缩为一段简洁摘要，保留关键医学信息和用户关注点（不超过300字）："),
                ("human", "{history}"),
            ])
            history_text = "\n".join([
                f"{'用户' if getattr(m, 'type', '') == 'human' else '助手'}: {m.content if hasattr(m, 'content') else str(m)}"
                for m in messages[:-4]
            ])
            chain = prompt | self.general_llm
            response = chain.invoke({"history": history_text})
            summary = response.content.strip()
            recent = "\n".join([
                f"{'用户' if getattr(m, 'type', '') == 'human' else '助手'}: {m.content if hasattr(m, 'content') else str(m)}"
                for m in messages[-4:]
            ])
            return f"[对话历史摘要]\n{summary}\n\n[最近对话]\n{recent}"
        except Exception:
            return ""

    def run(self, query: str, history: Optional[List] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """非流式执行，含记忆缓存 + 上下文压缩"""
        # === 记忆缓存：相同问题直接复用 ===
        cached = self._cache_get(query)
        if cached:
            cached["from_cache"] = True
            cached["agent_trace"] = ["[Memory] 命中缓存，直接返回历史回答"]
            return cached

        thread_id = session_id or "default"
        config = {"configurable": {"thread_id": thread_id}}

        # === 上下文压缩：加载并压缩历史对话 ===
        history_context = self._get_history_context(thread_id)

        initial_state: MultiAgentState = {
            "messages": [],
            "query": query,
            "safety_check": {},
            "intent": {},
            "retrieved_context": [],
            "answer": "",
            "review_result": {},
            "retry_count": 0,
            "agent_trace": [],
            "session_id": session_id,
        }

        result = self.graph.invoke(initial_state, config)

        # 收集来源（仅医疗问题且安全通过时才检索）
        sources = []
        intent_type = result.get("intent", {}).get("type", "")
        if result.get("safety_check", {}).get("safe", True) and intent_type == "medical_question":
            try:
                retrieved = self.rag.retrieve(query)
                for r in retrieved:
                    sources.append({
                        "content": r["content"][:200],
                        "score": r["score"],
                        "metadata": r.get("metadata", {}),
                    })
            except Exception:
                pass

        response = {
            "answer": result.get("answer", "抱歉，暂时无法回答您的问题。"),
            "sources": sources,
            "agent_trace": result.get("agent_trace", []),
            "safety_check": result.get("safety_check", {}),
            "review": result.get("review_result", {}),
            "from_cache": False,
        }

        # === 写入记忆缓存 ===
        self._cache_set(query, response)

        return response
    async def run_stream(self, query: str, session_id: Optional[str] = None):
        """真正的 Token 级流式输出，含记忆缓存 + 上下文压缩"""
        thread_id = session_id or "default"

        # === 记忆缓存：相同问题直接流式输出缓存结果 ===
        cached = self._cache_get(query)
        if cached:
            yield "[来自历史记录] "
            answer = cached.get("answer", "")
            chunk_size = 30
            for i in range(0, len(answer), chunk_size):
                yield answer[i:i+chunk_size]
                await asyncio.sleep(0.01)
            import json as _json
            yield "__META__" + _json.dumps({
                "sources_count": len(cached.get("sources", [])),
                "agent_trace": ["[Memory] 命中缓存，直接返回历史回答"],
                "safety": {"safe": True},
                "review": {},
                "from_cache": True,
            }, ensure_ascii=False)
            return

        config = {"configurable": {"thread_id": thread_id}}

        initial_state: MultiAgentState = {
            "messages": [],
            "query": query,
            "safety_check": {},
            "intent": {},
            "retrieved_context": [],
            "answer": "",
            "review_result": {},
            "retry_count": 0,
            "agent_trace": [],
            "session_id": session_id,
        }

        # === ???? graph????????????? ===
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            final_state = await loop.run_in_executor(executor, self.graph.invoke, initial_state, config)

        answer = final_state.get("answer", "??????????????")
        full_answer = answer
        # ????????????
        chunk_size = 25
        for i in range(0, len(answer), chunk_size):
            yield answer[i:i+chunk_size]
            await asyncio.sleep(0.02)


        # 元数据（仅医疗问题且安全通过时才检索）
        sources = []
        if final_state:
            intent_type = final_state.get("intent", {}).get("type", "")
            if final_state.get("safety_check", {}).get("safe", True) and intent_type == "medical_question":
                try:
                    retrieved = self.rag.retrieve(query)
                    sources = [{"content": r["content"][:200], "score": r["score"]} for r in retrieved]
                except Exception:
                    pass

        # === 写入记忆缓存 ===
        if full_answer:
            self._cache_set(query, {
                "answer": full_answer,
                "sources": sources,
                "agent_trace": final_state.get("agent_trace", []) if final_state else [],
                "safety_check": final_state.get("safety_check", {}) if final_state else {},
                "review": final_state.get("review_result", {}) if final_state else {},
                "from_cache": False,
            })

        import json as _json
        yield "__META__" + _json.dumps({
            "sources_count": len(sources),
            "agent_trace": final_state.get("agent_trace", []) if final_state else [],
            "safety": final_state.get("safety_check", {}) if final_state else {},
            "review": final_state.get("review_result", {}) if final_state else {},
            "from_cache": False,
        }, ensure_ascii=False)
# ============================================================================
# 全局单例
# ============================================================================

agent_system: Optional[MultiAgentSystem] = None


def get_agent() -> MultiAgentSystem:
    global agent_system
    if agent_system is None:
        agent_system = MultiAgentSystem()
    return agent_system




