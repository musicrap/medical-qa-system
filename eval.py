
# -*- coding: utf-8 -*-
"""
医疗知识问答系统 — 消融实验评估脚本 (eval.py)

比较三种检索配置：
  A) vector_only      — 纯向量语义检索（基线）
  B) hybrid_no_rerank — 向量 + BM25 混合检索（无重排序）
  C) full             — 全链路：向量 + BM25 + Cross-Encoder 重排序

指标：
  1) Recall@K  (K=1, 3, 5)
  2) 回答准确率 (ROUGE-L, Char-F1)

用法：
  python eval.py --no-agent     # 仅评测检索 Recall@K（快速）
  python eval.py                # 默认前 50 条，含回答准确率
  python eval.py --samples 100  # 指定样本数
  python eval.py --all          # 全部 500 条
"""

import sys, os, json, time, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import settings
from app.rag.pipeline import get_rag
from app.agents.medical_agent import get_agent

TRAIN_PATH = PROJECT_ROOT / "data" / "train-sft.jsonl"

RETRIEVAL_CONFIGS = {
    "vector_only":      "纯向量检索",
    "hybrid_no_rerank": "向量+BM25(无重排)",
    "full":             "全链路(含重排序)",
}

K_VALUES = [1, 3, 5, 10]

# ============================================================
# 工具函数
# ============================================================

def sample_from_train(path: Path, n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """从训练集中随机抽样 n 条作为自检索验证集，保留原始行号作为 id"""
    import random
    random.seed(seed)
    # First count total lines
    with open(path, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)
    if n > total:
        n = total
    indices = set(random.sample(range(total), n))
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx not in indices:
                continue
            try:
                record = json.loads(line.strip())
                src = record.get("src", [])
                tgt = record.get("tgt", [])
                question = "".join(src) if isinstance(src, list) else src
                answer   = "".join(tgt) if isinstance(tgt, list) else tgt
                if question and answer:
                    data.append({"id": idx, "question": question, "answer": answer})
            except json.JSONDecodeError:
                continue
    return data


def char_ngrams(text: str, n: int = 3) -> set:
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def jaccard_similarity(a: str, b: str) -> float:
    set_a = char_ngrams(a, 3)
    set_b = char_ngrams(b, 3)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# is_relevant not needed for self-retrieval; ID matching is done in evaluate_retrieval

def rouge_l_score(pred: str, ref: str) -> float:
    """ROUGE-L F1"""
    if not pred or not ref:
        return 0.0
    m, n = len(pred), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == ref[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / m if m > 0 else 0
    r = lcs / n if n > 0 else 0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def char_f1(pred: str, ref: str) -> float:
    """字符级 F1"""
    if not pred or not ref:
        return 0.0
    sp, sr = set(pred), set(ref)
    if not sp or not sr:
        return 0.0
    inter = len(sp & sr)
    p = inter / len(sp)
    r = inter / len(sr)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


# ============================================================
# 数据类
# ============================================================

@dataclass
class RetrievalResult:
    recall_at_k: Dict[int, bool] = field(default_factory=dict)
    top_contents: List[str] = field(default_factory=list)


@dataclass
class AnswerResult:
    answer: str = ""
    rouge_l: float = 0.0
    char_f1: float = 0.0
    latency_ms: float = 0.0


@dataclass
class ConfigMetrics:
    name: str = ""
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    avg_retrieved: float = 0.0
    avg_rouge_l: float = 0.0
    avg_char_f1: float = 0.0
    avg_latency_ms: float = 0.0
    total_queries: int = 0


# ============================================================
# 评估核心
# ============================================================

def evaluate_retrieval(rag, data, config_name):
    """评估检索 Recall@K"""
    metrics = ConfigMetrics(name=config_name)
    results = []

    retrieve_fn = {
        "vector_only":      rag.retrieve_vector_only,
        "hybrid_no_rerank": rag.retrieve_hybrid_no_rerank,
        "full":             rag.retrieve,
    }[config_name]

    recall_hits = {k: 0 for k in K_VALUES}
    max_k = max(K_VALUES)
    total_docs = 0

    for idx, sample in enumerate(data):
        query = sample["question"]
        gt    = sample["answer"]

        retrieved = retrieve_fn(query, top_k=max_k)
        total_docs += len(retrieved)

        expected_id = sample.get("id")
        rr = RetrievalResult()
        rr.top_contents = [r["content"] for r in retrieved]

        for k in K_VALUES:
            top_k_docs = retrieved[:k]
            hit = any(d.get("metadata", {}).get("id") == expected_id for d in top_k_docs)
            rr.recall_at_k[k] = hit
            if hit:
                recall_hits[k] += 1

        results.append(rr)

        pct = (idx + 1) / len(data)
        bar_len = 30
        filled = int(bar_len * pct)
        bar = chr(9608) * filled + chr(9617) * (bar_len - filled)
        print(f"\r  [{config_name}] |{bar}| {idx+1}/{len(data)}", end="", flush=True)
        if idx < 3:
            top_ids = [d.get("metadata", {}).get("id", "?") for d in retrieved[:3]] if retrieved else []
            hit_flag = "HIT" if any(d.get("metadata", {}).get("id") == sample.get("id") for d in retrieved[:3]) else "MISS"
            print(f"  [DIAG #" + str(idx+1) + "] expected_id=" + str(sample.get("id")) + " top_ids=" + str(top_ids) + " -> " + hit_flag)

    n = len(data)
    for k in K_VALUES:
        metrics.recall_at_k[k] = recall_hits[k] / n if n > 0 else 0.0
    # Miss analysis: show samples where no doc was relevant
    misses = []
    for idx2, (s2, r2) in enumerate(zip(data, results)):
        if not any(r2.recall_at_k.get(k, False) for k in K_VALUES):
            misses.append((idx2, s2.get(chr(113)+chr(117)+chr(101)+chr(115)+chr(116)+chr(105)+chr(111)+chr(110), chr(34)+chr(34))[:60], s2.get(chr(97)+chr(110)+chr(115)+chr(119)+chr(101)+chr(114), chr(34)+chr(34))[:40]))
    if misses:
        print(f'  [{config_name}] All-K miss: {len(misses)}/{n}')
        for mi, mq, ma in misses[:3]:
            print(f'    Miss #' + str(mi+1) + ': Q=' + mq + '... | A=' + ma + '...')

    print()  # newline after progress bar
    metrics.avg_retrieved = total_docs / n if n > 0 else 0.0
    metrics.total_queries = n

    return metrics, results


def evaluate_answers(agent, data, config_name):
    """评估回答准确率"""
    metrics = ConfigMetrics(name=config_name)
    results = []

    agent.retrieval_mode = config_name
    agent._answer_cache = {}

    rouge_scores, f1_scores, latencies = [], [], []

    for idx, sample in enumerate(data):
        query = sample["question"]
        gt    = sample["answer"]

        t0 = time.time()
        try:
            resp = agent.run(query=query, session_id=f"eval_{config_name}_{idx}")
            pred = resp.get("answer", "")
        except Exception as e:
            print(f"  [{config_name}] 第{idx+1}条出错: {e}")
            pred = ""
        latency = (time.time() - t0) * 1000

        ar = AnswerResult(
            answer=pred,
            rouge_l=rouge_l_score(pred, gt),
            char_f1=char_f1(pred, gt),
            latency_ms=latency,
        )
        rouge_scores.append(ar.rouge_l)
        f1_scores.append(ar.char_f1)
        latencies.append(ar.latency_ms)
        results.append(ar)

        pct = (idx + 1) / len(data)
        bar_len = 30
        filled = int(bar_len * pct)
        bar = chr(9608) * filled + chr(9617) * (bar_len - filled)
        print(f"\r  [{config_name}] |{bar}| {idx+1}/{len(data)}", end="", flush=True)

    n = len(data)
    print()  # newline after progress bar
    print()  # newline after progress bar
    metrics.avg_rouge_l = sum(rouge_scores) / n if n > 0 else 0.0
    metrics.avg_char_f1 = sum(f1_scores) / n if n > 0 else 0.0
    metrics.avg_latency_ms = sum(latencies) / n if n > 0 else 0.0
    metrics.total_queries = n

    return metrics, results


# ============================================================
# 报告输出
# ============================================================

def print_report(retrieval_metrics, answer_metrics, total_samples, run_answers):
    CN = {
        "vector_only":      "纯向量检索",
        "hybrid_no_rerank": "向量+BM25(无重排)",
        "full":             "全链路(含重排序)",
    }

    print()
    print("=" * 72)
    print("  医疗知识问答系统 — 消融实验评估报告")
    print("=" * 72)
    print(f"  验证集样本数: {total_samples}")
    print(f"  评估时间:     {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ── Recall@K ──
    print("-" * 72)
    print("  [1] Recall@K — 检索命中率")
    print("-" * 72)
    header = f"  {'配置':<24s}"
    for k in K_VALUES:
        header += f" {f'Recall@{k}':>12s}"
    header += f" {'平均检索数':>12s}"
    print(header)
    print("  " + "-" * 68)

    baseline = retrieval_metrics.get("vector_only")
    for name in ["vector_only", "hybrid_no_rerank", "full"]:
        m = retrieval_metrics.get(name)
        if not m:
            continue
        row = f"  {CN[name]:<24s}"
        for k in K_VALUES:
            val = m.recall_at_k.get(k, 0.0)
            marker = ""
            if baseline and name != "vector_only":
                base_val = baseline.recall_at_k.get(k, 0.0)
                marker = " +" if val > base_val else (" -" if val < base_val else "")
            row += f" {format_pct(val)+marker:>12s}"
        row += f" {m.avg_retrieved:>11.1f}"
        print(row)

    # ── Answer Accuracy ──
    if run_answers and answer_metrics:
        print()
        print("-" * 72)
        print("  [2] 回答准确率 — 生成质量对比")
        print("-" * 72)
        header = f"  {'配置':<24s} {'ROUGE-L':>10s} {'Char-F1':>10s} {'平均延迟':>14s}"
        print(header)
        print("  " + "-" * 60)

        for name in ["vector_only", "hybrid_no_rerank", "full"]:
            m = answer_metrics.get(name)
            if not m:
                continue
            row = f"  {CN[name]:<24s} {format_pct(m.avg_rouge_l):>10s} {format_pct(m.avg_char_f1):>10s} {m.avg_latency_ms:>11.0f} ms"
            print(row)

    # ── 结论 ──
    print()
    print("-" * 72)
    print("  [3] 分析结论")
    print("-" * 72)

    full_recall = retrieval_metrics.get("full")
    vec_recall  = retrieval_metrics.get("vector_only")
    if full_recall and vec_recall:
        for k in K_VALUES:
            rk_full = full_recall.recall_at_k.get(k, 0)
            rk_vec  = vec_recall.recall_at_k.get(k, 0)
            gain = (rk_full - rk_vec) / max(rk_vec, 0.001) * 100
            print(f"  Recall@{k} 提升: {format_pct(rk_vec)} -> {format_pct(rk_full)} (+{gain:.1f}%)")

    if run_answers:
        full_ans = answer_metrics.get("full")
        vec_ans  = answer_metrics.get("vector_only")
        if full_ans and vec_ans:
            print(f"  ROUGE-L 变化:  {format_pct(vec_ans.avg_rouge_l)} -> {format_pct(full_ans.avg_rouge_l)}")
            print(f"  延迟差异:      {vec_ans.avg_latency_ms:.0f} ms -> {full_ans.avg_latency_ms:.0f} ms")

    print()
    print("=" * 72)


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="医疗知识问答系统消融实验评估")
    parser.add_argument("--samples", type=int, default=50, help="评估样本数 (默认 50)")
    parser.add_argument("--all", action="store_true", help="使用全部验证集")
    parser.add_argument("--no-agent", action="store_true", help="仅评测检索 Recall@K，不评测回答准确率")
    args = parser.parse_args()

    max_samples = None if args.all else args.samples
    run_answers = not args.no_agent

    print("=" * 70)
    print("  医疗知识问答系统 — 消融实验评估")
    print("=" * 70)
    print()

    # 1. 加载数据
    print("[0/4] 加载验证集...")
    data = sample_from_train(TRAIN_PATH, max_samples if max_samples else 50)
    if not data:
        print(f"错误: 训练集为空或路径不正确: {TRAIN_PATH}")
        sys.exit(1)
    print(f"      已加载 {len(data)} 条验证样本")

    # 2. 初始化 RAG
    print("\n[1/4] 初始化 RAG Pipeline...")
    rag = get_rag()
    status = rag.get_status()
    print(f"      知识库状态: {status['status']}, 文档数: {status['total_documents']}")
    if status["status"] == "empty":
        print("提示: 知识库为空，尝试自动导入...")
        result = rag.load_once()
        print(f"      导入结果: {result['message']}")

    # 3. 检索评估
    print("\n[2/4] 检索 Recall@K 评估...")
    retrieval_metrics = {}
    for config_name, config_label in RETRIEVAL_CONFIGS.items():
        pass  # progress shown by bar
        metrics, _ = evaluate_retrieval(rag, data, config_name)
        retrieval_metrics[config_name] = metrics
        pass  # results shown in final report

    # 4. 回答评估（可选）
    answer_metrics = {}
    if run_answers:
        print("\n[3/4] 回答准确率评估（调用 Agent 全链路，较耗时）...")
        agent = get_agent()
        for config_name, config_label in RETRIEVAL_CONFIGS.items():
            pass  # progress shown by bar
            metrics, _ = evaluate_answers(agent, data, config_name)
            answer_metrics[config_name] = metrics
            pass  # results shown in final report
    else:
        print("\n[3/4] 跳过回答准确率评估 (--no-agent)")

    # 5. 输出报告
    print("\n[4/4] 生成评估报告...")
    print_report(retrieval_metrics, answer_metrics, len(data), run_answers)

    # 保存详细结果
    output_path = PROJECT_ROOT / "eval_results.json"
    output_data = {
        "total_samples": len(data),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "retrieval": {},
        "answers": {},
    }
    for name, m in retrieval_metrics.items():
        output_data["retrieval"][name] = {
            "recall_at_k": {str(k): round(v, 4) for k, v in m.recall_at_k.items()},
            "avg_retrieved": m.avg_retrieved,
        }
    for name, m in answer_metrics.items():
        output_data["answers"][name] = {
            "avg_rouge_l": round(m.avg_rouge_l, 4),
            "avg_char_f1": round(m.avg_char_f1, 4),
            "avg_latency_ms": round(m.avg_latency_ms, 1),
        }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
