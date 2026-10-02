"""
Phase 2 — RAGAS + Tool Correctness metrics.
Evaluates 6 dimensions:
1. Faithfulness (faithfulness to retrieved context)
2. Answer Relevancy (relevancy of answer to the user question)
3. Context Precision (ground-truth reference compared to retrieved contexts)
4. Context Recall (ground-truth coverage in retrieved contexts)
5. Answer Correctness (factual & semantic similarity to reference answer)
6. Tool Correctness (Jaccard similarity of expected vs actual tools called)
"""

import asyncio
import os
import sys

# Patch nest_asyncio to avoid Python 3.14 asyncio.timeouts collision
import nest_asyncio
nest_asyncio.apply = lambda *args, **kwargs: None

# Compatibility bridge for vertexai import in older ragas versions
try:
    import langchain_google_vertexai
    sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai
except Exception:
    pass

import logfire
import pandas as pd
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

OPENAI_BASE_URL = "https://api.openai.com/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_JUDGE_MODEL = "qwen/qwen3.8-27b"
CONTEXT_TRUNCATE = 500  # chars per context chunk
CONTEXT_LIMIT = 3  # number of context chunks passed per sample


def _build_judge():
    openai_key = os.getenv("JUDGE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if openai_key:
        raw_llm = ChatOpenAI(
            model=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
            api_key=openai_key,
            temperature=0.0,
        )
    else:
        # Fallback to Groq judge as documented in 03_evals.ipynb
        groq_key = os.getenv("JUDGE_GROQ") or os.getenv("GROQ_API_KEY")
        raw_llm = ChatOpenAI(
            base_url=GROQ_BASE_URL,
            api_key=groq_key,
            model=DEFAULT_GROQ_JUDGE_MODEL,
            temperature=0.0,
            max_tokens=400,
        )

    judge_llm = LangchainLLMWrapper(raw_llm)
    embeddings = LangchainEmbeddingsWrapper(FastEmbedEmbeddings())
    return judge_llm, embeddings


def _prep_samples(golden_dataset: dict) -> list:
    """
    Returns samples with actual_response populated.
    Truncates contexts to keep evaluations fast and focused.
    """
    valid = []
    for s in golden_dataset.get("rag_samples", []):
        response = s.get("actual_response", "").strip()
        if not response:
            continue
        raw_contexts = s.get("actual_contexts") or s.get("relevant_contexts") or []
        contexts = [c[:CONTEXT_TRUNCATE] for c in raw_contexts[:CONTEXT_LIMIT]]
        valid.append({**s, "actual_contexts": contexts})
    return valid


async def _score_single_turn(metric, sample: SingleTurnSample) -> float:
    try:
        val = await metric.single_turn_ascore(sample)
        if isinstance(val, (int, float)):
            return round(float(val), 3)
        if hasattr(val, "value"):
            return round(float(val.value), 3)
        return 0.0
    except Exception as e:
        logfire.warning(f"Error evaluating sample with {metric.name}: {e}")
        return 0.0


async def run_all_metrics(golden_dataset: dict, status_cb=None) -> dict:
    """
    Runs all 6 experiments. Returns dict keyed by metric name -> DataFrame.
    """
    judge_llm, ragas_embeddings = _build_judge()
    samples = _prep_samples(golden_dataset)

    if not samples:
        raise ValueError("No samples with actual_response found. Run Phase 1 first.")

    results = {}
    n = len(samples)

    with logfire.span("🧪 Eval Phase 2 — All Metrics", total_samples=n):
        # ── Exp 1: Faithfulness ───────────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 1/6 — Faithfulness ({n} samples)...")
        with logfire.span("🧪 Exp 1 — Faithfulness"):
            metric = Faithfulness(llm=judge_llm)
            scores = []
            for s in samples:
                st_sample = SingleTurnSample(
                    user_input=s["question"],
                    response=s["actual_response"],
                    retrieved_contexts=s["actual_contexts"],
                )
                sc = await _score_single_turn(metric, st_sample)
                scores.append({"question": s["question"][:65], "faithfulness": sc})
            df = pd.DataFrame(scores)
            results["faithfulness"] = df
            logfire.info("🧪 Faithfulness done", avg=round(df["faithfulness"].mean(), 3))

        # ── Exp 2: Answer Relevancy ───────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 2/6 — Answer Relevancy ({n} samples)...")
        with logfire.span("🧪 Exp 2 — Answer Relevancy"):
            metric = AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings)
            scores = []
            for s in samples:
                st_sample = SingleTurnSample(
                    user_input=s["question"],
                    response=s["actual_response"],
                )
                sc = await _score_single_turn(metric, st_sample)
                scores.append({"question": s["question"][:65], "answer_relevancy": sc})
            df = pd.DataFrame(scores)
            results["answer_relevancy"] = df
            logfire.info("🧪 Answer Relevancy done", avg=round(df["answer_relevancy"].mean(), 3))

        # ── Exp 3: Context Precision ──────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 3/6 — Context Precision ({n} samples)...")
        with logfire.span("🧪 Exp 3 — Context Precision"):
            metric = ContextPrecision(llm=judge_llm)
            scores = []
            for s in samples:
                st_sample = SingleTurnSample(
                    user_input=s["question"],
                    reference=s["reference"],
                    retrieved_contexts=s["actual_contexts"],
                )
                sc = await _score_single_turn(metric, st_sample)
                scores.append({"question": s["question"][:65], "context_precision": sc})
            df = pd.DataFrame(scores)
            results["context_precision"] = df
            logfire.info("🧪 Context Precision done", avg=round(df["context_precision"].mean(), 3))

        # ── Exp 4: Context Recall ─────────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 4/6 — Context Recall ({n} samples)...")
        with logfire.span("🧪 Exp 4 — Context Recall"):
            metric = ContextRecall(llm=judge_llm)
            scores = []
            for s in samples:
                st_sample = SingleTurnSample(
                    user_input=s["question"],
                    reference=s["reference"],
                    retrieved_contexts=s["actual_contexts"],
                )
                sc = await _score_single_turn(metric, st_sample)
                scores.append({"question": s["question"][:65], "context_recall": sc})
            df = pd.DataFrame(scores)
            results["context_recall"] = df
            logfire.info("🧪 Context Recall done", avg=round(df["context_recall"].mean(), 3))

        # ── Exp 5: Answer Correctness ─────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 5/6 — Answer Correctness ({n} samples)...")
        with logfire.span("🧪 Exp 5 — Answer Correctness"):
            metric = AnswerCorrectness(llm=judge_llm, embeddings=ragas_embeddings)
            scores = []
            for s in samples:
                st_sample = SingleTurnSample(
                    user_input=s["question"],
                    response=s["actual_response"],
                    reference=s["reference"],
                )
                sc = await _score_single_turn(metric, st_sample)
                scores.append({"question": s["question"][:65], "answer_correctness": sc})
            df = pd.DataFrame(scores)
            results["answer_correctness"] = df
            logfire.info("🧪 Answer Correctness done", avg=round(df["answer_correctness"].mean(), 3))

        # ── Exp 6: Tool Correctness (Jaccard similarity — zero LLM calls) ────
        if status_cb:
            status_cb("⚡ Exp 6/6 — Tool Correctness (zero LLM calls)...")
        with logfire.span("🧪 Exp 6 — Tool Correctness"):
            tool_rows = []
            for s in samples:
                called = set(s.get("actual_tools_called") or [])
                expected = set(s.get("expected_tools") or [])
                union = len(called | expected)
                score = len(called & expected) / union if union > 0 else 0.0
                tool_rows.append({"question": s["question"][:65], "tool_correctness": round(score, 3)})
            df = pd.DataFrame(tool_rows)
            results["tool_correctness"] = df
            logfire.info("🧪 Tool Correctness done", avg=round(df["tool_correctness"].mean(), 3))

        if status_cb:
            status_cb("✅ All 6 experiments complete!")

    return results
