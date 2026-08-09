"""
电力规则审查系统 - RAGAS 风格自动化评测指标（LLM-as-Judge）

自研实现 RAGAS 框架的四个核心指标，不引入 ragas 库：
- Faithfulness（忠实度）: 回答中的 claims 被检索 context 支持的比例
  score = supported_claims / total_claims
- Answer Relevancy（答案相关性）: 回答是否切题，LLM judge 直接打分 0-1
  （简化版；官方做法为「由回答反生成若干问题 + 嵌入相似度」，需加载 bge-m3，
  升级路径见 docs/rule-review.md 第三部分 方向九）
- Context Precision（上下文精度）: 检索结果中相关 chunk 是否排在前面
  score = Σ_k(P@k × rel_k) / Σ_k rel_k，其中 P@k = |{j≤k: rel_j}| / k
  （全不相关 → 0.0，RAGAS 口径）
- Context Recall（上下文召回）: 参考答案中的句子被 context 支持的比例
  score = supported_sentences / total_sentences

降级策略（与项目整体风格一致，绝不阻断评估主流程）：
- 缺少 reference_answer → Context Precision/Recall 返回 None
- 检索 context 为空 → Faithfulness/Precision/Recall 返回 None（避免误导性 0 分）
- LLM 调用超时/异常/输出不可解析 → 该指标返回 None，不重试
  （与 judge.py 的「重试 1 次」有意区分：评估为批量场景，重试会线性放大成本）
- 单个指标失败不影响其余指标

约定：4 个公开 compute_* 函数均返回 (score, details)：
- score: float | None，None 表示跳过（缺输入/调用失败）
- details: dict，含该指标的判定明细（claims/verdicts/sentences）与 error 信息

评测编排（EvalRunner 集成）与整体评估流程见 evaluation.py；
指标口径与面试话术见 docs/rule-review.md 第三部分「方向九」。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config import settings
from src.rule_review.judge import RuleReviewJudge
from src.utils.model_proxy import ProxyChatModel
from src.utils.output_parser import parse_json_block

logger = logging.getLogger(__name__)

# 单指标 LLM 调用超时（秒），与 judge.py 的 _JUDGE_TIMEOUT 保持一致
_METRICS_TIMEOUT = 60


# ---------------------------------------------------------------------------
# 指标 Prompt
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """你是电力交易领域的事实一致性审核专家。

## 任务
1. 从「回答」中抽取全部事实性陈述（claim）。每条 claim 必须是回答中独立可验证的一句话级事实，
   例如「冀北日前现货出清电价800元/MWh超过上限760元/MWh」「该申报不符合价格上限规定」。
   不要拆分过细（如把同一句话拆成多条），也不要遗漏重要事实。
2. 逐条判断该 claim 是否被「规则原文（检索到的 chunks）」支持：
   - supported=true：原文中存在明确依据（数值、条款、规则均可）
   - supported=false：原文中找不到依据，或与原文矛盾（编造/幻觉）
3. 判断依据只能是给出的规则原文，不能凭领域常识补全。

## 输出格式
必须严格输出以下 JSON：
{{"claims": [{{"claim": "陈述内容", "supported": true/false, "reason": "依据或缺失说明"}}]}}"""

RELEVANCY_PROMPT = """你是回答质量评审专家。

## 任务
判断「回答」是否切题：是否直接回答了用户问题、是否包含无关或冗余信息。
评分标准（0-1 分）：
- 1.0: 完全切题，直接、完整地回答了问题
- 0.7-0.9: 基本切题，存在少量冗余或未完全聚焦
- 0.3-0.6: 部分切题，包含较多无关信息或回答偏题
- 0.0-0.2: 几乎不相关，答非所问

## 输出格式
必须严格输出以下 JSON：
{{"score": 0.0-1.0, "reason": "评分理由"}}"""

CONTEXT_PRECISION_PROMPT = """你是检索质量评审专家。

## 任务
判断每条「规则原文（检索到的 chunks）」是否与用户问题相关。
相关性的判定标准：该 chunk 是否包含了回答「参考要点」所需的规则/数据依据。
- 包含与参考要点相关的规则条款、数值、约束 → relevant=true
- 与问题或参考要点无关 → relevant=false

## 输出格式
必须严格输出以下 JSON：
{{"verdicts": [{{"index": 0, "relevant": true/false, "reason": "判定理由"}}]}}

注意：
- verdicts 必须覆盖全部 chunk，index 从 0 开始依次对应
- 宁可严格：只有确实支撑回答依据的 chunk 才算 relevant"""

CONTEXT_RECALL_PROMPT = """你是检索完整性评审专家。

## 任务
逐句判断「参考答案」中的每个句子是否能在「规则原文（检索到的 chunks）」中找到依据。
- 句子表述可被某条 chunk 中的原文直接支撑（数值、条款、规则均算）→ supported=true
- 任何 chunk 都找不到对应依据 → supported=false

## 输出格式
必须严格输出以下 JSON：
{{"sentences": [{{"sentence": "句子内容", "supported": true/false, "reason": "依据或缺失说明"}}]}}

注意：
- sentences 必须与输入的句子列表一一对应，顺序一致
- 判断依据只能是给出的规则原文"""


# ---------------------------------------------------------------------------
# 模型工厂
# ---------------------------------------------------------------------------

_default_metrics_model: Any | None = None


def _create_metrics_model() -> Any:
    """创建评测专用 LLM 模型实例。

    与 judge 的模型工厂不同，评测模型**只走 httpx 代理路径**
    （ProxyChatModel），绝不 import langchain_qwq / ChatQwen：
    后者会加载 torch（自带 libomp.dylib），与同一进程内 faiss 的
    OpenMP 运行库冲突，导致 faiss.search 直接 abort——而离线评估
    必然同时使用检索（faiss）与 LLM judge，二者必须能共存于一个进程。

    配置回退链：模型名取 JUDGE_MODEL（在网关模型列表中则走网关，
    否则走 DashScope OpenAI 兼容端点）；api_key / base_url 按
    JUDGE → RULE_REVIEW → DASHSCOPE 顺序回退。
    temperature 固定 0.0（经 ainvoke kwargs 生效），保证评估可复现。
    """
    model_name = settings.JUDGE_MODEL
    api_key = (
        settings.JUDGE_API_KEY
        or settings.RULE_REVIEW_API_KEY
        or settings.DASHSCOPE_API_KEY
    )
    base_url = (
        settings.JUDGE_API_BASE
        or settings.RULE_REVIEW_API_BASE
        or settings.DASHSCOPE_API_BASE
    )
    if model_name in getattr(settings, "GATEWAY_MODELS", []) and settings.GATEWAY_BASE_URL:
        base_url = settings.GATEWAY_BASE_URL  # 网关 base_url 已是完整端点
    elif base_url and "/chat/completions" not in base_url:
        # DashScope OpenAI 兼容模式：base_url 是服务根路径，需补全端点
        base_url = base_url.rstrip("/") + "/chat/completions"

    model = ProxyChatModel(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        enable_thinking=False,
        timeout=getattr(settings, "REQUEST_TIMEOUT_SECONDS", 60),
    )
    model.temperature = 0.0
    return model


def _get_default_metrics_model() -> Any:
    """获取（懒加载并缓存）默认评测模型。"""
    global _default_metrics_model
    if _default_metrics_model is None:
        _default_metrics_model = _create_metrics_model()
    return _default_metrics_model


# ---------------------------------------------------------------------------
# 共享工具
# ---------------------------------------------------------------------------


def _extract_content(response: AIMessage) -> str:
    """从 LangChain AIMessage 提取文本（与 judge.py 同款容错）。"""
    return RuleReviewJudge._extract_content(response)


async def _ainvoke_json(
    model: Any,
    system_prompt: str,
    user_content: str,
) -> dict | None:
    """调用 LLM 并要求返回 JSON，容错解析。

    Returns:
        解析成功的 dict；超时/异常/解析失败返回 None（不重试，见模块 docstring）。
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    kwargs = {}
    temp = getattr(model, "temperature", None)
    if temp is not None:
        kwargs["temperature"] = temp
    try:
        response: AIMessage = await asyncio.wait_for(
            model.ainvoke(messages, **kwargs),
            timeout=_METRICS_TIMEOUT,
        )
        raw_text = _extract_content(response)
        parsed = parse_json_block(raw_text)
        if parsed is None or not isinstance(parsed, dict):
            logger.warning("[Metrics] 输出无法解析为 JSON，前 200 字符: %s", raw_text[:200])
            return None
        return parsed
    except asyncio.TimeoutError:
        logger.warning("[Metrics] LLM 调用超时 %ss", _METRICS_TIMEOUT)
        return None
    except Exception as e:
        logger.warning("[Metrics] LLM 调用异常: %s", e)
        return None


def _build_context_text(chunks: list[dict]) -> str:
    """将 chunks 列表组装为可读的上下文文本（复用 judge 的格式化）。"""
    return RuleReviewJudge._build_context_text(chunks)


def _clamp01(value: float) -> float:
    """钳制分数到 [0, 1]。"""
    return max(0.0, min(1.0, value))


def _split_sentences(text: str) -> list[str]:
    """按中文句读标点拆分句子，过滤空句。"""
    import re

    parts = re.split(r"[。！？!?；;]", text or "")
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# 四指标计算
# ---------------------------------------------------------------------------


async def compute_faithfulness(
    question: str,
    answer_text: str,
    context_chunks: list[dict],
    model: Any | None = None,
) -> tuple[float | None, dict]:
    """Faithfulness：回答中 claims 被检索 context 支持的比例。

    Args:
        question: 用户问题。
        answer_text: 生成的回答文本（reason + evidence 拼接）。
        context_chunks: 检索到的 chunks（[{"text": ..., ...}, ...]）。
        model: LLM 模型实例；None 时使用默认评测模型。

    Returns:
        (score | None, details)；score 为受支持 claims 占比；
        details 含 claims 明细；无 claim 或空 context → None。
    """
    if not answer_text or not context_chunks:
        return None, {"skipped_reason": "回答或检索上下文为空", "error": ""}

    model = model or _get_default_metrics_model()
    user_content = (
        f"## 用户问题\n{question}\n\n"
        f"## 回答\n{answer_text}\n\n"
        f"## 规则原文（检索到的 chunks）\n{_build_context_text(context_chunks)}"
    )
    parsed = await _ainvoke_json(model, FAITHFULNESS_PROMPT, user_content)
    if parsed is None:
        return None, {"error": "LLM 调用或解析失败"}

    claims = parsed.get("claims") or []
    if not claims:
        return None, {"error": "", "claims": [], "reason": "未抽取到任何 claim"}

    supported = sum(1 for c in claims if bool(c.get("supported")))
    score = supported / len(claims)
    return score, {
        "claims": claims,
        "total_claims": len(claims),
        "supported_claims": supported,
    }


async def compute_answer_relevancy(
    question: str,
    answer_text: str,
    model: Any | None = None,
) -> tuple[float | None, dict]:
    """Answer Relevancy：回答是否切题（LLM judge 直接打分 0-1）。

    Args:
        question: 用户问题。
        answer_text: 生成的回答文本。
        model: LLM 模型实例；None 时使用默认评测模型。

    Returns:
        (score | None, details)；score 已钳制到 [0, 1]。
    """
    if not answer_text:
        return None, {"skipped_reason": "回答为空", "error": ""}

    model = model or _get_default_metrics_model()
    user_content = (
        f"## 用户问题\n{question}\n\n"
        f"## 回答\n{answer_text}"
    )
    parsed = await _ainvoke_json(model, RELEVANCY_PROMPT, user_content)
    if parsed is None:
        return None, {"error": "LLM 调用或解析失败"}

    try:
        raw_score = float(parsed.get("score", 0.0))
    except (TypeError, ValueError):
        return None, {"error": "score 字段不可解析", "reason": parsed.get("reason", "")}

    return _clamp01(raw_score), {
        "reason": parsed.get("reason", ""),
    }


async def compute_context_precision(
    question: str,
    reference_answer: str,
    context_chunks: list[dict],
    model: Any | None = None,
) -> tuple[float | None, dict]:
    """Context Precision：检索结果中相关 chunk 的排名质量。

    RAGAS 口径：score = Σ_k(P@k × rel_k) / Σ_k rel_k，
    其中 P@k 为前 k 个 chunk 中相关的比例；全不相关 → 0.0。

    Args:
        question: 用户问题。
        reference_answer: 参考答案（Context Precision/Recall 的判定基准）。
        context_chunks: 检索到的 chunks（按相关性排序）。
        model: LLM 模型实例；None 时使用默认评测模型。

    Returns:
        (score | None, details)；缺 reference_answer → None；
        details 含逐 chunk 判定明细。
    """
    if not context_chunks:
        return 0.0, {"error": "", "reason": "无检索上下文", "verdicts": []}
    if not reference_answer:
        return None, {"skipped_reason": "缺少 reference_answer", "error": ""}

    model = model or _get_default_metrics_model()
    user_content = (
        f"## 用户问题\n{question}\n\n"
        f"## 参考要点\n{reference_answer}\n\n"
        f"## 规则原文（检索到的 chunks）\n{_build_context_text(context_chunks)}"
    )
    parsed = await _ainvoke_json(model, CONTEXT_PRECISION_PROMPT, user_content)
    if parsed is None:
        return None, {"error": "LLM 调用或解析失败"}

    verdicts = parsed.get("verdicts") or []
    if not verdicts:
        return None, {"error": "", "verdicts": [], "reason": "无判定结果"}

    # 按 index 对齐 chunk 顺序（防御乱序/缺项）
    rel_flags: list[bool] = [False] * len(context_chunks)
    for v in verdicts:
        idx = int(v.get("index", -1))
        if 0 <= idx < len(context_chunks):
            rel_flags[idx] = bool(v.get("relevant"))

    # P@k 逐位累加：在位置 k 相关的 chunk 处，用当前已相关计数 / k
    relevant_count = 0
    numerator = 0.0
    for k, rel in enumerate(rel_flags, start=1):
        if rel:
            relevant_count += 1
            numerator += relevant_count / k

    total_relevant = relevant_count
    if total_relevant == 0:
        score = 0.0  # RAGAS 口径：无相关 chunk → 0
    else:
        score = numerator / total_relevant

    return score, {
        "verdicts": verdicts,
        "relevant_count": total_relevant,
        "total_chunks": len(context_chunks),
    }


async def compute_context_recall(
    question: str,
    reference_answer: str,
    context_chunks: list[dict],
    model: Any | None = None,
) -> tuple[float | None, dict]:
    """Context Recall：参考答案中的句子被 context 支持的比例。

    Args:
        question: 用户问题。
        reference_answer: 参考答案（判定基准）。
        context_chunks: 检索到的 chunks。
        model: LLM 模型实例；None 时使用默认评测模型。

    Returns:
        (score | None, details)；reference_answer 为空 → None；
        details 含逐句判定明细。
    """
    if not reference_answer:
        return None, {"skipped_reason": "缺少 reference_answer", "error": ""}

    sentences = _split_sentences(reference_answer)
    if not sentences:
        return None, {"skipped_reason": "参考答案拆分后为空", "error": ""}
    if not context_chunks:
        return 0.0, {"error": "", "reason": "无检索上下文", "sentences": []}

    model = model or _get_default_metrics_model()
    user_content = (
        f"## 用户问题\n{question}\n\n"
        f"## 参考答案句子\n" + "\n".join(
            f"{i + 1}. {s}" for i, s in enumerate(sentences)
        ) +
        f"\n\n## 规则原文（检索到的 chunks）\n{_build_context_text(context_chunks)}"
    )
    parsed = await _ainvoke_json(model, CONTEXT_RECALL_PROMPT, user_content)
    if parsed is None:
        return None, {"error": "LLM 调用或解析失败"}

    judgments = parsed.get("sentences") or []
    if not judgments:
        return None, {"error": "", "sentences": [], "reason": "无判定结果"}

    # 与输入句子按顺序对齐（防御模型改句子文本）
    supported = sum(
        1 for i, s in enumerate(sentences)
        if i < len(judgments) and bool(judgments[i].get("supported"))
    )
    score = supported / len(sentences)
    return score, {
        "sentences": judgments,
        "total_sentences": len(sentences),
        "supported_sentences": supported,
    }


# ---------------------------------------------------------------------------
# 编排器
# ---------------------------------------------------------------------------


@dataclass
class RagasMetrics:
    """单条用例的 RAGAS 四指标汇总。"""

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    details: dict = field(default_factory=dict)  # 各指标明细 + judge_latency_ms
    skipped: bool = False  # 整体跳过（如回答为空）
    skip_reason: str = ""
    errors: list[str] = field(default_factory=list)


async def compute_ragas_metrics(
    question: str,
    answer_text: str,
    context_chunks: list[dict],
    reference_answer: str = "",
    model: Any | None = None,
) -> RagasMetrics:
    """计算单条用例的 RAGAS 四指标。

    每项指标独立 try/except，单点失败不影响其余指标；
    缺失输入（空回答/空上下文/缺 reference）按降级策略置 None。

    Args:
        question: 用户问题。
        answer_text: 生成的回答文本。
        context_chunks: 检索到的 chunks。
        reference_answer: 参考答案（可选；缺省时 Context Precision/Recall 降级）。
        model: LLM 模型实例；None 时使用默认评测模型。

    Returns:
        RagasMetrics 汇总结果。
    """
    import time

    if not answer_text:
        return RagasMetrics(
            skipped=True,
            skip_reason="回答为空，全部指标跳过",
        )

    result = RagasMetrics()
    start = time.monotonic()

    async def _safe(metric_name: str, coro):
        """执行单指标，异常时记录 error 不影响其余。"""
        try:
            return await coro
        except Exception as e:
            logger.warning("[Metrics] %s 计算异常: %s", metric_name, e)
            result.errors.append(f"{metric_name}: {e}")
            return None, {"error": str(e)}

    # Faithfulness
    score, details = await _safe("faithfulness", compute_faithfulness(
        question, answer_text, context_chunks, model=model,
    ))
    result.faithfulness = score
    result.details["faithfulness"] = details

    # Answer Relevancy
    score, details = await _safe("answer_relevancy", compute_answer_relevancy(
        question, answer_text, model=model,
    ))
    result.answer_relevancy = score
    result.details["answer_relevancy"] = details

    # Context Precision（需 reference）
    score, details = await _safe("context_precision", compute_context_precision(
        question, reference_answer, context_chunks, model=model,
    ))
    result.context_precision = score
    result.details["context_precision"] = details

    # Context Recall（需 reference）
    score, details = await _safe("context_recall", compute_context_recall(
        question, reference_answer, context_chunks, model=model,
    ))
    result.context_recall = score
    result.details["context_recall"] = details

    result.details["judge_latency_ms"] = round((time.monotonic() - start) * 1000, 2)
    return result
