"""
A 层四件套工具：直击规则审查现有痛点的工具扩展。

1. locate_clause 条款定位器     — 按「第X条第Y款」在规则文档精确定位原文，
                                  服务 Corrective-RAG missing_rules 补检
2. verify_citation 证据引用核验  — 生成侧预校验 evidence 元数据与文本一致性，防幻觉
3. extract_numeric_fact 数值事实提取 — 非表格正文提取「主体+数值+单位」三元组
4. detect_rule_conflict 多文档冲突检测 — 多文档对同一事项规定不一致时定位冲突

全部纯 Python 不调 LLM；返回结构与现有工具一致：{"success": bool, "data"|"error"}。
注册方式：tool_executor.py 的 TOOL_MAP + data/env_variables/tools_config.json。
"""

from __future__ import annotations

import re

from src.rule_review.audit import _longest_common_substring
from src.rule_review.sandbox_utils import safe_parse_number
from src.rule_review.tool_executor import (
    _ALL_ENERGY_UNITS,
    _cn_to_int,
    unit_converter,
)

# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------

# 条款号正则（阿拉伯/中文数字，支持百/千）
_ARTICLE_RE = re.compile(r"第\s*([0-9]+|[一二三四五六七八九十百千]+)\s*条")
# 款标识：「第X款」或中文规则文本常见的「（一）」「(二)」格式
_PARAGRAPH_RE = re.compile(
    r"第\s*([0-9]+|[一二三四五六七八九十百千]+)\s*款"
    r"|[（(]\s*([0-9]+|[一二三四五六七八九十百千]+)\s*[）)]"
)

# 数值正则
_NUMBER_RE = re.compile(r"[+-]?\d+\.?\d*")
# 中文数字片段（"一万二千"等简单形式）
_CN_NUMBER_RE = re.compile(r"[一二三四五六七八九十百千万亿两]+点?[一二三四五六七八九十百千万亿两]*")

# 主体提取停止词（数值前语境词，不是主体名词）
_SUBJECT_STOPWORDS = {
    "上限", "下限", "价格", "电价", "规定", "按照", "依据", "根据", "参照",
    "不得", "不应", "不能", "不超过", "不低于", "高于", "低于", "超过",
    "上限为", "下限为", "为", "是", "系", "达", "达到", "介于",
    "执行", "实施", "适用", "按", "的", "应", "须", "均", "并",
    "元", "万元", "元/MWh", "元/千度", "元/万kWh", "分/kWh",
    "MWh", "kWh", "GWh", "万kWh", "亿kWh", "MW", "万千瓦", "兆瓦", "%",
}

# 数值后单位匹配（最长优先）
_UNIT_PATTERNS = sorted(
    set(_ALL_ENERGY_UNITS) | {"%", "百分之", "万千瓦", "兆瓦", "MW", "万千瓦时"},
    key=len,
    reverse=True,
)


def _normalize_text(text: str) -> str:
    """文本归一化：全角→半角、去空白与标点，用于比较。"""
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            code = 32
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        if 0x21 <= code <= 0x7E or "一" <= ch <= "鿿":
            out.append(chr(code))
    return "".join(out)


def _parse_article_no(value) -> int | None:
    """条款号统一转阿拉伯数字（支持中文数字/阿拉伯/含'第X条'完整文本）。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        s = value.strip()
        # 完整形式："第12条" / "第十二条" / "（二）"
        m = _ARTICLE_RE.search(s)
        if m:
            s = m.group(1)
        else:
            pm = _PARAGRAPH_RE.search(s)
            if pm:
                s = pm.group(1) or pm.group(2) or s
        return _cn_to_int(s) if s else None
    return None


# ---------------------------------------------------------------------------
# Tool 6: locate_clause 条款定位器
# ---------------------------------------------------------------------------


def locate_clause(
    article_no: str | int,
    paragraph_no: str | int | None = None,
    doc_name: str | None = None,
    chunks: list[dict] | None = None,
    all_chunks: list[dict] | None = None,
) -> dict:
    """按「第X条(第Y款)」在规则 chunks 中精确定位条款原文。

    与 resolve_cross_reference 的分工：后者只解析引用文本并做首次命中粗定位
    （300 字截断、不做条款区间切分）；本工具按结构化条款号做精确区间切分，
    返回完整条款正文与位置元数据，供校验引用或 Corrective-RAG 补检遗漏规则。

    Args:
        article_no: 条款号（阿拉伯/中文数字，或"第X条"完整形式）。
        paragraph_no: 款号，可选。
        doc_name: 文档名过滤，跨文档时必须传。
        chunks: 候选检索 chunks；未命中时 fallback 到 all_chunks。
        all_chunks: 全量 chunks（备选）。

    Returns:
        {"success": True, "data": {"found": bool, "clause_text", "paragraph_text",
         "source", "section", "page", "chunk_id", "match_level"}}
        定位失败时 found=false 且 success=true（不是工具错误，不触发降级）。
    """
    target = _parse_article_no(article_no)
    if target is None:
        return {"success": False, "error": f"条款号无效: {article_no!r}"}

    target_para = _parse_article_no(paragraph_no) if paragraph_no is not None else None

    search_chunks = chunks or []
    if not search_chunks and all_chunks:
        search_chunks = all_chunks

    if not search_chunks:
        return {
            "success": True,
            "data": {"found": False, "message": "无可用 chunks 供条款定位"},
        }

    for chunk in search_chunks:
        if doc_name:
            src = chunk.get("source", "")
            if src and doc_name not in src and src not in doc_name:
                continue
        text = chunk.get("text", "")
        located = _locate_in_text(text, target, target_para)
        if located:
            return {
                "success": True,
                "data": {
                    **located,
                    "source": chunk.get("source", ""),
                    "section": chunk.get("section", ""),
                    "page": chunk.get("page", 0),
                    "chunk_id": chunk.get("chunk_id", ""),
                },
            }

    return {
        "success": True,
        "data": {
            "found": False,
            "message": f"未在可用 chunks 中定位到第{target}条"
                       + (f"第{target_para}款" if target_para else ""),
        },
    }


def _locate_in_text(text: str, article_no: int, paragraph_no: int | None) -> dict | None:
    """在单条 chunk 文本内按条款号切分定位。

    策略：按行扫描找到目标「第X条」起始行，收集直到下一个条款号为止的正文；
    条款正文内再按「第Y款」切款段。返回 match_level: exact(条+款) / article(仅条)。
    """
    lines = text.split("\n")
    article_start: int | None = None
    article_end: int | None = None

    for i, line in enumerate(lines):
        m = _ARTICLE_RE.search(line)
        if m:
            num = _parse_article_no(m.group(0))
            if num == article_no and article_start is None:
                article_start = i
            elif article_start is not None:
                # 遇到下一个条款 → 结束
                article_end = i
                break
    if article_start is None:
        return None

    if article_end is None:
        article_end = len(lines)

    article_lines = lines[article_start:article_end]
    clause_text = "\n".join(l for l in article_lines if l.strip()).strip()
    if not clause_text:
        return None

    # 款定位
    paragraph_text = ""
    match_level = "article"
    if paragraph_no is not None:
        para_start: int | None = None
        para_end: int | None = None
        for i, line in enumerate(article_lines):
            pm = _PARAGRAPH_RE.search(line)
            if pm:
                pnum = _parse_article_no(pm.group(0))
                if pnum == paragraph_no and para_start is None:
                    para_start = i
                elif para_start is not None:
                    para_end = i
                    break
        if para_start is not None:
            para_lines = article_lines[para_start:] if para_end is None else article_lines[para_start:para_end]
            paragraph_text = "\n".join(l for l in para_lines if l.strip()).strip()
            match_level = "exact"

    return {
        "found": True,
        "clause_text": clause_text,
        "paragraph_text": paragraph_text,
        "match_level": match_level,
    }


# ---------------------------------------------------------------------------
# Tool 7: verify_citation 证据引用核验
# ---------------------------------------------------------------------------


def verify_citation(
    evidence_text: str,
    claimed_source: str = "",
    claimed_section: str = "",
    claimed_page: int = 0,
    claimed_chunk_id: str = "",
    chunks: list[dict] | None = None,
    lcs_threshold: float = 0.5,
) -> dict:
    """校验 evidence 引用与规则原文是否一致（生成侧防幻觉预校验）。

    机械一致性校验：
    - 文本匹配：evidence 与最佳 chunk 的 LCS 覆盖率（归一化后计算）
    - 元数据匹配：source 包含匹配、section 归一化相等/包含、page 严格相等
    未声称的项跳过校验（如 claimed_page=0 不校验页码）。

    Args:
        evidence_text: 待核验的 evidence 原文引用。
        claimed_source: 声称的文档名（可带《》）。
        claimed_section: 声称的章节标题。
        claimed_page: 声称的页码，0 表示未声称。
        claimed_chunk_id: 声称的 chunk_id，非空时优先与该 chunk 比对。
        chunks: 检索结果 chunks（含 text/source/section/page/chunk_id）。
        lcs_threshold: 文本匹配通过阈值（LCS 占比）。

    Returns:
        {"success": True, "data": {"verified", "text_matched", "text_match_ratio",
         "source_matched", "section_matched", "page_matched", "best_chunk_id", "issues"}}
    """
    if not chunks:
        return {
            "success": True,
            "data": {
                "verified": False,
                "issues": ["无可用 chunks，无法核验引用"],
                "text_matched": False,
                "source_matched": False,
                "section_matched": False,
                "page_matched": False,
                "text_match_ratio": 0.0,
                "best_chunk_id": "",
            },
        }

    target_norm = _normalize_text(evidence_text)
    if not target_norm:
        return {
            "success": True,
            "data": {
                "verified": False,
                "issues": ["evidence_text 为空"],
                "text_matched": False,
                "source_matched": False,
                "section_matched": False,
                "page_matched": False,
                "text_match_ratio": 0.0,
                "best_chunk_id": "",
            },
        }

    # 1. 文本匹配：与最佳 chunk 计算 LCS 覆盖率
    best_ratio = 0.0
    best_chunk: dict | None = None
    for chunk in chunks:
        chunk_norm = _normalize_text(chunk.get("text", ""))
        if not chunk_norm:
            continue
        lcs_len = _longest_common_substring(target_norm, chunk_norm)
        ratio = lcs_len / max(len(target_norm), 1)
        if ratio > best_ratio:
            best_ratio = ratio
            best_chunk = chunk

    text_matched = best_ratio >= lcs_threshold
    issues: list[str] = []

    if best_chunk is not None:
        best_chunk_id = best_chunk.get("chunk_id", "")
        # 2. 元数据匹配
        source_matched = True
        if claimed_source.strip():
            claimed_clean = claimed_source.strip().strip("《》").strip()
            src = str(best_chunk.get("source", ""))
            src_clean = src.strip("《》").strip()
            source_matched = bool(claimed_clean and (claimed_clean in src_clean or src_clean in claimed_clean))
            if not source_matched:
                issues.append(f"来源不匹配: 声称《{claimed_source}》，实际为《{src}》")

        section_matched = True
        if claimed_section.strip():
            claimed_sec = _normalize_text(claimed_section)
            sec = _normalize_text(str(best_chunk.get("section", "")))
            section_matched = bool(
                sec and (claimed_sec == sec or claimed_sec in sec or sec in claimed_sec)
            )
            if not section_matched:
                issues.append(f"章节不匹配: 声称[{claimed_section}]，实际为[{best_chunk.get('section', '')}]")

        page_matched = True
        if claimed_page > 0:
            actual_page = best_chunk.get("page", 0)
            page_matched = int(claimed_page) == int(actual_page) if isinstance(actual_page, (int, float)) else False
            if not page_matched:
                issues.append(f"页码不匹配: 声称第{claimed_page}页，实际为第{actual_page}页")
    else:
        best_chunk_id = ""
        source_matched = section_matched = page_matched = False
        issues.append("文本与所有 chunk 均无重叠")

    if not text_matched:
        issues.append(f"文本匹配率 {best_ratio:.2f} 低于阈值 {lcs_threshold}，引用疑似改写或编造")

    verified = text_matched and source_matched and section_matched and page_matched

    return {
        "success": True,
        "data": {
            "verified": verified,
            "text_matched": text_matched,
            "text_match_ratio": round(best_ratio, 4),
            "source_matched": source_matched,
            "section_matched": section_matched,
            "page_matched": page_matched,
            "best_chunk_id": best_chunk_id,
            "issues": issues,
        },
    }


# ---------------------------------------------------------------------------
# Tool 8: extract_numeric_fact 数值事实提取
# ---------------------------------------------------------------------------


def extract_numeric_fact(
    text: str,
    subject: str | None = None,
    max_facts: int = 20,
) -> dict:
    """从非表格规则正文中提取「主体+数值+单位」三元组。

    数值匹配：阿拉伯数字优先；数值附近尝试中文数字（safe_parse_number）。
    主体提取：数值位置向前回溯最近的标点/连接词分隔段，过滤停止词。

    Args:
        text: 规则正文文本。
        subject: 可选主体限定，只返回该主体的 fact。
        max_facts: 最多返回的 fact 数。

    Returns:
        {"success": True, "data": {"facts": [{"subject", "value", "unit", "raw", "position"}]}}
    """
    if not text or not text.strip():
        return {"success": True, "data": {"facts": []}}

    facts: list[dict] = []
    seen_raw: set[str] = set()

    for m in _NUMBER_RE.finditer(text):
        if len(facts) >= max_facts:
            break

        raw_num = m.group()
        # 跳过明显不是数据的场景（如年份在句首被当作主体的一部分？仍提取，交给上层判断）
        value = safe_parse_number(raw_num)
        if value is None:
            continue

        # 单位：数值后 0-6 字符内匹配单位词表
        tail = text[m.end(): m.end() + 6]
        unit = ""
        for u in _UNIT_PATTERNS:
            if tail.startswith(u):
                unit = u
                break

        # 原始片段：数值 + 单位
        raw = text[m.start(): m.end() + len(unit)]

        # 主体：向前回溯分隔符（标点/连接词/停顿时）
        subject_text = _extract_subject(text, m.start())

        if subject and subject_text and subject not in subject_text:
            continue  # 主体限定不匹配

        fact = {
            "subject": subject_text,
            "value": value,
            "unit": unit,
            "raw": raw,
            "position": m.start(),
        }
        key = (subject_text, value, unit)
        if key in seen_raw:
            continue
        seen_raw.add(key)
        facts.append(fact)

    # 中文数字片段（"一万二千元"等，阿拉伯数字未覆盖）
    for m in _CN_NUMBER_RE.finditer(text):
        if len(facts) >= max_facts:
            break
        # 排除条款号/章节号中的中文数字（"第十二条"的"十二"）
        after = text[m.end(): m.end() + 1]
        if after in "条款章节号部分":
            continue
        value = safe_parse_number(m.group())
        if value is None:
            continue
        # 单位与主体复用同一套提取逻辑
        tail = text[m.end(): m.end() + 6]
        unit = ""
        for u in _UNIT_PATTERNS:
            if tail.startswith(u):
                unit = u
                break
        raw = text[m.start(): m.end() + len(unit)]
        subject_text = _extract_subject(text, m.start())
        key = (subject_text, value, unit)
        if key in seen_raw:
            continue
        seen_raw.add(key)
        facts.append({
            "subject": subject_text,
            "value": value,
            "unit": unit,
            "raw": raw,
            "position": m.start(),
        })

    return {"success": True, "data": {"facts": facts}}


# 主体前分隔标点
_SUBJECT_BOUNDARY_CHARS = set("，。；、,.;:：（）()《》「」\"'『』\n")


def _extract_subject(text: str, num_pos: int) -> str:
    """提取数值前 1-16 字符内的主体名词短语。"""
    start = max(0, num_pos - 16)
    prefix = text[start:num_pos]

    # 从后往前找标点分隔边界
    seg = prefix
    for i in range(len(prefix) - 1, -1, -1):
        if prefix[i] in _SUBJECT_BOUNDARY_CHARS:
            seg = prefix[i + 1:]
            break

    # 停止词子串过滤（支持"上限为""电价上限为"等组合词）
    candidate = seg.strip()
    for w in sorted(_SUBJECT_STOPWORDS, key=len, reverse=True):
        candidate = candidate.replace(w, "")

    # 过滤后为空（纯停止词语境），回退为数值前最近 4 字符
    if not candidate:
        candidate = prefix[-4:].strip()
    return candidate


# ---------------------------------------------------------------------------
# Tool 9: detect_rule_conflict 多文档冲突检测
# ---------------------------------------------------------------------------


def detect_rule_conflict(pairs: list[dict]) -> dict:
    """检测多个规则文档对同一事项的规定是否冲突。

    分层判定（纯 Python，不硬判语义）：
    1. 双方均有数值 → 单位归一化后比较，差值 > 1% 判「数值冲突」
    2. 双方数值相等 → 「无冲突」
    3. 一方/双方无数值 → LCS 文本相似度 > 0.7 → 「无冲突」，否则「疑似冲突（待人工复核）」

    Args:
        pairs: [{"doc_a", "doc_b", "topic", "rule_a_text", "rule_b_text"}]。
            建议 LLM 先用 locate_clause 定位双方条款原文再调本工具。

    Returns:
        {"success": True, "data": {"conflicts": [{"doc_a", "doc_b", "topic",
         "conflict", "conflict_type", "reason", "rule_a_source", "rule_b_source"}]}}
    """
    if not isinstance(pairs, list):
        return {"success": False, "error": "pairs 必须是数组"}

    conflicts: list[dict] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        doc_a = str(pair.get("doc_a", ""))
        doc_b = str(pair.get("doc_b", ""))
        topic = str(pair.get("topic", ""))
        text_a = str(pair.get("rule_a_text", ""))
        text_b = str(pair.get("rule_b_text", ""))
        rule_a_source = str(pair.get("rule_a_source", ""))
        rule_b_source = str(pair.get("rule_b_source", ""))

        result = _judge_pair_conflict(text_a, text_b)
        conflicts.append({
            "doc_a": doc_a,
            "doc_b": doc_b,
            "topic": topic,
            "conflict": result["conflict"],
            "conflict_type": result["conflict_type"],
            "reason": result["reason"],
            "rule_a_source": rule_a_source,
            "rule_b_source": rule_b_source,
        })

    return {"success": True, "data": {"conflicts": conflicts}}


def _judge_pair_conflict(text_a: str, text_b: str) -> dict:
    """判定两条规则文本是否冲突。"""
    # 1. 双方数值提取（单位归一化后比较）
    facts_a = extract_numeric_fact(text_a)["data"]["facts"]
    facts_b = extract_numeric_fact(text_b)["data"]["facts"]

    val_a = _first_normalized_value(facts_a)
    val_b = _first_normalized_value(facts_b)

    if val_a is not None and val_b is not None:
        diff_ratio = abs(val_a - val_b) / max(abs(val_b), 1e-9)
        if diff_ratio > 0.01:
            return {
                "conflict": True,
                "conflict_type": "数值冲突",
                "reason": (
                    f"A 规定 {val_a:g}，B 规定 {val_b:g}"
                    f"（归一化后差 {diff_ratio * 100:.1f}%）"
                ),
            }
        return {
            "conflict": False,
            "conflict_type": "无冲突",
            "reason": f"双方数值一致（归一化后均为 {val_a:g}）",
        }

    # 2. 无数值场景：文本相似度
    a_norm = _normalize_text(text_a)
    b_norm = _normalize_text(text_b)
    if not a_norm or not b_norm:
        return {
            "conflict": False,
            "conflict_type": "无冲突",
            "reason": "双方文本均无数值且无可比文本，默认不判冲突",
        }
    lcs_len = _longest_common_substring(a_norm, b_norm)
    similarity = lcs_len / max(max(len(a_norm), len(b_norm)), 1)
    if similarity > 0.7:
        return {
            "conflict": False,
            "conflict_type": "无冲突",
            "reason": f"文本高度相似（LCS 占比 {similarity:.2f}）",
        }
    return {
        "conflict": True,
        "conflict_type": "疑似冲突",
        "reason": "双方无数值可比较且文本不相似，疑似规定不一致，需人工复核",
    }


def _first_normalized_value(facts: list[dict]) -> float | None:
    """取第一个 fact 并单位归一化到基准单位（元/MWh 或 MWh）。"""
    for fact in facts:
        value = fact.get("value")
        unit = fact.get("unit", "")
        if value is None:
            continue
        if unit in _ALL_ENERGY_UNITS:
            conv = unit_converter(value, unit, "元/MWh" if "元" in unit or "分" in unit else "MWh")
            if conv.get("success"):
                return float(conv["data"]["value"])
        return float(value)
    return None
