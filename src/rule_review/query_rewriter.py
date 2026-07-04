"""
电力规则审查系统 - 问题改写器

按设计文档 Phase 1 步骤 1.2 实现：
- 时间标准化（绝对日期、相对时间、节假日）
- 实体名标准化（地名归一化，复用 data_standard.json）
- 术语标准化（rule_terms.json 别名映射）

不调用 LLM，纯 Python 规则实现。
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timedelta
from typing import Any

from src.utils.data_standard import data_matching


# 匹配连续中文/数字/字母的 token，用于实体模糊匹配
_TOKEN_RE = re.compile(r"[一-龥A-Za-z0-9]+")


class QueryRewriter:
    """规则审查问题改写器。

    将用户原始问题标准化，便于后续检索与 LLM 推理。
    """

    def __init__(
        self,
        reference_date: date | None = None,
        term_config_path: str = "data/env_variables/rule_terms.json",
        data_standard_path: str = "data/env_variables/data_standard.json",
        holidays_path: str = "data/knowledge/holidays.json",
    ) -> None:
        """
        Args:
            reference_date: 用于相对时间计算的基准日期，默认使用当天。
            term_config_path: 术语与实体别名映射配置。
            data_standard_path: 标准地名库（name_abbreviation）。
            holidays_path: 节假日 JSON 库。
        """
        self.reference_date = reference_date or date.today()
        self.term_config_path = term_config_path
        self.data_standard_path = data_standard_path
        self.holidays_path = holidays_path

        self.term_config = self._load_json(term_config_path)
        self.data_standard = self._load_json(data_standard_path)
        self.holidays = self._load_json(holidays_path)

        self.standard_names: list[str] = self.data_standard.get("name_abbreviation", [])

        # 实体别名：alias -> standard
        self.entity_aliases: dict[str, str] = (
            self.term_config.get("entities", {}).get("aliases", {})
        )
        # 术语别名：alias -> standard term
        self.term_aliases: dict[str, str] = {}
        for term, info in self.term_config.get("terms", {}).items():
            for alias in info.get("aliases", []):
                self.term_aliases[alias] = term

        # 按长度降序排列，优先匹配更长、更精确的别名
        self.entity_alias_items = sorted(
            self.entity_aliases.items(), key=lambda x: len(x[0]), reverse=True
        )
        self.term_alias_items = sorted(
            self.term_aliases.items(), key=lambda x: len(x[0]), reverse=True
        )

    @staticmethod
    def _load_json(path: str) -> Any:
        if not os.path.isabs(path):
            # 以项目根目录为基准；测试与运行均在项目根目录执行
            path = os.path.join(os.getcwd(), path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def rewrite(self, query: str) -> str:
        """执行完整的问题改写。"""
        query = self._normalize_time(query)
        query = self._normalize_entities(query)
        query = self._normalize_terms(query)
        return query

    # ------------------------------------------------------------------
    # 0. 时间标准化
    # ------------------------------------------------------------------
    def _normalize_time(self, query: str) -> str:
        query = self._normalize_absolute_dates(query)
        query = self._normalize_relative_dates(query)
        query = self._normalize_weeks(query)
        query = self._normalize_month(query)
        query = self._normalize_year(query)
        query = self._normalize_holidays(query)
        return query

    def _normalize_absolute_dates(self, query: str) -> str:
        """将 '2025年3月15日' / '2025-3-15' / '2025/3/15' 统一为 YYYY-MM-DD。"""

        def repl(m: re.Match) -> str:
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return f"{year}-{month:02d}-{day:02d}"

        return re.sub(
            r"(\d{4})[年\\/-](\d{1,2})[月\\/-](\d{1,2})日?",
            repl,
            query,
        )

    def _normalize_relative_dates(self, query: str) -> str:
        """处理 今天/昨天/前天 等相对日期。"""
        today = self.reference_date
        mapping = {
            "今天": today,
            "今日": today,
            "昨天": today - timedelta(days=1),
            "昨日": today - timedelta(days=1),
            "前天": today - timedelta(days=2),
            "前日": today - timedelta(days=2),
        }
        for word, dt in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
            query = self._replace_word(query, word, dt.strftime("%Y-%m-%d"))
        return query

    def _normalize_weeks(self, query: str) -> str:
        """处理 本周/上周/下周，统一为 'YYYY-MM-DD 至 YYYY-MM-DD'。"""
        today = self.reference_date
        weekday = today.weekday()  # Monday=0
        this_monday = today - timedelta(days=weekday)
        weeks = {
            "本周": (this_monday, this_monday + timedelta(days=6)),
            "这周": (this_monday, this_monday + timedelta(days=6)),
            "上周": (this_monday - timedelta(days=7), this_monday - timedelta(days=1)),
            "下周": (this_monday + timedelta(days=7), this_monday + timedelta(days=13)),
        }
        for word, (start, end) in weeks.items():
            range_str = f"{start.strftime('%Y-%m-%d')} 至 {end.strftime('%Y-%m-%d')}"
            query = self._replace_word(query, word, range_str)
        return query

    def _normalize_month(self, query: str) -> str:
        """处理 '本月' / '这个月'。"""
        today = self.reference_date
        start = today.replace(day=1)
        if start.month == 12:
            end = date(start.year + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(start.year, start.month + 1, 1) - timedelta(days=1)
        range_str = f"{start.strftime('%Y-%m-%d')} 至 {end.strftime('%Y-%m-%d')}"
        for word in ["本月", "这个月"]:
            query = self._replace_word(query, word, range_str)
        return query

    def _normalize_year(self, query: str) -> str:
        """处理 '今年' / '这一年'。"""
        for word in ["今年", "这一年"]:
            query = self._replace_word(query, word, str(self.reference_date.year))
        return query

    def _normalize_holidays(self, query: str) -> str:
        """将 '2026年劳动节' 替换为 '劳动节（YYYY-MM-DD 至 YYYY-MM-DD）'。"""
        holidays_by_year = self.holidays.get("holidays", {})
        all_names = set()
        for year_data in holidays_by_year.values():
            all_names.update(year_data.keys())
        if not all_names:
            return query

        names_pattern = "|".join(re.escape(n) for n in sorted(all_names, key=len, reverse=True))
        pattern = re.compile(r"(\d{4})?年?(" + names_pattern + r")")

        def repl(m: re.Match) -> str:
            year_str = m.group(1)
            name = m.group(2)
            year = int(year_str) if year_str else self.reference_date.year
            ranges = holidays_by_year.get(str(year), {})
            date_range = ranges.get(name)
            if not date_range:
                # 未找到对应年份时回退到当前年
                year = self.reference_date.year
                ranges = holidays_by_year.get(str(year), {})
                date_range = ranges.get(name, "")
            if not date_range:
                return m.group(0)
            return f"{name}（{date_range}）"

        return pattern.sub(repl, query)

    @staticmethod
    def _replace_word(query: str, word: str, replacement: str) -> str:
        """直接替换词汇。替换结果中不会再次包含该词汇，因此不会级联重复。"""
        return query.replace(word, replacement)

    # ------------------------------------------------------------------
    # 1. 实体名标准化
    # ------------------------------------------------------------------
    def _normalize_entities(self, query: str) -> str:
        # 先通过配置的精确别名替换，保护标准名不被破坏
        protected = list(self.standard_names) + list(self.entity_aliases.values())
        query = self._replace_aliases_with_protection(
            query, self.entity_alias_items, protected
        )
        # 再对未命中的 token 做 difflib 模糊匹配
        query = self._fuzzy_normalize_entities(query)
        return query

    # 候选实体段的分隔符：常见虚词、标点、空白
    _ENTITY_DELIM_RE = re.compile(
        r"([的是在和与或等中下上前后内外里间及以及如而但都也还很到从向把被让给跟同比为有无不过"
        r"了吗呢吧啊嘛地得，。、；：？！\"\"''（）【】《》\s]+)"
    )

    def _fuzzy_normalize_entities(self, query: str) -> str:
        """使用 difflib 对未命中的候选实体段做模糊归一化。"""
        parts = self._ENTITY_DELIM_RE.split(query)
        out: list[str] = []
        for part in parts:
            if not part:
                continue
            # 分隔符直接保留
            if self._ENTITY_DELIM_RE.fullmatch(part):
                out.append(part)
                continue
            if len(part) < 2:
                out.append(part)
                continue
            matched = data_matching(part, "name_abbreviation")
            out.append(matched if matched else part)
        return "".join(out)

    # ------------------------------------------------------------------
    # 2. 术语标准化
    # ------------------------------------------------------------------
    def _normalize_terms(self, query: str) -> str:
        protected = list(self.term_config.get("terms", {}).keys())
        return self._replace_aliases_with_protection(
            query, self.term_alias_items, protected
        )

    # ------------------------------------------------------------------
    # 通用别名替换（带保护）
    # ------------------------------------------------------------------
    @staticmethod
    def _replace_aliases_with_protection(
        query: str,
        alias_items: list[tuple[str, str]],
        protected_terms: list[str],
    ) -> str:
        """
        一次性正则 alternation 完成别名替换，同时把标准术语当作“恒等替换”保护起来。

        关键点：
        - 所有 key（标准术语 + 别名）按长度降序排列，长匹配优先命中。
        - 因此不会出现短别名破坏已命中的长别名/标准术语的情况。
        - 也无需额外的占位符-恢复步骤，避免占位符误伤别名匹配。
        """
        mapping: dict[str, str] = {term: term for term in protected_terms if term}
        mapping.update(dict(alias_items))
        if not mapping:
            return query

        sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(k) for k in sorted_keys))
        return pattern.sub(lambda m: mapping[m.group(0)], query)


# 提供一个便捷的默认实例，方便 pipeline 直接使用
_default_rewriter: QueryRewriter | None = None


def get_default_rewriter() -> QueryRewriter:
    """返回默认 QueryRewriter 单例（惰性初始化）。"""
    global _default_rewriter
    if _default_rewriter is None:
        _default_rewriter = QueryRewriter()
    return _default_rewriter
