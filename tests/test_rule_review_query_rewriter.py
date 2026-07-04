"""
规则审查系统问题改写模块单元测试

覆盖 src/rule_review/query_rewriter.py 的时间标准化、实体名标准化、术语标准化。
"""

from datetime import date

import pytest

from src.rule_review.query_rewriter import QueryRewriter


@pytest.fixture
def rewriter():
    """使用固定基准日期（2026-07-05，周日）的改写器，保证测试结果可复现。"""
    return QueryRewriter(reference_date=date(2026, 7, 5))


class TestTimeNormalization:
    """时间标准化测试"""

    def test_absolute_date_year_month_day(self, rewriter):
        query = "2025年3月15日冀北的日前电价"
        assert rewriter.rewrite(query) == "2025-03-15冀北的日前现货出清电价"

    def test_absolute_date_with_slash(self, rewriter):
        query = "2025/3/15冀北的日前电价"
        assert rewriter.rewrite(query) == "2025-03-15冀北的日前现货出清电价"

    def test_yesterday(self, rewriter):
        query = "昨天冀北的日前电价"
        assert rewriter.rewrite(query) == "2026-07-04冀北的日前现货出清电价"

    def test_today(self, rewriter):
        query = "今天冀北的日前电价"
        assert rewriter.rewrite(query) == "2026-07-05冀北的日前现货出清电价"

    def test_this_week(self, rewriter):
        query = "本周的日前电价"
        assert rewriter.rewrite(query) == "2026-06-29 至 2026-07-05的日前现货出清电价"

    def test_last_week(self, rewriter):
        query = "上周的日前电价"
        assert rewriter.rewrite(query) == "2026-06-22 至 2026-06-28的日前现货出清电价"

    def test_next_week(self, rewriter):
        query = "下周的日前电价"
        assert rewriter.rewrite(query) == "2026-07-06 至 2026-07-12的日前现货出清电价"

    def test_this_month(self, rewriter):
        query = "本月的日前电价"
        assert rewriter.rewrite(query) == "2026-07-01 至 2026-07-31的日前现货出清电价"

    def test_this_year(self, rewriter):
        query = "今年的日前电价"
        assert rewriter.rewrite(query) == "2026的日前现货出清电价"

    def test_holiday_with_year(self, rewriter):
        query = "2026年劳动节的价格上限"
        assert rewriter.rewrite(query) == "劳动节（2026-05-01 至 2026-05-05）的价格上限"

    def test_holiday_without_year(self, rewriter):
        query = "劳动节的日前电价"
        assert rewriter.rewrite(query) == "劳动节（2026-05-01 至 2026-05-05）的日前现货出清电价"


class TestEntityNormalization:
    """实体名标准化测试"""

    def test_entity_alias_jibei(self, rewriter):
        query = "冀北电网的日前电价"
        assert rewriter.rewrite(query) == "冀北的日前现货出清电价"

    def test_entity_alias_sichuan(self, rewriter):
        query = "四川电网的日前电价"
        assert rewriter.rewrite(query) == "四川主网的日前现货出清电价"

    def test_entity_alias_gansu(self, rewriter):
        query = "甘肃的日前电价"
        assert rewriter.rewrite(query) == "甘肃东部和甘肃西部的日前现货出清电价"

    def test_standard_name_preserved(self, rewriter):
        query = "冀北的日前电价"
        assert rewriter.rewrite(query) == "冀北的日前现货出清电价"


class TestTermNormalization:
    """术语标准化测试"""

    def test_term_day_ahead_price(self, rewriter):
        query = "冀北的日前电价"
        assert rewriter.rewrite(query) == "冀北的日前现货出清电价"

    def test_term_price_cap(self, rewriter):
        query = "冀北的上限是多少"
        assert rewriter.rewrite(query) == "冀北的价格上限是多少"

    def test_standard_term_not_doubled(self, rewriter):
        """价格上限本身已是标准术语，不应被重复改写为 价格价格上限。"""
        query = "冀北的价格上限是多少"
        assert rewriter.rewrite(query) == "冀北的价格上限是多少"

    def test_term_emergency_dispatch(self, rewriter):
        query = "应急交易的时间"
        assert rewriter.rewrite(query) == "省间应急调度交易的时间"


class TestFullRewrite:
    """完整改写链路测试"""

    def test_full_example(self, rewriter):
        query = "2025年3月15日冀北电网日前电价达到800元/MWh，是否符合价格上限？"
        expected = "2025-03-15冀北日前现货出清电价达到800元/MWh，是否符合价格上限？"
        assert rewriter.rewrite(query) == expected

    def test_no_match_preserved(self, rewriter):
        query = "hello world"
        assert rewriter.rewrite(query) == "hello world"


class TestQueryRewriterInitialization:
    """初始化与边界测试"""

    def test_default_reference_date_is_today(self):
        rewriter = QueryRewriter()
        assert rewriter.reference_date == date.today()

    def test_empty_query(self, rewriter):
        assert rewriter.rewrite("") == ""

    def test_load_missing_file_gracefully(self):
        """缺失配置文件时不应崩溃，未命中的 token 保持原样。"""
        rewriter = QueryRewriter(
            reference_date=date(2026, 7, 5),
            term_config_path="/nonexistent/rule_terms.json",
            data_standard_path="/nonexistent/data_standard.json",
            holidays_path="/nonexistent/holidays.json",
        )
        # 使用一个不在标准地名库中的 token，确保不因配置缺失而崩溃
        assert rewriter.rewrite("未知地区的日前电价") == "未知地区的日前电价"
