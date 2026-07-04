"""
聚合计算工具函数库

每个聚合操作提供两个版本：
- 普通函数（raw_*）：供 AggregationAgent 直接调用，无任何依赖
- @tool 装饰版本：保留备用，供未来大模型工具调用场景使用
"""
import json
import re
from datetime import datetime, timedelta, timezone
from calendar import monthrange
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any
from langchain_core.tools import tool
from src.utils.fuzzy_match import fuzzy_match, get_fuzzy_match_type, is_fuzzy_field


def _to_decimal(value):
    """将值安全转换为 Decimal，用于精确数值累加。"""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return Decimal(text)
        except InvalidOperation:
            return None
    return None


def _round_half_up(value, ndigits: int) -> Decimal:
    """传统四舍五入（round half up），替代 Python round() 的银行家舍入。

    返回 Decimal 以保证精确的四舍五入结果，避免 float 转换引入的精度问题。
    """
    if isinstance(value, (int, float)):
        value = Decimal(str(value))
    quantize_str = '0.' + '0' * ndigits
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


with open("data/env_variables/time_point_mapping.json", "r", encoding="utf-8") as f:
    _time_point_mapping = json.load(f)
_ALL_TIME_POINT_FIELDS = {k:v for k, v in _time_point_mapping.items() if k != "aliases"}
_DATE_ONLY_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# 将输电、日前水力预测出电、水力总出电数据转化
def _handledData(data: list,field: str):
    fields = field.split(",")
    handled_data = []  # 修复拼写错误

    for data_item in data:
        if isinstance(data_item, dict):
            base_dict = {k: v for k, v in data_item.items() if k not in _ALL_TIME_POINT_FIELDS}

            for f in fields:
                new_dict = base_dict.copy()
                new_dict["moment"] = _ALL_TIME_POINT_FIELDS.get(f, f)
                new_dict["value"] = data_item.get(f, 0)

                handled_data.append(new_dict)

    return handled_data, "value"


# ─────────────────────────────────────────────
# 普通函数（核心逻辑）
# ─────────────────────────────────────────────

def raw_filter(data: list, field: str, operator: str, value: Any, domain: str = "") -> list:
    """按条件过滤数据。

    支持的操作符：>, <, >=, <=, ==, =, !=, between, in, contains, startswith, endswith

    字符串字段匹配规则（支持模糊匹配）：
    - == 或 = ：精确匹配、前缀匹配（日期）、或模糊匹配（包含）
    - != ：不匹配上述任何规则
    - contains ：模糊匹配（包含），用于 voltageName 等字段
    - startswith ：前缀匹配，用于 caseId 等字段（如 "00" 开头）
    - endswith ：后缀匹配（可选，目前未用）

    模糊匹配支持：
    - voltageName 字段值为 "±800kV"，用户输入 "800" 时匹配（数值模糊匹配）
    - name 字段值为 "南北联络线"，用户输入 "南北" 时匹配（包含匹配）
    - businessTime 字段值为 "2025-01-01T23:50:00.000+00:00"，用户输入 "2025-01-01" 时匹配（前缀匹配）

    Args:
        data: 数据列表
        field: 字段名
        operator: 操作符
        value: 比较值
        domain: 业务域（transmission/transaction），用于查找模糊匹配配置
    """
    result = []

    # 兼容模型常见的英文比较操作符写法
    operator_aliases = {
        "gt": ">",
        "lt": "<",
        "gte": ">=",
        "ge": ">=",
        "lte": "<=",
        "le": "<=",
        "eq": "==",
        "neq": "!=",
        "ne": "!=",
    }
    operator = operator_aliases.get(operator, operator)
    # 支持 value='field:xxx' 格式，表示与另一个字段的值做比较（字段间比较）
    _compare_field = None
    if isinstance(value, str) and value.startswith('field:'):
        _compare_field = value[6:]  # 取出被比较字段名

    for item in data:
        if not isinstance(item, dict):
            continue

        v = item.get(field)
        if v is None:
            continue

        # 字段间比较：将 value 替换为当前记录中另一字段的值
        if _compare_field:
            actual_value = item.get(_compare_field)
            if actual_value is None:
                continue  # 比较字段不存在则跳过
            cmp_value = actual_value  # 用当前行的字段值作为比较阈值（局部变量，不污染外层 value）
        else:
            cmp_value = value

        # 处理 startswith 操作符
        if operator == 'startswith':
            str_v = str(v) if v is not None else ''
            str_val = str(cmp_value) if cmp_value is not None else ''
            if str_v.startswith(str_val):
                result.append(item)
            continue

        # 处理 endswith 操作符（可选，目前未用）
        if operator == 'endswith':
            str_v = str(v) if v is not None else ''
            str_val = str(cmp_value) if cmp_value is not None else ''
            if str_v.endswith(str_val):
                result.append(item)
            continue

        # 先尝试字符串等值/不等值匹配（兼容 == 和 = 两种写法）
        if operator in ('==', '=', '!=', 'contains'):
            # 日期等值匹配兜底：
            # 当用户给的是 YYYY-MM-DD 且记录值是 ISO UTC（如 businessTime）时，
            # 先按北京时间归一化到日期后再比较，避免跨时区导致“当天数据被误过滤”。
            if isinstance(cmp_value, str) and _DATE_ONLY_PATTERN.match(cmp_value.strip()):
                date_cmp = cmp_value.strip()
                date_v = _normalize_to_beijing_date(v)
                if date_v is not None:
                    if operator in ('==', '=') and date_v == date_cmp:
                        result.append(item)
                        continue
                    if operator == '!=' and date_v != date_cmp:
                        result.append(item)
                        continue

            str_v = str(v)
            str_val = str(cmp_value)

            # 使用模糊匹配工具（支持配置化的匹配策略）
            match_type = get_fuzzy_match_type(field, domain)
            str_match = fuzzy_match(v, cmp_value, match_type, field, domain)

            if operator in ('==', '=', 'contains'):
                if str_match:
                    result.append(item)
                    continue
                # 再尝试数值比较
                try:
                    if float(v) == float(cmp_value):
                        result.append(item)
                except (TypeError, ValueError):
                    pass
            else:  # !=
                if not str_match:
                    result.append(item)
                    continue
                try:
                    if float(v) != float(cmp_value):
                        result.append(item)
                except (TypeError, ValueError):
                    result.append(item)  # 无法数值比较时，字符串不等即保留
            continue

        # 比较操作符
        try:
            fv = parse_param_for_compare(v)
            cmp_value = parse_param_for_compare(cmp_value) if isinstance(cmp_value, str) else cmp_value
            match = False
            if operator == ">":
                match = fv > cmp_value
            elif operator == "<":
                match = fv < cmp_value
            elif operator == ">=":
                match = fv >= cmp_value
            elif operator == "<=":
                match = fv <= cmp_value
            elif operator == "between":
                if isinstance(cmp_value, (list, tuple)) and len(cmp_value) == 2:
                    lower = parse_param_for_compare(cmp_value[0]) if isinstance(cmp_value[0], str) else cmp_value[0]
                    upper = parse_param_for_compare(cmp_value[1]) if isinstance(cmp_value[1], str) else cmp_value[1]
                    match = lower <= fv <= upper
            elif operator == "in":
                if isinstance(cmp_value, (list, tuple)):
                    match = fv in [parse_param_for_compare(vi) for vi in cmp_value]
            if match:
                result.append(item)
        except (TypeError, ValueError):
            pass

    return result


import re  # 若文件顶部未导入需加上


def raw_count_if(data: list, fields: list, operator: str, value: Any) -> dict:
    """统计多个字段中满足条件的字段个数，并附带元数据与命中详情。"""
    count = 0
    matched_values = []
    matched_details = []

    for item in data:
        if not isinstance(item, dict):
            continue
        record_matches = []
        for f in fields:
            v = item.get(f)
            if v is None:
                continue
            try:
                fv = float(v)
            except (ValueError, TypeError):
                continue
            match = False
            if operator == '>':
                match = fv > float(value)
            elif operator == '<':
                match = fv < float(value)
            elif operator == '>=':
                match = fv >= float(value)
            elif operator == '<=':
                match = fv <= float(value)
            elif operator == '==':
                match = fv == float(value)
            elif operator == '!=':
                match = fv != float(value)
            elif operator == 'between':
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    low, high = float(value[0]), float(value[1])
                    match = low <= fv <= high
            if match:
                count += 1
                matched_values.append(fv)
                record_matches.append({"field": f, "value": fv})
        if record_matches:
            trace = {k: v for k, v in item.items()
                     if not re.match(r'^v\d{4}$', k)}
            trace["matched_items"] = record_matches
            matched_details.append(trace)

    return {
        "count": count,
        "result": matched_values,
        "matched_details": matched_details,
    }

def parse_param_for_compare(value):
    """
    通用转换方法：自动把值转为 可比较的格式
    - 能转数字 → 转 float
    - 能转时间 → 转 datetime
    - 都不行 → 返回原值
    支持的时间格式：
    2025-03-15、2025/03/15、2025-03-15 08:00、2025-03-15 08:00:00
    """
    if value is None:
        return None

    # 先尝试转 数字
    try:
        return float(value)
    except (ValueError, TypeError):
        pass

    # 再尝试转 时间
    time_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
    ]
    for fmt in time_formats:
        try:
            return datetime.strptime(str(value).strip(), fmt)
        except (ValueError, TypeError):
            continue

    # 再尝试 ISO 8601（统一转换到北京时间并去掉时区）
    try:
        value_str = str(value).strip().replace("Z", "+00:00")
        if "T" in value_str:
            dt = datetime.fromisoformat(value_str)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone(timedelta(hours=8))).replace(tzinfo=None)
            return dt
    except (ValueError, TypeError):
        pass

    # 都不行，返回字符串
    return str(value).strip()


def _normalize_datetime_text(value: Any) -> str:
    """将 ISO 8601 UTC 时间转换为北京时间文本，其他格式原样返回。"""
    value_str = str(value).strip()
    if 'T' not in value_str:
        return value_str

    try:
        dt = datetime.fromisoformat(value_str.replace("Z", "+00:00"))
    except ValueError:
        return value_str

    if dt.tzinfo is None:
        return value_str

    beijing_tz = timezone(timedelta(hours=8))
    return dt.astimezone(beijing_tz).strftime("%Y-%m-%d %H:%M:%S")


def _resolve_sort_metric(item: Any, field: str) -> float | None:
    """将记录字段值转为可排序数值（支持数值/日期），无法解析返回 None。"""
    if not isinstance(item, dict):
        return None

    value = item.get(field)
    if value is None:
        return None

    parsed = parse_param_for_compare(value)
    if isinstance(parsed, (int, float)):
        return float(parsed)
    if isinstance(parsed, datetime):
        return parsed.timestamp()
    return None


def _sort_records_by_field(data: list, field: str, descending: bool = False) -> list:
    """按字段排序：支持数值与日期；无法解析值放到末尾并保持原顺序。"""
    valid_records = []
    invalid_records = []

    for item in data:
        metric = _resolve_sort_metric(item, field)
        if metric is None:
            invalid_records.append(item)
        else:
            valid_records.append((metric, item))

    # Python sort 稳定：相同 metric 会保持原始输入顺序
    valid_sorted = sorted(valid_records, key=lambda x: x[0], reverse=descending)
    return [item for _, item in valid_sorted] + invalid_records

def raw_top_n(data: list, field: str, n: int, order: str = "desc") -> list:
    """取前N条记录（按指定字段排序）。

    order: "desc" 表示降序（最大值优先），"asc" 表示升序（最小值优先）
    """
    if not data or n <= 0:
        return data

    if "," in field:
        data, field = _handledData(data,field)

    # 按指定字段排序（支持数值与日期）
    sorted_data = _sort_records_by_field(data, field, descending=(order == "desc"))

    return sorted_data[:n]


def raw_bottom_n(data: list, field: str, n: int, order: str = "asc") -> list:
    """取最小的N条记录（按指定字段升序排序后取前N条）。

    排序时排除字段值为 0 的记录（0 视为无效数据）。

    order: 保留参数以兼容接口，实际始终按升序取最小值
    """
    if not data or n <= 0:
        return data

    if "," in field:
        data, field = _handledData(data,field)

    # 排除字段值为 0 的记录后排序
    nonzero_data = []
    for item in data:
        metric = _resolve_sort_metric(item, field)
        if metric is None or metric == 0:
            continue
        nonzero_data.append(item)

    sorted_data = _sort_records_by_field(nonzero_data, field, descending=False)

    return sorted_data[:n]


def raw_group_by(data: list, group_field: str, agg_field, agg_op: str = "sum", date_granularity: str = "month", domain: str = "") -> list:
    """按指定字段分组，对每组执行聚合操作。

    支持模糊分组：对于字符串分组字段，使用模糊匹配将相似的值归为一组。
    例如：voltageName 字段中 "±800kV" 和 "800" 会被归为同一组。

    Args:
        data: 原始数据列表
        group_field: 分组字段（如 beginDate、endDate、tradeseqCaption 等）
        agg_field: 聚合字段，支持以下形式：
                   - 单个字段名字符串（如 "bidEnergy"）
                   - 多字段逗号分隔字符串（如 "v0015,v0030,v0045"）→ 每条记录先对所有字段求和再聚合
                   - 字段名列表（如 ["v0015", "v0030"]）→ 同上
        agg_op: 聚合操作（sum/average/count/max/min）
        date_granularity: 日期分组粒度，仅当 group_field 为日期字段时生效
                         - "year"          : 按年分组，截取 YYYY（4位）
                         - "month"         : 按月分组，截取 YYYY-MM（7位，默认）
                         - "week"          : 按ISO周分组，格式 YYYY-Www（如 2025-W13）
                         - "week_in_month" : 按月内周段分组（1-7、8-14、15-21、22-28、29-月末）
                         - "day"           : 按日分组，截取 YYYY-MM-DD（10位，即不截断）
        domain: 业务域（transmission/transaction），用于模糊匹配

    Returns:
        分组聚合后的数据列表，每条记录包含分组键和聚合结果
    """
    if not data:
        return data

    from src.utils.logging_setup import logger

    # 判断是否为仅分组模式（不聚合）
    no_aggregation = agg_op is None or agg_op == "none"

    # 解析 agg_field：支持单字段字符串、逗号分隔多字段字符串、字段名列表
    if not no_aggregation:
        # 聚合模式下，解析 agg_field（原有逻辑不变）
        if isinstance(agg_field, list):
            agg_fields = [f.strip() for f in agg_field if f.strip()]
        elif isinstance(agg_field, str) and ',' in agg_field:
            agg_fields = [f.strip() for f in agg_field.split(',') if f.strip()]
        else:
            agg_fields = [agg_field] if agg_field else []

        is_multi_field = len(agg_fields) > 1

        if agg_op == "field_max" or agg_op == "field_min":
            result_field_name = f"{agg_op}_value"  # 与意图识别中后续 max 操作的 field 一致
        elif is_multi_field:
            result_field_name = f"{agg_op}_fields_sum"
        else:
            result_field_name = f"{agg_op}_{agg_field}"

        logger.info(f"[raw_group_by] 聚合模式 - is_multi_field={is_multi_field}, agg_fields={agg_fields}")

    # 日期字段粒度截取长度映射
    _DATE_GRANULARITY_LEN = {
        "year": 4,
        "month": 7,
        "day": 10,
    }

    def _get_week_key(date_str: str) -> str:
        """将日期字符串转换为 ISO 周标识，如 '2025-03-29' -> '2025-W13'。"""
        try:
            parsed = datetime.strptime(date_str[:10], "%Y-%m-%d")
            iso_year, iso_week, _ = parsed.isocalendar()
            return f"{iso_year}-W{iso_week:02d}"
        except (ValueError, TypeError):
            return date_str[:7]  # 解析失败时降级为按月

    def _get_week_in_month_key(date_str: str) -> str:
        """将日期字符串转换为“月内周段”标识，如 '2025-04-03' -> '2025-04-01~07'。"""
        try:
            parsed = datetime.strptime(date_str[:10], "%Y-%m-%d")
            start_day = ((parsed.day - 1) // 7) * 7 + 1
            end_day = min(start_day + 6, monthrange(parsed.year, parsed.month)[1])
            return f"{parsed.year:04d}-{parsed.month:02d}-{start_day:02d}~{end_day:02d}"
        except (ValueError, TypeError):
            return date_str[:7]  # 解析失败时降级为按月

    def _get_quarter_key(date_str: str) -> str:
        """将日期字符串转换为季度标识，如 '2025-01-15' -> '2025-Q1'"""
        try:
            year = date_str[:4]
            month = int(date_str[5:7]) if len(date_str) >= 7 else 1
            quarter = (month - 1) // 3 + 1
            return f"{year}-Q{quarter}"
        except (ValueError, IndexError):
            return date_str[:7]  # 解析失败时降级为按月

    # 分组循环（保留原有分组逻辑，包括日期处理、跨期间跳过等）
    groups = {}
    is_date_field = group_field in ('beginDate', 'endDate', 'businessTime', 'rq')
    if is_date_field and not no_aggregation:
        logger.info(f"[raw_group_by] 检测到日期字段 {group_field}，按粒度 {date_granularity} 分组")

    for item in data:
        if not isinstance(item, dict):
            continue

        group_key = item.get(group_field)
        if group_key is None:
            continue

        # 日期字段处理（原有逻辑，保持不变）
        group_key_str = _normalize_datetime_text(group_key)
        if is_date_field:
            if date_granularity == 'quarter':
                group_key_str = _get_quarter_key(group_key_str)
                end_date_val = item.get('endDate')
                if end_date_val is not None:
                    end_quarter_key = _get_quarter_key(_normalize_datetime_text(end_date_val))
                    if end_quarter_key != group_key_str:
                        logger.debug(f"[raw_group_by] 跨季度记录被跳过...")
                        continue
            elif date_granularity == 'week':
                group_key_str = _get_week_key(group_key_str)
                end_date_val = item.get('endDate')
                if end_date_val is not None:
                    end_week_key = _get_week_key(_normalize_datetime_text(end_date_val))
                    if end_week_key != group_key_str:
                        logger.debug(f"[raw_group_by] 跨周记录被跳过...")
                        continue
            elif date_granularity == 'week_in_month':
                group_key_str = _get_week_in_month_key(group_key_str)
                end_date_val = item.get('endDate')
                if end_date_val is not None:
                    end_week_in_month_key = _get_week_in_month_key(_normalize_datetime_text(end_date_val))
                    if end_week_in_month_key != group_key_str:
                        logger.debug(f"[raw_group_by] 跨月内周段记录被跳过...")
                        continue
            else:
                cut_len = _DATE_GRANULARITY_LEN.get(date_granularity, 7)
                if len(group_key_str) >= cut_len:
                    group_key_str = group_key_str[:cut_len]
                end_date_val = item.get('endDate')
                if end_date_val is not None:
                    end_key_str = _normalize_datetime_text(end_date_val)
                    if len(end_key_str) >= cut_len:
                        end_key_str = end_key_str[:cut_len]
                    if end_key_str != group_key_str:
                        logger.debug(f"[raw_group_by] 跨期间记录被跳过...")
                        continue
        else:
            # 非日期字段精确匹配分组
            found_key = None
            for existing_key in groups.keys():
                if existing_key == group_key_str:
                    found_key = existing_key
                    break
            if found_key:
                group_key_str = found_key

        # 初始化分组结构（统一使用字典存储 items 和可选的 values）
        if group_key_str not in groups:
            groups[group_key_str] = {
                "group_key": group_key_str,
                "items": [],
                "values": [] if not no_aggregation else None  # 仅分组模式不需要 values
            }

        groups[group_key_str]["items"].append(item)

        # 聚合模式：收集字段值
        if not no_aggregation:
            if is_multi_field:
                row_sum = Decimal('0')
                has_value = False
                for f in agg_fields:
                    v = item.get(f)
                    if v is not None:
                        d = _to_decimal(v)
                        if d is not None:
                            row_sum += d
                            has_value = True
                if has_value:
                    if agg_op == "min" and row_sum == 0:
                        continue
                    groups[group_key_str]["values"].append(row_sum)
            else:
                agg_value = item.get(agg_fields[0]) if agg_fields else None
                if agg_value is not None:
                    d = _to_decimal(agg_value)
                    if d is not None:
                        if agg_op == "min" and d == 0:
                            continue
                        groups[group_key_str]["values"].append(d)
    # 构建结果
    result = []
    for group_key_str, group_data in groups.items():
        if no_aggregation:
            # 仅分组模式：返回分组键 + items 数组
            result.append({
                group_field: group_data["group_key"],
                "items": group_data["items"],
                "count": len(group_data["items"])
            })
        else:
            # 聚合模式：原有逻辑
            values = group_data["values"]
            items_count = len(group_data["items"])
            best_fields = None
            best_times = None
            best_sources = None

            if agg_op == "count":
                agg_result = items_count
            elif not values:
                agg_result = None
            elif agg_op == "sum":
                agg_result = _round_half_up(sum(values), 4)
            elif agg_op == "average":
                if is_multi_field:
                    # 多字段均值：逐条记录、逐个字段收集非零值（Decimal 精确累加）
                    all_vals = []
                    for item in group_data["items"]:
                        for f in agg_fields:
                            v = item.get(f)
                            if v is not None:
                                d = _to_decimal(v)
                                if d is not None and not d.is_zero():
                                    all_vals.append(d)
                    agg_result = _round_half_up(sum(all_vals) / len(all_vals), 4) if all_vals else None
                else:
                    non_zero = [v for v in values if not v.is_zero()]
                    if non_zero:
                        agg_result = _round_half_up(sum(non_zero) / len(non_zero), 4)
                    else:
                        agg_result = None
            elif agg_op == "max":
                agg_result = max(values)
            elif agg_op == "min":
                agg_result = min(values)
            elif agg_op == "field_max":
                best_val = None
                best_fields = []
                best_times = []
                best_sources = []
                for item in group_data["items"]:
                    for f in agg_fields:
                        v = item.get(f)
                        if v is None:
                            continue
                        try:
                            fv = float(v)
                            if best_val is None or fv > best_val:
                                best_val = fv
                                best_fields = [f]
                                best_times = [item.get("businessTime")]
                                best_sources = [_strip_time_point_fields(item)]
                            elif fv == best_val:
                                best_fields.append(f)
                                best_times.append(item.get("businessTime"))
                                best_sources.append(_strip_time_point_fields(item))
                        except (TypeError, ValueError):
                            pass
                agg_result = best_val
            elif agg_op == "field_min":
                best_val = None
                best_fields = []
                best_times = []
                best_sources = []
                for item in group_data["items"]:
                    for f in agg_fields:
                        v = item.get(f)
                        if v is None:
                            continue
                        try:
                            fv = float(v)
                            if fv == 0:  # 排除 0 值（与 raw_min_field 行为一致）
                                continue
                            if best_val is None or fv < best_val:
                                best_val = fv
                                best_fields = [f]
                                best_times = [item.get("businessTime")]
                                best_sources = [_strip_time_point_fields(item)]
                            elif fv == best_val:
                                best_fields.append(f)
                                best_times.append(item.get("businessTime"))
                                best_sources.append(_strip_time_point_fields(item))
                        except (TypeError, ValueError):
                            pass
                agg_result = best_val


            else:
                agg_result = None


            result_item = {
                group_field: group_data["group_key"],
                result_field_name: agg_result,
            }
            if agg_op != "count":
                result_item["count"] = items_count
            if best_fields:
                result_item["best_fields"] = best_fields
                result_item["best_moments"] = [
                    f"{bf[1:3]}:{bf[3:5]}" if re.match(r"^v\d{4}$", bf) else bf
                    for bf in best_fields
                ]
                result_item["best_times"] = best_times
            if best_sources:
                result_item["best_sources"] = best_sources
            result.append(result_item)

    # 按分组键排序
    result = sorted(result, key=lambda x: str(x.get(group_field, "")))

    if no_aggregation:
        logger.info(f"[raw_group_by] 仅分组模式 - 分组字段={group_field}, 分组数={len(result)}")
    else:
        logger.info(
            f"[raw_group_by] 聚合模式 - 分组字段={group_field}, 聚合字段={agg_fields}, 聚合操作={agg_op}, 分组数={len(result)}")

    return result


def raw_sum_field(data: list, field) -> dict:
    """对数据列表中指定字段求和。

    field 可以是单个字段名（str），也可以是字段名列表（list）。
    当 field 为列表时，对每条记录先将所有字段值累加，再对所有记录求总和。
    """
    values = []
    # 统一成列表处理：支持 "a,b,c" 的逗号字符串，并自动清洗数据
    if isinstance(field, list):
        fields = field.copy()
    elif isinstance(field, str) and ',' in field:
        fields = [f.strip() for f in field.split(',') if f.strip()]
    else:
        fields = [field]
    for item in data:
        if not isinstance(item, dict):
            continue
        row_sum = Decimal('0')
        has_value = False
        for f in fields:
            v = item.get(f)
            if v is not None:
                d = _to_decimal(v)
                if d is not None:
                    row_sum += d
                    has_value = True
        if has_value:
            values.append(row_sum)
    if not values:
        return {"result": None, "field": field, "count": 0, "error": f"字段 {field} 无有效数值"}
    total = sum(values)
    return {"result": _round_half_up(total, 4), "field": field, "count": len(values)}


def raw_row_sum_fields(data: list, field, result_field: str = "sum_fields_sum") -> list:
    """对数据列表中每条记录按指定字段求和，并将结果写回到 result_field。

    field 支持单字段、逗号分隔多字段字符串或字段名列表。
    """
    if isinstance(field, list):
        fields = [f.strip() for f in field if isinstance(f, str) and f.strip()]
    elif isinstance(field, str) and ',' in field:
        fields = [f.strip() for f in field.split(',') if f.strip()]
    elif isinstance(field, str) and field.strip():
        fields = [field.strip()]
    else:
        fields = []

    result = []
    for item in data:
        new_item = dict(item) if isinstance(item, dict) else item
        if isinstance(item, dict):
            row_sum = 0.0
            has_value = False
            for f in fields:
                v = item.get(f)
                if v is not None:
                    try:
                        row_sum += float(v)
                        has_value = True
                    except (TypeError, ValueError):
                        pass
            new_item[result_field] = row_sum if has_value else None
        result.append(new_item)
    return result


def _normalize_to_beijing_date(value: Any) -> str | None:
    """将输入值归一化为北京时间日期 YYYY-MM-DD（无法解析返回 None）。"""
    if value is None:
        return None

    value_str = str(value).strip()
    if not value_str:
        return None

    # 已是日期
    if _DATE_ONLY_PATTERN.match(value_str):
        return value_str

    # 常见本地时间格式
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(value_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # ISO 8601（含时区）统一按北京时间转换
    try:
        dt = datetime.fromisoformat(value_str.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone(timedelta(hours=8)))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass

    # 兼容字符串前 10 位是日期的情况
    prefix = value_str[:10]
    if _DATE_ONLY_PATTERN.match(prefix):
        return prefix

    return None


def raw_average_field(data: list, field) -> dict:
    """对数据列表中指定字段求简单平均值。

    field 可以是单个字段名（str）、逗号分隔多字段字符串，或字段名列表（list）。
    当 field 为多字段时，对所有记录中的所有有效字段值统一求平均。
    """
    values = []

    if isinstance(field, list):
        fields = [f.strip() for f in field if isinstance(f, str) and f.strip()]
    elif isinstance(field, str) and ',' in field:
        fields = [f.strip() for f in field.split(',') if f.strip()]
    else:
        fields = [field]

    is_multi_field = len(fields) > 1

    for item in data:
        if not isinstance(item, dict):
            continue

        if is_multi_field:
            for f in fields:
                v = item.get(f)
                if v is not None:
                    d = _to_decimal(v)
                    if d is not None and not d.is_zero():
                        values.append(d)
        else:
            v = item.get(fields[0])
            if v is not None:
                d = _to_decimal(v)
                if d is not None and not d.is_zero():
                    values.append(d)

    if not values:
        return {"result": None, "field": field, "count": 0, "error": f"字段 {field} 无有效数值"}
    return {"result": _round_half_up(sum(values) / len(values), 4), "field": field, "count": len(values)}


def raw_weighted_average_field(data: list, value_field: str, weight_field: str) -> dict:
    """对数据列表中指定字段按权重字段计算加权平均值。"""
    weighted_sum = Decimal('0')
    weight_total = Decimal('0')
    count = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        v = item.get(value_field)
        w = item.get(weight_field)
        if v is not None and w is not None:
            dv = _to_decimal(v)
            if dv is None or dv.is_zero():
                continue
            dw = _to_decimal(w)
            if dw is None:
                continue
            weighted_sum += dv * dw
            weight_total += dw
            count += 1
    if weight_total.is_zero():
        return {"result": None, "value_field": value_field, "weight_field": weight_field,
                "count": 0, "error": "权重总和为0，无法计算加权平均"}
    return {
        "result": _round_half_up(weighted_sum / weight_total, 4),
        "value_field": value_field,
        "weight_field": weight_field,
        "count": count,
    }


def _resolve_multi_field_value(item: dict, field: str) -> float | None:
    """从单条记录中解析字段值，支持逗号分隔多字段（求和）或字段名列表。

    - 单字段：直接取值
    - 逗号分隔字符串（如 'v0800,v0815'）：将多个字段值求和
    - 列表（如 ['v0800', 'v0815']）：将多个字段值求和
    """
    if isinstance(field, list):
        fields = field
    elif isinstance(field, str) and ',' in field:
        fields = [f.strip() for f in field.split(',') if f.strip()]
    else:
        v = item.get(field)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    # 多字段求和
    total = 0.0
    has_value = False
    for f in fields:
        v = item.get(f)
        if v is not None:
            try:
                total += float(v)
                has_value = True
            except (TypeError, ValueError):
                pass
    return total if has_value else None


def raw_subtract_fields(data: list, field_a: str, field_b: str, result_field: str = "difference") -> list:
    """对数据列表中每条记录计算 field_a - field_b，结果追加到每条记录中。

    field_a / field_b 支持逗号分隔多字段字符串或列表，此时先对各字段求和再做减法。
    """
    result = []
    for item in data:
        new_item = dict(item) if isinstance(item, dict) else item
        if isinstance(item, dict):
            a = _resolve_multi_field_value(item, field_a)
            b = _resolve_multi_field_value(item, field_b)
            if a is not None and b is not None:
                new_item[result_field] = a - b
            else:
                new_item[result_field] = None
        result.append(new_item)
    return result


def raw_multiply_fields(data: list, field_a: str, field_b: str, result_field: str = "product") -> list:
    """对数据列表中每条记录计算 field_a × field_b，结果追加到每条记录中。

    field_a / field_b 支持逗号分隔多字段字符串或列表，此时先对各字段求和再做乘法。
    """
    result = []
    for item in data:
        new_item = dict(item) if isinstance(item, dict) else item
        if isinstance(item, dict):
            a = _resolve_multi_field_value(item, field_a)
            b = _resolve_multi_field_value(item, field_b)
            if a is not None and b is not None:
                new_item[result_field] = a * b
            else:
                new_item[result_field] = None
        result.append(new_item)
    return result


def raw_divide_fields(data: list, field_a: str, field_b: str, result_field: str = "quotient") -> list:
    """对数据列表中每条记录计算 field_a ÷ field_b，结果追加到每条记录中。

    field_a / field_b 支持逗号分隔多字段字符串或列表，此时先对各字段求和再做除法。
    """
    result = []
    for item in data:
        new_item = dict(item) if isinstance(item, dict) else item
        if isinstance(item, dict):
            a = _resolve_multi_field_value(item, field_a)
            b = _resolve_multi_field_value(item, field_b)
            if a is not None and b is not None:
                new_item[result_field] = a / b if b != 0 else None
            else:
                new_item[result_field] = None
        result.append(new_item)
    return result


def raw_mom_change(data: list, field: str, time_field: str = "businessTime",
                   date_granularity: str = "month", result_field: str = "mom_change_pct",
                   domain: str = "") -> list:
    """
    计算环比变化率（Month-over-Month 或同粒度环比），按时间字段排序后逐条计算。

    新增特性：
    - 支持 date_granularity 参数，取值与 group_by / yoy_change 相同（year/quarter/month/day）
    - 校验时间连续性：若相邻两条记录的时间差不是 1 个粒度单位，则上一条的环比标记为 None，
      并记录 _calc_error 为 "time_gap"
    - 当某条记录对应的数据量明显少于正常范围时，添加 _calc_warning 字段（如 "low_data_density"）
    - 结果中始终包含 _calc_error 或 _calc_warning 字段，便于下游感知数据质量

    Args:
        data: 包含各期聚合数据的列表，每项应包含 time_field 和 field
        field: 要计算变化率的数值字段名（支持逗号分隔多字段，求各行之和）
        time_field: 时间字段名（默认 "businessTime"）
        date_granularity: 时间粒度，'year', 'quarter', 'month', 'day'（默认 'month'）
        result_field: 环比结果字段名（默认 "mom_change_pct"）
        domain: 保留参数，未使用

    Returns:
        仅返回成功计算环比变化率的记录，无法计算的记录（首期、时间断档、基值为零等）会被过滤掉。
    """
    from src.utils.logging_setup import logger
    from datetime import datetime, timedelta

    if not data:
        return data

    # 字段兼容：单字段 / 多字段逗号分隔 / 字段列表
    is_multi = isinstance(field, list) or (isinstance(field, str) and ',' in field)

    # 排序
    sorted_data = sorted(data, key=lambda x: str(x.get(time_field, "")) if isinstance(x, dict) else "")

    # 定义粒度对应的 timedelta 或 季度递推函数
    if date_granularity == 'year':
        def add_one(t: datetime) -> datetime:
            return t.replace(year=t.year + 1)
    elif date_granularity == 'quarter':
        def add_one(t: datetime) -> datetime:
            new_month = t.month + 3
            new_year = t.year
            if new_month > 12:
                new_month -= 12
                new_year += 1
            return t.replace(year=new_year, month=new_month)
    elif date_granularity == 'month':
        def add_one(t: datetime) -> datetime:
            new_year = t.year
            new_month = t.month + 1
            if new_month > 12:
                new_month = 1
                new_year += 1
            last_day = monthrange(new_year, new_month)[1]
            safe_day = min(t.day, last_day)
            return datetime(new_year, new_month, safe_day)
    elif date_granularity == 'day':
        def add_one(t: datetime) -> datetime:
            return t + timedelta(days=1)
    else:
        logger.error(f"[raw_mom_change] 不支持的 date_granularity: {date_granularity}")
        raise ValueError(f"不支持的 date_granularity: {date_granularity}")

    # 解析时间字符串为 datetime 用于连续性检查
    def parse_time(val: str) -> datetime | None:
        if not val:
            return None
        val = val.strip()
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y-Q%q", "%Y"):
            try:
                if fmt == "%Y":
                    return datetime.strptime(val, "%Y")
                elif fmt == "%Y-Q%q":
                    # 季度格式 "2024-Q2" → 转换成该季度的第一天
                    parts = val.split('-')
                    if len(parts) == 2 and parts[1].startswith('Q'):
                        q = int(parts[1][1])
                        month = (q - 1) * 3 + 1
                        return datetime(int(parts[0]), month, 1)
                else:
                    return datetime.strptime(val[:len(fmt)], fmt)
            except (ValueError, IndexError):
                continue
        # 若仍失败，尝试截取前10位作为日期
        try:
            return datetime.strptime(val[:10], "%Y-%m-%d")
        except ValueError:
            return None

    result = []
    for i, item in enumerate(sorted_data):
        new_item = dict(item) if isinstance(item, dict) else item
        # 初始状态
        new_item.setdefault('_calc_error', None)
        new_item.setdefault('_calc_warning', None)

        if i == 0:
            # 第一期无环比
            new_item[result_field] = None
            new_item['_calc_error'] = 'first_period'
            continue
        else:
            # 获取本期和上期时间
            curr_time_str = str(item.get(time_field, "")) if isinstance(item, dict) else ""
            prev_time_str = str(sorted_data[i - 1].get(time_field, "")) if isinstance(sorted_data[i - 1], dict) else ""

            curr_dt = parse_time(curr_time_str)
            prev_dt = parse_time(prev_time_str)

            # 时间连续性检查
            if curr_dt and prev_dt:
                expected_prev_dt = add_one(prev_dt)
                if curr_dt != expected_prev_dt:
                    new_item[result_field] = None
                    new_item['_calc_error'] = 'time_gap'
                    continue
            else:
                # 无法解析时间，不阻断计算，但记录警告
                new_item['_calc_warning'] = 'unparseable_time'

            # 取值（统一转为 Decimal 保证精度）
            if is_multi:
                curr_raw = _resolve_multi_field_value(item, field) if isinstance(item, dict) else None
                prev_raw = _resolve_multi_field_value(sorted_data[i - 1], field) if isinstance(sorted_data[i - 1], dict) else None
                curr_d = _to_decimal(curr_raw) if curr_raw is not None else None
                prev_d = _to_decimal(prev_raw) if prev_raw is not None else None
            else:
                curr = item.get(field) if isinstance(item, dict) else None
                prev = sorted_data[i - 1].get(field) if isinstance(sorted_data[i - 1], dict) else None
                curr_d = _to_decimal(curr) if curr is not None else None
                prev_d = _to_decimal(prev) if prev is not None else None

            if curr_d is not None and prev_d is not None:
                if not prev_d.is_zero():
                    new_item[result_field] = _round_half_up((curr_d - prev_d) / abs(prev_d) * 100, 2)
                    new_item['mom_base_time'] = prev_time_str
                    new_item['mom_base_value'] = prev_d
                else:
                    new_item[result_field] = None
                    new_item['_calc_error'] = 'base_zero'
            else:
                new_item[result_field] = None
                new_item['_calc_error'] = 'missing_value'

        # 数据完整性警告：如果本期的记录数少于其他期平均的 30%，则标记
        # 可根据业务需求调整阈值
        if 'count' in new_item:
            counts = [x.get('count', 0) for x in sorted_data if isinstance(x, dict) and x.get('count') is not None]
            if counts:
                avg_count = sum(counts) / len(counts)
                if new_item['count'] < 0.3 * avg_count:
                    new_item['_calc_warning'] = 'low_data_density' if not new_item.get('_calc_warning') else new_item['_calc_warning']

        if new_item.get('_calc_error') is None:
            result.append(new_item)

    if not result:
        logger.warning("[raw_mom_change] 所有记录均无法计算环比")

    return result


def raw_yoy_change(data: list, field: str, time_field: str = "businessTime",
                           date_granularity: str = "month", result_field: str = "yoy_change_pct",
                           domain: str = "") -> list:
    """
    基于时间字段自动进行同比计算。

    算法：
        1. 按 time_field 中的年份将数据拆分为 current（最近一年）和 previous（前一年）。
        2. 根据 date_granularity 提取时间标识（如 "2025-06" 或 "2025"）。
        3. 对 current 中的每一项，找 previous 中相同时间标识的记录，计算增长率。

    Args:
        data: 包含多期聚合数据的列表，每项必须包含 time_field 和 field。
        field: 要计算增长率的数值字段名（如 "sum_fields_sum"）。
        time_field: 时间字段名，值形如 "2025-06" (月粒度) 或 "2025" (年粒度)。
        date_granularity: 时间粒度，'month' 或 'year'。
        result_field: 结果字段名。
        domain: 保留参数，未使用。

    Returns:
        仅返回成功计算同比增长率的当期数据记录。
        无法计算的记录（缺少基期、值为空等）会被过滤掉，不包含在返回结果中。
    """
    from src.utils.logging_setup import logger

    # ---------- 1. 防御性检查 ----------
    if not field:
        logger.error("[raw_yoy_change_by_time] field 参数为空，无法计算")
        raise ValueError("field 参数不能为空")
    if not data:
        logger.warning("[raw_yoy_change_by_time] 输入 data 为空")
        return []

    # ---------- 2. 按年份拆分数据 ----------
    year_map = {}
    for item in data:
        time_val = item.get(time_field)
        if not time_val:
            continue
        # 提取年份（假设 time_field 值格式如 "2025-06" 或 "2025"）
        year = str(time_val)[:4]  # 前4位是年份
        year_map.setdefault(year, []).append(item)

    years = sorted(year_map.keys())
    if len(years) < 2:
        logger.warning(f"[raw_yoy_change_by_time] 数据只包含一个年份，无法计算同比。年份={years}")
        return []

    # 取最近两年作为当前期和基期
    current_year = years[-1]
    previous_year = years[-2]
    current_data = year_map[current_year]
    previous_data = year_map[previous_year]
    logger.info(f"[raw_yoy_change_by_time] 当前年份={current_year}, 记录数={len(current_data)}; "
                f"基期年份={previous_year}, 记录数={len(previous_data)}")

    # ---------- 3. 构建基期时间索引 ----------
    def _normalize_time_key(time_str: str, granularity: str) -> str:
        s = str(time_str).strip()
        if granularity == 'day':
            return s[:10]  # YYYY-MM-DD
        elif granularity == 'month':
            return s[:7]  # YYYY-MM
        elif granularity == 'year':
            return s[:4]  # YYYY
        elif granularity == 'quarter':
            # 输入可能是 "2024-Q2" 或 "2024-Q2..."，提取前7位
            if len(s) >= 7 and s[4] == '-' and s[5] == 'Q':
                return s[:7]
            # 如果格式不对，尝试从 "2024-04-01" 等日期字符串手动转换为季度
            from datetime import datetime
            try:
                dt = datetime.strptime(s[:10], "%Y-%m-%d")
                quarter = (dt.month - 1) // 3 + 1
                return f"{dt.year}-Q{quarter}"
            except ValueError:
                pass
            return s[:7]  # 降级
        else:
            return s[:7]  # 默认按月

    prev_index = {}
    for p in previous_data:
        tv = p.get(time_field)
        if not tv:
            continue
        key = _normalize_time_key(tv, date_granularity)
        prev_index[key] = p

    # ---------- 4. 匹配并计算 ----------
    result = []
    for item in current_data:
        new_item = dict(item)
        curr_time = item.get(time_field)
        if not curr_time:
            new_item[result_field] = None
            new_item['_calc_error'] = 'missing_time_field'
            result.append(new_item)
            continue

        # 构造基期对应的时间字符串
        prev_time = None
        if date_granularity == 'month':
            # 提取月份部分（假设格式 "2025-06"）
            parts = str(curr_time).split('-')
            if len(parts) >= 2:
                month_part = parts[1]  # 如 "06"
                prev_time = f"{previous_year}-{month_part}"
            else:
                # 无法解析，记录错误
                new_item[result_field] = None
                new_item['_calc_error'] = 'invalid_time_format'
                result.append(new_item)
                continue
        elif date_granularity == 'day':
            # 提取月-日部分（格式 "YYYY-MM-DD" → "MM-DD"），与基期年份拼接
            parts = str(curr_time).split('-')
            if len(parts) >= 3:
                month_day = parts[1] + '-' + parts[2]
                prev_time = f"{previous_year}-{month_day}"
            else:
                new_item[result_field] = None
                new_item['_calc_error'] = 'invalid_time_format'
                result.append(new_item)
                continue
        elif date_granularity == 'quarter':
            # curr_time 形如 "2025-Q1"
            curr_str = str(curr_time).strip()
            if len(curr_str) >= 7 and curr_str[4] == '-' and curr_str[5] == 'Q':
                quarter_part = curr_str[5:7]  # "Q1"
                prev_time = f"{previous_year}-{quarter_part}"
            else:
                # 降级：尝试从完整日期转换
                from datetime import datetime
                try:
                    dt = datetime.strptime(curr_str[:10], "%Y-%m-%d")
                    quarter = (dt.month - 1) // 3 + 1
                    prev_time = f"{previous_year}-Q{quarter}"
                except ValueError:
                    new_item[result_field] = None
                    new_item['_calc_error'] = 'invalid_time_format'
                    result.append(new_item)
                    continue
        else:  # year
            prev_time = previous_year  # 直接上一年

        # 查找基期记录
        prev_item = prev_index.get(prev_time)
        if prev_item is None:
            logger.debug(f"[raw_yoy_change_by_time] 未找到基期数据: {prev_time}")
            new_item[result_field] = None
            new_item['_calc_error'] = 'no_base_period'
        else:
            curr_val = item.get(field)
            prev_val = prev_item.get(field)
            if curr_val is not None and prev_val is not None:
                d_curr = _to_decimal(curr_val)
                d_prev = _to_decimal(prev_val)
                if d_curr is not None and d_prev is not None:
                    if not d_prev.is_zero():
                        new_item[result_field] = _round_half_up((d_curr - d_prev) / abs(d_prev) * 100, 2)
                        new_item['yoy_base_time'] = prev_time
                        new_item['yoy_base_value'] = d_prev
                    else:
                        new_item[result_field] = None
                        new_item['_calc_error'] = 'base_zero'
                else:
                    new_item[result_field] = None
                    new_item['_calc_error'] = 'value_error'
            else:
                new_item[result_field] = None
                new_item['_calc_error'] = 'missing_value'

        if new_item.get('_calc_error') is None:
            result.append(new_item)

    if not result:
        logger.warning("[raw_yoy_change_by_time] 所有当前期数据均无法计算同比")

    return result


def _strip_time_point_fields(item: dict) -> dict:
    """去掉记录中的时点字段（vHHMM），只保留元数据（节点名、日期等）。"""
    return {k: v for k, v in item.items() if not re.match(r"^v\d{4}$", k)}


def raw_max_field(data: list, field: str) -> dict:
    """找出数据列表中指定字段的最大值及所在记录。

    field 支持逗号分隔多字段字符串或列表，依据字段取最大值。
    多字段时返回 best_fields/best_moments/best_records。
    若有并列最大值，best_fields/best_moments/best_records 记录全部。
    record 只保留元数据字段（节点名称、日期等），不含时点明细。
    """
    is_multi = isinstance(field, list) or (isinstance(field, str) and ',' in field)
    best_val = None
    best_record = None
    best_records = []
    best_fields_list = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if is_multi:
            if isinstance(field, str):
                fields = [f.strip() for f in field.split(',')]
            else:
                fields = field
        else:
            fields = [field]
        current_values = []  # (value, field_name) 元组列表
        for f in fields:
            val = item.get(f)
            if val is None:
                continue
            try:
                current_values.append((float(val), f))
            except (TypeError, ValueError):
                continue
        if not current_values:
            continue
        max_val = max(v for v, _ in current_values)
        for fv, fv_field in current_values:
            if fv != max_val:
                continue
            if best_val is None or fv > best_val:
                best_val = fv
                best_record = _strip_time_point_fields(item)
                best_records = [best_record]
                best_fields_list = [fv_field]
            elif fv == best_val:
                best_records.append(_strip_time_point_fields(item))
                best_fields_list.append(fv_field)
    if best_val is None:
        return {"result": None, "field": field, "record": None, "error": f"字段 {field} 无有效数值"}
    result = {"result": best_val, "field": field, "record": best_record}
    if best_fields_list:
        result["best_fields"] = best_fields_list
        result["best_moments"] = [
            f"{bf[1:3]}:{bf[3:5]}" if re.match(r"^v\d{4}$", bf) else bf
            for bf in best_fields_list
        ]
        result["best_records"] = best_records
    return result



def raw_min_field(data: list, field: str) -> dict:
    """找出数据列表中指定字段的最小值及所在记录。

    field 支持逗号分隔多字段字符串或列表，依据字段取最小值。
    排除值为 0 的记录（0 视为无效数据）。
    多字段时返回 best_fields/best_moments/best_records。
    若有并列最小值，best_fields/best_moments/best_records 记录全部。
    record 只保留元数据字段（节点名称、日期等），不含时点明细。
    """
    is_multi = isinstance(field, list) or (isinstance(field, str) and ',' in field)
    best_val = None
    best_record = None
    best_records = []
    best_fields_list = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if is_multi:
            if isinstance(field, str):
                fields = [f.strip() for f in field.split(',')]
            else:
                fields = field
        else:
            fields = [field]
        current_values = []  # (value, field_name) 元组列表
        for f in fields:
            val = item.get(f)
            if val is None:
                continue
            try:
                fv = float(val)
                if fv != 0:
                    current_values.append((fv, f))
            except (TypeError, ValueError):
                continue
        if not current_values:
            continue
        min_val = min(v for v, _ in current_values)
        for fv, fv_field in current_values:
            if fv != min_val:
                continue
            if best_val is None or fv < best_val:
                best_val = fv
                best_record = _strip_time_point_fields(item)
                best_records = [best_record]
                best_fields_list = [fv_field]
            elif fv == best_val:
                best_records.append(_strip_time_point_fields(item))
                best_fields_list.append(fv_field)
    if best_val is None:
        return {"result": None, "field": field, "record": None, "error": f"字段 {field} 无有效数值"}
    result = {"result": best_val, "field": field, "record": best_record}
    if best_fields_list:
        result["best_fields"] = best_fields_list
        result["best_moments"] = [
            f"{bf[1:3]}:{bf[3:5]}" if re.match(r"^v\d{4}$", bf) else bf
            for bf in best_fields_list
        ]
        result["best_records"] = best_records
    return result


def raw_count_field(data: list, field: str = "") -> dict:
    """统计数据条数（计数操作）。

    field 可以是具体字段名（统计该字段非空的记录数），
    也可以为空字符串（统计总记录数）。
    """
    if not data:
        return {"result": 0, "field": field, "count": 0}

    if field:
        # 统计指定字段非空的记录数
        count = sum(1 for item in data if isinstance(item, dict) and item.get(field) is not None)
    else:
        # 统计总记录数
        count = len(data)

    return {"result": count, "field": field, "count": count}


# ─────────────────────────────────────────────
# @tool 装饰版本（保留备用）
# ─────────────────────────────────────────────

@tool
def filter(data: list[dict], field: str, operator: str, value: Any) -> list[dict]:
    """按条件过滤数据。支持操作符：>, <, >=, <=, ==, !=, between, in"""
    return raw_filter(data, field, operator, value)


@tool
def top_n(data: list[dict], field: str, n: int, order: str = "desc") -> list[dict]:
    """取前N条记录（按指定字段排序）。order: desc/asc"""
    return raw_top_n(data, field, n, order)


@tool
def bottom_n(data: list[dict], field: str, n: int, order: str = "asc") -> list[dict]:
    """取后N条记录（按指定字段排序）。order: asc/desc"""
    return raw_bottom_n(data, field, n, order)


@tool
def group_by(data: list[dict], group_field: str, agg_field: str, agg_op: str = "sum") -> list[dict]:
    """按指定字段分组，对每组执行聚合操作。agg_op: sum/average/count/max/min"""
    return raw_group_by(data, group_field, agg_field, agg_op)


@tool
def sum_field(data: list[dict], field: str) -> dict:
    """对数据列表中指定字段求和。"""
    return raw_sum_field(data, field)


@tool
def average_field(data: list[dict], field: str) -> dict:
    """对数据列表中指定字段求简单平均值。"""
    return raw_average_field(data, field)


@tool
def weighted_average_field(data: list[dict], value_field: str, weight_field: str) -> dict:
    """对数据列表中指定字段按权重字段计算加权平均值。"""
    return raw_weighted_average_field(data, value_field, weight_field)


@tool
def subtract_fields(data: list[dict], field_a: str, field_b: str, result_field: str = "difference") -> list[dict]:
    """对数据列表中每条记录计算 field_a - field_b，结果追加到每条记录中。"""
    return raw_subtract_fields(data, field_a, field_b, result_field)


@tool
def multiply_fields(data: list[dict], field_a: str, field_b: str, result_field: str = "product") -> list[dict]:
    """对数据列表中每条记录计算 field_a × field_b，结果追加到每条记录中。"""
    return raw_multiply_fields(data, field_a, field_b, result_field)


@tool
def divide_fields(data: list[dict], field_a: str, field_b: str, result_field: str = "quotient") -> list[dict]:
    """对数据列表中每条记录计算 field_a ÷ field_b，结果追加到每条记录中。"""
    return raw_divide_fields(data, field_a, field_b, result_field)


@tool
def mom_change(data: list[dict], field: str, time_field: str, result_field: str = "mom_change_pct") -> list[dict]:
    """计算环比变化率（Month-over-Month）。"""
    return raw_mom_change(data, field, time_field, result_field)


@tool
def yoy_change(data: list[dict], field: str, time_field: str = "businessTime",
               date_granularity: str = "month", result_field: str = "yoy_change_pct",
               domain: str = "") -> list[dict]:
    """计算同比变化率（Year-over-Year）。

    基于时间字段自动配对，无需手动传入前后期数据及 key_field。

    Args:
        data: 包含多期聚合数据的完整列表（通常来自 group_by 结果）。
        field: 要计算增长率的数值字段名（如 'sum_fields_sum'）。
        time_field: 时间字段名（默认为 'businessTime'）。
        date_granularity: 时间粒度，'month' 或 'year'（默认 'month'）。
        result_field: 结果字段名（默认 'yoy_change_pct'）。
        domain: 保留参数。
    """
    return raw_yoy_change(data, field, time_field, date_granularity, result_field, domain)


@tool
def max_field(data: list[dict], field: str) -> dict:
    """找出数据列表中指定字段的最大值及所在记录。"""
    return raw_max_field(data, field)


@tool
def min_field(data: list[dict], field: str) -> dict:
    """找出数据列表中指定字段的最小值及所在记录。"""
    return raw_min_field(data, field)


@tool
def count_field(data: list[dict], field: str = "") -> dict:
    """统计数据条数（计数操作）。field 为空时统计总记录数，否则统计该字段非空的记录数。"""
    return raw_count_field(data, field)


# 导出所有 @tool 工具供未来大模型工具调用场景使用
AGGREGATION_TOOLS = [
    filter,
    top_n,
    bottom_n,
    group_by,
    sum_field,
    average_field,
    weighted_average_field,
    subtract_fields,
    multiply_fields,
    divide_fields,
    mom_change,
    yoy_change,
    max_field,
    min_field,
    count_field,
]
