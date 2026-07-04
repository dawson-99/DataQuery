from typing import Dict, Any, List, Optional
import json
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

from src.utils.echarts_utils import sort_time_fields_batch



with open("data/env_variables/time_point_mapping.json", "r", encoding="utf-8") as f:
    time_point_mapping = json.load(f)

keys = time_point_mapping.keys()

def handle_summary_input(data: List[Dict[str, Any]] | Dict[str, Any]):
    records = sort_time_fields_batch(data)
    if isinstance(records, list):
        totalKeys = 0
        for record in records:
            if isinstance(record, dict):
                totalKeys += len(record.keys() & keys)

        if totalKeys >= 96:
            needAvg = len(records) > 5
            time_point_data = _build_condensed_data(records,needAvg=needAvg)
            # handle_data, all_keys = process_dicts(records)
            return {
                "_record_count": len(records),
                "numerical_value":time_point_data,
                # "other_key_fields":handle_data
            }

    return records


def _build_condensed_data(data: List[Dict[str, Any]], needAvg: bool = True) -> Dict[str, Any]:
    # 收集每个时间点的 (值, 记录索引) 列表
    records: dict[str, list[tuple[Decimal, int]]] = {}
    for idx, record in enumerate(data):
        for k, v in record.items():
            if k in keys:
                num = _to_decimal(v)
                if num is not None:
                    records.setdefault(k, []).append((num, idx))

    max_data: Dict[str, Any] = {}
    min_data: Dict[str, Any] = {}
    avg_data: Dict[str, Any] = {}
    max_source_map: Dict[str, list[int]] = {}
    min_source_map: Dict[str, list[int]] = {}

    for k, v_list in records.items():
        valid = [(val, idx) for val, idx in v_list if not val.is_zero()]
        if not valid:
            max_data[k] = 0
            min_data[k] = 0
            avg_data[k] = 0
            continue

        max_val = max(v[0] for v in valid)
        min_val = min(v[0] for v in valid)
        max_data[k] = _decimal_to_number(max_val)
        min_data[k] = _decimal_to_number(min_val)
        max_source_map[k] = [idx for val, idx in valid if val == max_val]
        min_source_map[k] = [idx for val, idx in valid if val == min_val]

        non_zero_vals = [val for val, _ in valid]
        avg_value = sum(non_zero_vals) / Decimal(len(non_zero_vals))
        avg_data[k] = _decimal_to_number(avg_value, keep_scale=2)

    def _get_identity(record_idx: int) -> dict:
        """提取记录中非时间点的身份字段（日期、网省、节点名称等）。"""
        record = data[record_idx]
        return {k: v for k, v in record.items() if k not in keys and not k.startswith('_')}

    # 最大点、最小点
    max_point = min_max_dict(max_data, is_max=True, source_map=max_source_map, get_identity=_get_identity)
    min_point = min_max_dict(min_data, is_max=False, source_map=min_source_map, get_identity=_get_identity)
    avg = {"avg_nonzero": avg_data} if needAvg else {}

    result = max_point | min_point | avg
    return result

def _vcode_to_minutes(v: str) -> int:
    """vHHMM → 从 0 点开始的分钟数。"""
    return int(v[1:3]) * 60 + int(v[3:5])


def _vcode_to_hhmm(v: str) -> str:
    """vHHMM → HH:MM。"""
    m = _vcode_to_minutes(v)
    return f"{m // 60:02d}:{m % 60:02d}"


def _split_consecutive_groups(vcodes: list[str]) -> list[list[str]]:
    """按 15 分钟连续性拆分 v 编码列表，每子组内时刻连续。"""
    if not vcodes:
        return []
    sorted_codes = sorted(vcodes, key=_vcode_to_minutes)
    groups = []
    cur = [sorted_codes[0]]
    for i in range(1, len(sorted_codes)):
        if _vcode_to_minutes(sorted_codes[i]) - _vcode_to_minutes(sorted_codes[i - 1]) == 15:
            cur.append(sorted_codes[i])
        else:
            groups.append(cur)
            cur = [sorted_codes[i]]
    groups.append(cur)
    return groups


def min_max_dict(
    data: dict,
    is_max: bool = True,
    source_map: dict | None = None,
    get_identity=None,
):
    if not data:
        return {}

    tag = "max_point" if is_max else "min_point"
    non_zero_items = {k: v for k, v in data.items() if v is not None and v != 0}
    if not non_zero_items:
        return {tag: None}

    value = max(non_zero_items.values()) if is_max else min(non_zero_items.values())
    keys = [k for k, v in data.items() if v == value]

    # 顶层 time：逗号分隔的 HH:MM 汇总
    time_str = ",".join(_vcode_to_hhmm(k) for k in keys)
    point_data: dict = {"time": time_str, "value": value}

    if source_map and get_identity:
        # 按来源分组时刻：相同来源集合的时刻归为一组
        time_source_groups: dict[tuple, list[str]] = {}
        for k in keys:
            if k in source_map:
                indices = tuple(sorted(source_map[k]))
                time_source_groups.setdefault(indices, []).append(k)

        if time_source_groups:
            time_sources = []
            for indices, time_keys in time_source_groups.items():
                # 按连续性拆分为子组，每个子组独立输出
                for subgroup in _split_consecutive_groups(time_keys):
                    hhmm_list = [_vcode_to_hhmm(v) for v in subgroup]
                    if len(hhmm_list) == 1:
                        time_repr = hhmm_list[0]
                    else:
                        time_repr = f"{hhmm_list[0]}-{hhmm_list[-1]}"
                    entry: dict = {
                        "time": time_repr,
                        "sources": [get_identity(idx) for idx in indices],
                    }
                    time_sources.append(entry)
            point_data["time_sources"] = time_sources

    return {tag: point_data}

def _to_decimal(value: Any) -> Optional[Decimal]:
    if value is None or isinstance(value, bool):
        return None
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


def _decimal_to_number(value: Decimal, keep_scale: Optional[int] = None) -> Any:
    if keep_scale is not None:
        quantize_unit = Decimal('1').scaleb(-keep_scale)
        value = value.quantize(quantize_unit, rounding=ROUND_HALF_UP)
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return int(normalized)
    return float(normalized)