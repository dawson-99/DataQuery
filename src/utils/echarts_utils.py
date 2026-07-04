import copy
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, List, Optional


SERIES_NAME_PRIORITY = {
    "desc": 0,
    "nameAbbreviation": 1,
    "targetType": 2,
    "name": 3,
    "voltage_name": 4,
}

DEFAULT_COLOR_PALETTE = [
    "#5470c6",
    "#fac858",
    "#ee6666",
    "#3ba272",
    "#ff9f7f",
    "#9a60b4",
    "#c0e0ff",
    "#ffdb5c",
]

DEFAULT_UNITS = {
    "电量": "MWh",
    "电价": "元/MWh",
}


def sort_time_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    对单条记录中的时间字段（vHHMM）按时间升序排序。
    非时间字段保留原顺序并放在前面。
    """
    if not isinstance(record, dict):
        return record

    time_fields: Dict[str, Any] = {}
    other_fields: Dict[str, Any] = {}

    for key, value in record.items():
        if is_time_field(key):
            time_fields[key] = value
        else:
            other_fields[key] = value

    result: Dict[str, Any] = {}
    result.update(other_fields)
    result.update(dict(sorted(time_fields.items())))
    return result



def sort_time_fields_batch(records: List[Dict[str, Any]] | Dict[str, Any]):
    """
    对批量记录执行时间字段排序；若是 merge 结构，先展开后再排序。
    """
    if isinstance(records, dict):
        return sort_time_fields(records)
    if not isinstance(records, list):
        return records
    if not records:
        return records

    normalized_records = _normalize_merged_records(records) if judge_merge(records) else records
    return [sort_time_fields(record) for record in normalized_records]



def judge_merge(records: List[Any]) -> bool:
    for record in records:
        if isinstance(record, dict) and "sub_question" in record and "data" in record:
            return True
    return False



def is_dense_time_series_records(records: Any) -> bool:
    if not isinstance(records, list) or len(records) <= 1:
        return False
    return len(collect_time_fields(records)) >= 96



def _get_unit_by_intent(intent: str):
    if not intent:
        return ""
    for key in DEFAULT_UNITS.keys():
        if key in intent:
            return DEFAULT_UNITS.get(key)

    return ""

def build_dense_time_series_echarts(records: List[Dict[str, Any]], intent: str = "") -> Optional[Dict[str, Any]]:
    """
    为密集时间点宽表数据直接生成 ECharts 配置。

    规则：
    - 时间点字段个数 < 96：返回 None，由上层继续走模型流程。
    - 只有 1 条记录：返回 None，由上层决定不生成图。
    - 2~5 条记录：每条记录生成一条折线。
    - >5 条记录：直接聚合为最大值、最小值、平均值三条折线。
    """
    if not isinstance(records, list) or len(records) <= 1:
        return None

    time_fields = collect_time_fields(records)
    if len(time_fields) < 96:
        return None

    x_axis_data = [format_time_label(field) for field in time_fields]
    series = (
        build_aggregated_series(records, time_fields)
        if len(records) > 5
        else build_record_series(records, time_fields)
    )

    if not has_non_zero_series(series):
        return None

    unit = _get_unit_by_intent(intent)

    option = {
        "color": DEFAULT_COLOR_PALETTE,
        "tooltip": {
            "formatter": build_string_tooltip_formatter(unit)
        },
        "legend": {
            "data": [item.get("name", "系列") for item in series],
            "left": "center",
            "bottom": "5%",
        },
        "grid": {
            "left": "3%",
            "right": "4%",
            "top": "8%",
            "bottom": "20%",
            "containLabel": False,
        },
        "xAxis": {
            "type": "category",
            "data": x_axis_data,
            "axisLabel": {
                "interval": 4,
            },
        },
        "yAxis": {
            "type": "value",
            "name": normalize_unit(unit),
        },
        "series": series,
    }
    return add_global_extremes_to_echarts(option)

def build_string_tooltip_formatter(unit: str) -> str:
    normalized_unit = normalize_unit(unit)
    return "{a}<br/>{b}: {c}" if not normalized_unit else f"{{a}}<br/>{{b}}: {{c}} {normalized_unit}"

def normalize_unit(unit: Any) -> str:
    if unit is None:
        return ""
    return str(unit).strip()


def has_non_zero_series(series_list: List[Dict[str, Any]]) -> bool:
    for series in series_list:
        if not isinstance(series, dict):
            continue
        data = series.get("data", [])
        if not isinstance(data, list):
            continue
        for value in data:
            number = to_decimal(value)
            if number is not None and not number.is_zero():
                return True
    return False



def add_global_extremes_to_echarts(option: Dict[str, Any]) -> Dict[str, Any]:
    """
    为 ECharts 配置添加整体最高点和最低点的标记与水平线。
    """
    new_option = copy.deepcopy(option)
    series_list = new_option.get("series", [])
    if not isinstance(series_list, list) or not series_list:
        return new_option

    x_axis = new_option.get("xAxis", {})
    if isinstance(x_axis, list):
        x_axis = x_axis[0] if x_axis else {}
    x_data = x_axis.get("data", [])
    if not x_data:
        return new_option

    global_max = -float("inf")
    global_min = float("inf")
    max_series_idx = max_data_idx = None
    min_series_idx = min_data_idx = None

    for series_idx, series in enumerate(series_list):
        data = series.get("data", [])
        if not isinstance(data, list):
            continue
        for data_idx, value in enumerate(data):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if numeric_value > global_max:
                global_max = numeric_value
                max_series_idx = series_idx
                max_data_idx = data_idx
            if global_min > numeric_value > 0:
                global_min = numeric_value
                min_series_idx = series_idx
                min_data_idx = data_idx

    if global_max == -float("inf") or global_min == float("inf"):
        return new_option

    if max_series_idx is not None:
        add_series_mark(
            series_list[max_series_idx], global_max, max_data_idx, "最高", "#ff4d4f", is_max=True
        )
    if min_series_idx is not None:
        add_series_mark(
            series_list[min_series_idx], global_min, min_data_idx, "最低", "#52c41a", is_max=False
        )

    return new_option



def collect_time_fields(records: List[Dict[str, Any]]) -> List[str]:
    fields = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        for key in record.keys():
            if is_time_field(key):
                fields.add(key)
    return sorted(fields)



def build_record_series(records: List[Dict[str, Any]], time_fields: List[str]) -> List[Dict[str, Any]]:
    series_name_fields = get_varying_non_time_keys(records)
    series_list: List[Dict[str, Any]] = []

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        series_list.append({
            "name": build_record_series_name(record, series_name_fields, idx),
            "type": "line",
            "data": [normalize_number(record.get(field, 0)) for field in time_fields],
        })

    return series_list



def build_aggregated_series(records: List[Dict[str, Any]], time_fields: List[str]) -> List[Dict[str, Any]]:
    max_data: List[Any] = []
    min_data: List[Any] = []
    avg_data: List[Any] = []

    for field in time_fields:
        values = [
            value
            for value in (to_decimal(record.get(field)) for record in records if isinstance(record, dict))
            if value is not None and not value.is_zero()
        ]

        if not values:
            max_data.append(0)
            min_data.append(0)
            avg_data.append(0)
            continue

        max_data.append(decimal_to_number(max(values)))
        min_data.append(decimal_to_number(min(values)))
        avg_data.append(decimal_to_number(sum(values) / Decimal(len(values)), keep_scale=2))

    return [
        {"name": "最大值", "type": "line", "data": max_data},
        {"name": "最小值", "type": "line", "data": min_data},
        {"name": "平均值", "type": "line", "data": avg_data},
    ]



def get_varying_non_time_keys(records: List[Dict[str, Any]]) -> List[str]:
    if not records or len(records) == 1:
        return []

    candidate_keys = [key for key in records[0].keys() if not is_time_field(key)]
    varying_keys = []

    for key in candidate_keys:
        values = [record.get(key) for record in records if isinstance(record, dict)]
        if len(set(values)) > 1:
            varying_keys.append(key)

    return sorted(varying_keys, key=lambda key: (0, SERIES_NAME_PRIORITY[key]) if key in SERIES_NAME_PRIORITY else (1, key))



def build_record_series_name(record: Dict[str, Any], name_fields: List[str], idx: int) -> str:
    parts: List[str] = []
    for field in name_fields:
        value = record.get(field)
        if value in (None, ""):
            continue
        text = str(value)
        if text not in parts:
            parts.append(text)
    return "-".join(parts) if parts else f"系列{idx + 1}"



def add_series_mark(series: Dict[str, Any], value: float, data_idx: int,
                    label: str, color: str, is_max: bool) -> None:
    mark_point = series.get("markPoint", {})
    point_data = mark_point.setdefault("data", [])

    point_exists = False
    for item in point_data:
        item_value = None
        if "yAxis" in item:
            item_value = item.get("yAxis")
        elif "coord" in item and len(item["coord"]) > 1:
            item_value = item["coord"][1]
        try:
            if item_value is not None and float(item_value) == value:
                point_exists = True
                break
        except (TypeError, ValueError):
            continue

    if not point_exists:
        point_data.append({
            "coord": [data_idx, value],
            "name": label,
            "value": value,
            "label": {
                "show": False,
                "formatter": f"{value}",
                "position": "top" if is_max else "bottom",
                "offset": [0, -2] if is_max else [0, 2],
            },
            "itemStyle": {"color": color},
        })
    series["markPoint"] = mark_point

    mark_line = series.get("markLine", {})
    line_data = mark_line.setdefault("data", [])

    line_exists = False
    for item in line_data:
        try:
            if "yAxis" in item and float(item["yAxis"]) == value:
                line_exists = True
                break
        except (TypeError, ValueError):
            continue

    if not line_exists:
        line_data.append({
            "yAxis": value,
            "name": label,
            "lineStyle": {"color": color, "type": "dashed", "width": 1},
            "label": {"show": False, "formatter": f"{label}: {value}", "position": "end"},
            "symbol": "none",
        })
    series["markLine"] = mark_line



def format_time_label(field: str) -> str:
    return f"{field[1:3]}:{field[3:5]}"



def is_time_field(key: Any) -> bool:
    return isinstance(key, str) and key.startswith("v") and len(key) == 5 and key[1:].isdigit()



def to_decimal(value: Any) -> Optional[Decimal]:
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



def normalize_number(value: Any) -> Any:
    decimal_value = to_decimal(value)
    if decimal_value is None:
        return 0 if value is None else value
    return decimal_to_number(decimal_value)



def decimal_to_number(value: Decimal, keep_scale: Optional[int] = None) -> Any:
    if keep_scale is not None:
        quantize_unit = Decimal("1").scaleb(-keep_scale)
        value = value.quantize(quantize_unit, rounding=ROUND_HALF_UP)
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return int(normalized)
    return float(normalized)



def _normalize_merged_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_records: List[Dict[str, Any]] = []

    for record in records:
        if not isinstance(record, dict):
            continue
        sub_question = record.get("sub_question")
        data = record.get("data")
        if not sub_question:
            continue

        normalized = {"sub_question": sub_question}
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                operation = item.get("operation")
                result = item.get("result")
                if operation:
                    normalized[operation] = result
                else:
                    normalized.update(item)

        normalized_records.append(normalized)

    return normalized_records
