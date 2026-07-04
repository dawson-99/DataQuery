"""
通用聚合计算 Agent

直接在 Python 侧调用聚合工具函数执行计算，无需大模型介入。
速度快、结果准确，不受上下文长度限制。
"""
import re
from typing import Any

from src.utils.aggregation_tools import (
    raw_filter,
    raw_top_n,
    raw_bottom_n,
    raw_group_by,
    raw_sum_field,
    raw_row_sum_fields,
    raw_average_field,
    raw_weighted_average_field,
    raw_subtract_fields,
    raw_multiply_fields,
    raw_divide_fields,
    raw_mom_change,
    raw_yoy_change,
    raw_max_field,
    raw_min_field,
    raw_count_field, raw_count_if,
)
from src.utils.logging_setup import logger


# 汇总型操作：结果是单个聚合值，放入 summary
_SUMMARY_OPS = {'sum', 'average', 'weighted_average', 'max', 'min', 'count', 'count_if'}
# 过滤/排序/分组操作：作用于整个数据集，返回处理后的列表
_FILTER_SORT_GROUP_OPS = {'filter', 'top_n', 'bottom_n', 'group_by'}


class AggregationAgent:
    """通用聚合计算 Agent

    直接在 Python 侧根据 operation 类型 dispatch 到对应函数执行计算，
    不依赖大模型，速度快且结果准确，不受上下文长度限制。
    """

    @staticmethod
    def _resolve_date_granularity(user_query: str, group_field: str, date_granularity: str) -> str:
        """按用户问题语义修正日期分组粒度。

        兜底规则：当 group_field 为日期字段且模型默认给了 month，但问题明显是“按日”，
        则自动修正为 day，避免出现“3月每天”被按“3月整月”聚合的问题。
        """
        if group_field not in ('beginDate', 'endDate', 'businessTime', 'rq'):
            return date_granularity

        # 若上游已明确给出非默认粒度（如 week_in_month/week/day/year/quarter），
        # 不再被关键词兜底二次改写，避免业务口径被覆盖。
        explicit = (date_granularity or "").strip().lower()
        if explicit in ("week_in_month", "week", "day", "year", "quarter"):
            return explicit

        query = (user_query or "").replace(" ", "")
        q_lower = query.lower()

        day_cues = ('各日', '每日', '每天', '逐日', '按日', '每一天', '每天的', '每天8点', '各天', '哪天', '某天', '按天', '日统计', '当日', '的日期', '日期最高', '日期最低', '哪一日')
        month_cues = ('各月', '每月', '逐月', '按月')
        week_cues = ('各周', '每周', '逐周', '按周', '周度')
        year_cues = ('各年', '每年', '逐年', '按年')

        if any(cue in query for cue in day_cues):
            return 'day'
        if any(cue in query for cue in week_cues):
            return 'week'
        if any(cue in query for cue in month_cues):
            return 'month'
        if any(cue in query for cue in year_cues):
            return 'year'

        # 英文兜底关键词
        if any(cue in q_lower for cue in ('daily', 'perday', 'byday')):
            return 'day'
        if any(cue in q_lower for cue in ('weekly', 'perweek', 'byweek')):
            return 'week'
        if any(cue in q_lower for cue in ('monthly', 'permonth', 'bymonth')):
            return 'month'
        if any(cue in q_lower for cue in ('yearly', 'peryear', 'byyear')):
            return 'year'

        return date_granularity

    @staticmethod
    def _get_groupby_output_fields(op: dict) -> list:
        """返回 group_by 操作在输出数据中产生的字段名列表。

        与 raw_group_by 内部的命名规则保持一致：
        - 单字段聚合：{agg_op}_{agg_field}
        - 多字段聚合：{agg_op}_fields_sum
        - field_max / field_min：{agg_op}_value
        - 始终包含 count 字段
        """
        agg_op = op.get('agg_op', '')
        agg_field = op.get('agg_field', '')
        group_field = op.get('group_field', '')
        fields = []
        if group_field:
            fields.append(group_field)
        if agg_op in ('field_max', 'field_min'):
            fields.append(f'{agg_op}_value')
        else:
            if isinstance(agg_field, list):
                is_multi = len(agg_field) > 1
            elif isinstance(agg_field, str) and ',' in agg_field:
                is_multi = True
            else:
                is_multi = False
            if is_multi:
                fields.append(f'{agg_op}_fields_sum')
            elif agg_field:
                fields.append(f'{agg_op}_{agg_field}')
        fields.append('count')
        return fields

    @staticmethod
    def _resolve_field_from_registry(op_field: str, sample: dict, registry: list) -> str | None:
        """在数据 sample 中解析操作引用的字段名。

        依次尝试：
        1) 前缀匹配：在已有字段名前加 agg 前缀
        2) 注册表匹配：与 group_by 注册的输出字段按最长公共前缀匹配（≥4 字符）
        """
        # 1) 前缀匹配：兼容 v0800 → sum_v0800 等场景
        for prefix in ('average_', 'sum_', 'max_', 'min_', 'count_'):
            candidate = f'{prefix}{op_field}'
            if candidate in sample:
                return candidate
        # 2) 注册表匹配：处理 LLM 把字段名后缀写错的情况（如 average_fields_average → average_fields_sum）
        if registry:
            best_match = None
            best_prefix_len = 0
            for reg_field in registry:
                common = 0
                for a, b in zip(op_field, reg_field):
                    if a == b:
                        common += 1
                    else:
                        break
                if common > best_prefix_len and common >= 4:
                    best_prefix_len = common
                    best_match = reg_field
            if best_match is not None:
                return best_match
        return None

    def _execute_one(self, op: dict, data: Any, user_query: str = "", domain: str = "") -> tuple[str, str, Any]:
        """执行单个操作，返回 (operation, field, result)
        
        Args:
            op: 操作配置
            data: 数据
            domain: 业务域（transmission/transaction），用于模糊匹配
        """
        operation = op.get('operation', '')
        field = op.get('field', '')

        # 打印操作前的输入数据
        import json
        data_count = len(data) if isinstance(data, list) else (1 if data else 0)
        logger.info(f"[AggregationAgent] 执行操作前 - operation={operation}, field={field}, domain={domain}, 输入数据条数={data_count}")


        # 如果数据量不大，打印前几条样本数据
        if isinstance(data, list) and data:
            sample_size = min(2, len(data))
            sample_data = data[:sample_size]
            try:
                logger.info(f"[AggregationAgent] 输入数据样本（前{sample_size}条）: {json.dumps(sample_data, ensure_ascii=False, default=str)}")
            except Exception as e:
                logger.info(f"[AggregationAgent] 输入数据样本（无法序列化）: {str(sample_data)}")

        try:
            # === 新增：处理抽象计数操作 count_records ===
            if operation == 'count_records':
                # 统计记录总数，忽略 field 参数（或使用 op 中的 result_field）
                result_field = op.get('result_field', 'record_count')
                # 调用 raw_count_field，field 传 None 表示统计所有记录
                result = raw_count_field(data=data, field=None)
                if isinstance(result, dict) and 'result' in result:
                    # 如果返回的是 {"result": 12} 格式，重命名结果字段
                    result['result_field'] = result_field
                else:
                    # 兼容处理：如果 raw_count_field 直接返回数值
                    result = {'result': result, 'result_field': result_field}
                logger.info(f"[AggregationAgent] count_records 完成，记录总数: {result.get('result')}")
                return operation, None, result

            if operation == 'filter':
                result = raw_filter(
                    data=data,
                    field=field,
                    operator=op.get('operator', ''),
                    value=op.get('value'),
                    domain=domain,
                )



            elif operation == 'count_if':
                fields = op.get('fields', field)
                if isinstance(fields, str):
                    fields = [f.strip() for f in fields.split(',') if f.strip()]
                condition = op.get('condition', {})
                cond_operator = condition.get('operator', '>=')
                cond_value = condition.get('value')
                result = raw_count_if(data=data, fields=fields, operator=cond_operator, value=cond_value)
                field = ','.join(fields)  # ← 关键修复
                logger.info(f"[AggregationAgent] count_if({fields}) 完成, 满足条件个数={result.get('result')}")


            elif operation == 'top_n':
                result = raw_top_n(
                    data=data,
                    field=field,
                    n=op.get('n', 10),
                    order=op.get('order', 'desc'),
                )

            elif operation == 'bottom_n':
                result = raw_bottom_n(
                    data=data,
                    field=field,
                    n=op.get('n', 10),
                    order=op.get('order', 'asc'),
                )

            elif operation == 'group_by':
                # group_by 操作使用 group_field 而非 field 作为分组字段
                group_field = op.get('group_field', field)
                date_granularity = self._resolve_date_granularity(
                    user_query=user_query,
                    group_field=group_field,
                    date_granularity=op.get('date_granularity', 'month'),
                )
                result = raw_group_by(
                    data=data,
                    group_field=group_field,
                    agg_field=op.get('agg_field', ''),
                    agg_op=op.get('agg_op', ''),
                    # date_granularity 由意图识别阶段根据用户问题确定：
                    # "year"（按年）、"month"（按月，默认）、"day"（按日）
                    date_granularity=date_granularity,
                    domain=domain,
                )

            elif operation == 'sum':
                result = raw_sum_field(data=data, field=field)

            elif operation == 'row_sum':
                result = raw_row_sum_fields(
                    data=data,
                    field=field,
                    result_field=op.get('result_field', 'sum_fields_sum'),
                )

            elif operation == 'average':
                result = raw_average_field(data=data, field=field)

            elif operation == 'weighted_average':
                result = raw_weighted_average_field(
                    data=data,
                    value_field=field,
                    weight_field=op.get('weight_field', ''),
                )

            elif operation == 'subtract':
                field_a_val = op.get('field', field)
                field_b_val = op.get('field_b', '')
                # 兼容模型将两个字段放在同一个 field 里的写法，如 field='v0800,v0815' 且无 field_b
                # 此时语义是 v0800 - v0815，需拆分为 field_a=v0800, field_b=v0815
                if isinstance(field_a_val, str) and ',' in field_a_val and not field_b_val:
                    parts = [f.strip() for f in field_a_val.split(',') if f.strip()]
                    field_a_val, field_b_val = parts[0], parts[1] if len(parts) > 1 else ''
                    logger.info(f"[AggregationAgent] subtract 自动拆分字段: field_a={field_a_val}, field_b={field_b_val}")
                elif isinstance(field_a_val, str) and ',' in field_a_val:
                    logger.info(f"[AggregationAgent] subtract field_a 含多字段: {field_a_val}，将按行求和")
                if isinstance(field_b_val, str) and ',' in field_b_val:
                    logger.info(f"[AggregationAgent] subtract field_b 含多字段: {field_b_val}，将按行求和")
                result = raw_subtract_fields(
                    data=data,
                    field_a=field_a_val,
                    field_b=field_b_val,
                    result_field=op.get('result_field', 'difference'),
                )

            elif operation == 'multiply':
                field_a_val = op.get('field', field)
                field_b_val = op.get('field_b', '')
                # 兼容模型将两个字段放在同一个 field 里的写法，如 field='v0800,v0815' 且无 field_b
                # 此时语义是 v0800 × v0815，需拆分为 field_a=v0800, field_b=v0815
                if isinstance(field_a_val, str) and ',' in field_a_val and not field_b_val:
                    parts = [f.strip() for f in field_a_val.split(',') if f.strip()]
                    field_a_val, field_b_val = parts[0], parts[1] if len(parts) > 1 else ''
                    logger.info(f"[AggregationAgent] multiply 自动拆分字段: field_a={field_a_val}, field_b={field_b_val}")
                elif isinstance(field_a_val, str) and ',' in field_a_val:
                    logger.info(f"[AggregationAgent] multiply field_a 含多字段: {field_a_val}，将按行求和")
                if isinstance(field_b_val, str) and ',' in field_b_val:
                    logger.info(f"[AggregationAgent] multiply field_b 含多字段: {field_b_val}，将按行求和")
                result = raw_multiply_fields(
                    data=data,
                    field_a=field_a_val,
                    field_b=field_b_val,
                    result_field=op.get('result_field', 'product'),
                )

            elif operation == 'divide':
                field_a_val = op.get('field', field)
                field_b_val = op.get('field_b', '')
                # 兼容模型将两个字段放在同一个 field 里的写法，如 field='v0800,v0815' 且无 field_b
                # 此时语义是 v0800 ÷ v0815，需拆分为 field_a=v0800, field_b=v0815
                if isinstance(field_a_val, str) and ',' in field_a_val and not field_b_val:
                    parts = [f.strip() for f in field_a_val.split(',') if f.strip()]
                    field_a_val, field_b_val = parts[0], parts[1] if len(parts) > 1 else ''
                    logger.info(f"[AggregationAgent] divide 自动拆分字段: field_a={field_a_val}, field_b={field_b_val}")
                elif isinstance(field_a_val, str) and ',' in field_a_val:
                    logger.info(f"[AggregationAgent] divide field_a 含多字段: {field_a_val}，将按行求和")
                if isinstance(field_b_val, str) and ',' in field_b_val:
                    logger.info(f"[AggregationAgent] divide field_b 含多字段: {field_b_val}，将按行求和")
                result = raw_divide_fields(
                    data=data,
                    field_a=field_a_val,
                    field_b=field_b_val,
                    result_field=op.get('result_field', 'quotient'),
                )


            elif operation == 'mom_change':
                result = raw_mom_change(
                    data=data,
                    field=field,
                    time_field=op.get('time_field', 'businessTime'),
                    date_granularity=op.get('date_granularity', 'month'),  # 新增
                    result_field=op.get('result_field', 'mom_change_pct'),
                )


            elif operation == 'yoy_change':
                result = raw_yoy_change(
                    data=data,
                    field=field,
                    time_field=op.get('time_field', 'businessTime'),
                    date_granularity=op.get('date_granularity', 'month'),
                    result_field=op.get('result_field', 'yoy_change_pct'),
                    domain=domain,
                )

            elif operation == 'max':
                result = raw_max_field(data=data, field=field)

            elif operation == 'min':
                result = raw_min_field(data=data, field=field)

            elif operation == 'count':
                result = raw_count_field(data=data, field=field)

            else:
                logger.warning(f"[AggregationAgent] 未知操作类型: {operation}")
                return operation, field, None

            logger.info(f"[AggregationAgent] {operation}({field}) 完成")
            return operation, field, result

        except Exception as e:
            logger.error(f"[AggregationAgent] {operation}({field}) 执行失败: {e}")
            return operation, field, None

    def _merge_sum_ops(self, operations: list) -> list:
        """将连续的多个单字段 sum 操作合并为一个多字段 sum 操作。
        
        如果后续存在 divide/subtract/multiply 操作引用了 sum_{field} 格式的字段名，
        说明每个 sum 结果需要独立保存，此时不合并，保持各自独立。
        """
        # 检查是否有操作引用了 sum_{field} 形式的字段名
        dependent_sum_fields = set()
        for op in operations:
            if op.get('operation') in ('divide', 'subtract', 'multiply'):
                for key in ('field', 'field_b'):
                    val = op.get(key, '')
                    if isinstance(val, str) and val.startswith('sum_'):
                        dependent_sum_fields.add(val[4:])  # 去掉 "sum_" 前缀，得到原始字段名

        merged = []
        i = 0
        while i < len(operations):
            op = operations[i]
            if op.get('operation') == 'sum':
                # 收集连续的 sum 操作
                sum_fields = []
                while i < len(operations) and operations[i].get('operation') == 'sum':
                    f = operations[i].get('field')
                    if isinstance(f, list):
                        sum_fields.extend(f)
                    elif f:
                        sum_fields.append(f)
                    i += 1
                # 如果任意一个字段被后续操作单独引用（sum_xxx），则不合并，拆分为独立操作
                if dependent_sum_fields and any(f in dependent_sum_fields for f in sum_fields):
                    for f in sum_fields:
                        merged.append({'operation': 'sum', 'field': f})
                elif len(sum_fields) == 1:
                    merged.append({'operation': 'sum', 'field': sum_fields[0]})
                else:
                    merged.append({'operation': 'sum', 'field': sum_fields})
            else:
                merged.append(op)
                i += 1
        return merged

    def _run(self, operations: list, data: Any, domain: str = "", user_query: str = "") -> dict:
        """执行所有操作并整理结果
        
        执行顺序：filter → group_by → top_n/bottom_n → aggregation（聚合）
        
        Args:
            operations: 操作列表
            data: 数据
            domain: 业务域（transmission/transaction），用于模糊匹配
            user_query: 用户原始问题（用于日期粒度语义修正）
        """
        summary = []
        aggregated_data = data  # 逐行型操作会不断更新此引用
        sum_results = {}  # 存储 sum 操作的结果，用于后续 divide 等操作
        # 保存 group_by 执行前的原始明细数据，用于「分组均值回填」场景
        pre_groupby_data = None
        last_groupby_op = None
        grouped_data_full = None

        # 合并连续的多个单字段 sum 为一个多字段 sum
        operations = self._merge_sum_ops(operations)

        _ROW_OPS = {'subtract', 'multiply', 'divide', 'mom_change', 'yoy_change'}
        # 存储所有汇总结果（用于动态阈值替换）：key 为 "avg_xxx"/"sum_xxx"/"max_xxx"/"min_xxx"
        dynamic_refs = {}
        # 存储 group_by 输出字段名列表，供后续 filter/max/min 等操作的字段名动态解析
        gb_field_registry: list = []

        for op in operations:
            op_type = op.get('operation', '')

            # 逐行型操作执行前，先将已有的 sum_results 广播到数据
            # 确保 divide(sum_dealUnitsNum, sum_bidUnitsNum) 等能在数据中找到对应字段
            if op_type in _ROW_OPS and sum_results and isinstance(aggregated_data, list):
                for item in aggregated_data:
                    if isinstance(item, dict):
                        item.update(sum_results)
                logger.info(f"[AggregationAgent] 逐行操作({op_type})前广播 sum 结果: {list(sum_results.keys())}")

            # filter 操作：
            # 1. 将 value 中的动态引用（如 "average_dealAvgPrice"）替换为实际计算值
            # 2. 若紧接在 group_by 后，且 filter 的 field 是原始明细字段（不是聚合结果字段），
            #    说明是「分组均值回填」场景：把 group_by 的聚合结果按分组键回填到原始明细，
            #    再对原始明细执行 filter，实现"每条记录与其所在分组均值比较"的语义
            if op_type == 'filter':
                op = dict(op)  # 浅拷贝，避免修改原始 operations
                raw_value = op.get('value')
                if isinstance(raw_value, str) and raw_value in dynamic_refs:
                    resolved = dynamic_refs[raw_value]
                    logger.info(f"[AggregationAgent] filter 动态阈值替换: value='{raw_value}' → {resolved}")
                    op['value'] = resolved
                filter_field = op.get('field', '')
                raw_value_after = op.get('value')  # 可能已被上面替换
                # 判断是否为「分组均值回填」场景：
                # 条件：紧接 group_by 后、filter field 是原始字段、value 是指向 group_by 聚合结果的字符串引用
                if (
                    pre_groupby_data is not None
                    and last_groupby_op is not None
                    and isinstance(raw_value_after, str)  # value 尚未被替换（不在 dynamic_refs 中）
                    and filter_field
                ):
                    gb_group_field = last_groupby_op.get('group_field', '')
                    gb_agg_op = last_groupby_op.get('agg_op', 'sum')
                    gb_agg_field = last_groupby_op.get('agg_field', '')
                    gb_result_field = f"{gb_agg_op}_{gb_agg_field}"  # 如 average_dealAvgPrice
                    gb_date_granularity = last_groupby_op.get('date_granularity', 'month')
                    # value 引用的字段名与 group_by 结果字段名一致，才触发回填
                    if raw_value_after == gb_result_field:
                        # 构建 group_key → 聚合值 的索引
                        group_val_index = {}
                        for grp_item in aggregated_data:
                            if isinstance(grp_item, dict):
                                gk = grp_item.get(gb_group_field)
                                gv = grp_item.get(gb_result_field)
                                if gk is not None and gv is not None:
                                    group_val_index[str(gk)] = gv
                        # 将 group_by 前的原始明细中，每条记录追加所属分组的聚合值
                        # 同时复用 raw_group_by 里的粒度截取逻辑确定分组键
                        from src.utils.aggregation_tools import raw_group_by as _rgb  # noqa
                        _DATE_LEN = {'year': 4, 'month': 7, 'day': 10}
                        def _get_quarter_key(s):
                            try:
                                year = s[:4]; month = int(s[5:7]) if len(s) >= 7 else 1
                                return f"{year}-Q{(month-1)//3+1}"
                            except Exception:
                                return s[:7]
                        enriched = []
                        skipped_cross_period = 0
                        for orig_item in pre_groupby_data:
                            if not isinstance(orig_item, dict):
                                enriched.append(orig_item)
                                continue
                            gk_raw = orig_item.get(gb_group_field)
                            if gk_raw is None:
                                continue
                            gk_str = str(gk_raw)
                            if gb_date_granularity == 'quarter':
                                gk_str = _get_quarter_key(gk_str)
                            else:
                                cut = _DATE_LEN.get(gb_date_granularity, 7)
                                gk_str = gk_str[:cut] if len(gk_str) >= cut else gk_str
                            # ── 跨期检查：endDate 存在时，截取后必须与 beginDate 的分组键一致
                            # 与 raw_group_by 内部的跨期过滤逻辑保持一致，确保回填阶段同样排除跨期记录
                            if gb_group_field in ('beginDate', 'endDate', 'businessTime'):
                                end_raw = orig_item.get('endDate')
                                if end_raw is not None:
                                    end_str = str(end_raw)
                                    if gb_date_granularity == 'quarter':
                                        end_key = _get_quarter_key(end_str)
                                    else:
                                        cut = _DATE_LEN.get(gb_date_granularity, 7)
                                        end_key = end_str[:cut] if len(end_str) >= cut else end_str
                                    if end_key != gk_str:
                                        skipped_cross_period += 1
                                        logger.debug(
                                            f"[AggregationAgent] 回填阶段跨期记录跳过({gb_date_granularity}): "
                                            f"beginDate={gk_raw}({gk_str}), endDate={end_raw}({end_key})"
                                        )
                                        continue
                            agg_val = group_val_index.get(gk_str)
                            if agg_val is None:
                                # 该记录的分组键在 group_by 结果中不存在（可能被跨期过滤掉），跳过
                                continue
                            new_item = dict(orig_item)
                            new_item[gb_result_field] = agg_val
                            enriched.append(new_item)
                        if skipped_cross_period > 0:
                            logger.info(
                                f"[AggregationAgent] 回填阶段跨期过滤: 跳过 {skipped_cross_period} 条跨期记录，"
                                f"剩余 {len(enriched)} 条同期记录参与后续 filter"
                            )
                        aggregated_data = enriched
                        # filter 的 value 替换为 group_by 聚合结果字段名（逐行比较）
                        op['value'] = gb_result_field  # 此时 value 改为字段名
                        # 修改 operator 为字段间比较，需要 raw_filter 支持字段引用
                        # 这里用特殊 value 格式 "field:xxx" 告知 raw_filter 做字段间比较
                        op['value'] = f"field:{gb_result_field}"
                        logger.info(
                            f"[AggregationAgent] 分组均值回填完成: group_field={gb_group_field}, "
                            f"agg_field={gb_result_field}, 回填后数据条数={len(enriched)}, "
                            f"filter 改为字段间比较: {filter_field} {op.get('operator')} {gb_result_field}"
                        )
                elif filter_field and isinstance(aggregated_data, list) and aggregated_data:
                    sample = aggregated_data[0]
                    if isinstance(sample, dict) and filter_field not in sample:
                        resolved = self._resolve_field_from_registry(filter_field, sample, gb_field_registry)
                        if resolved:
                            logger.info(
                                f"[AggregationAgent] filter 字段动态解析: '{filter_field}' → '{resolved}'"
                            )
                            op['field'] = resolved

            # max/min/top_n/bottom_n 操作：若 field 在当前数据中不存在，
            # 通过注册表动态解析到 group_by 实际输出的字段名
            if op_type in ('max', 'min', 'top_n', 'bottom_n') and last_groupby_op is not None:
                op_field = op.get('field', '')
                if op_field and isinstance(aggregated_data, list) and aggregated_data:
                    sample = aggregated_data[0]
                    if isinstance(sample, dict) and op_field not in sample:
                        resolved = self._resolve_field_from_registry(op_field, sample, gb_field_registry)
                        if resolved:
                            op = dict(op)
                            op['field'] = resolved
                            logger.info(
                                f"[AggregationAgent] {op_type} 字段动态解析: '{op_field}' → '{resolved}'"
                            )

            # ---- 操作执行前的通用字段名动态解析 ----
            # 覆盖所有带 field/field_b/value_field/weight_field 参数的操作
            # （mom_change / yoy_change / subtract / multiply / divide 等），
            # 若字段在当前数据中不存在，从 group_by 注册表中按公共前缀匹配
            if isinstance(aggregated_data, list) and aggregated_data:
                sample = aggregated_data[0]
                if isinstance(sample, dict):
                    for key in ('field', 'field_b', 'value_field', 'weight_field'):
                        raw_val = op.get(key, '')
                        if isinstance(raw_val, str) and raw_val and raw_val not in sample:
                            resolved = self._resolve_field_from_registry(raw_val, sample, gb_field_registry)
                            if resolved:
                                op = dict(op)
                                op[key] = resolved
                                logger.info(
                                    f"[AggregationAgent] {op_type} {key} 动态解析: '{raw_val}' → '{resolved}'"
                                )

            operation, field, result = self._execute_one(op, aggregated_data, user_query=user_query, domain=domain)

            if result is None:
                continue

            if operation in _FILTER_SORT_GROUP_OPS:
                # 过滤/排序/分组操作：result 是处理后的完整列表
                if isinstance(result, list):
                    if operation == 'group_by':
                        # 保存 group_by 前的原始明细，以及本次 group_by 的操作配置
                        # 用于后续 filter 的「分组均值回填」场景
                        pre_groupby_data = aggregated_data
                        last_groupby_op = op
                        grouped_data_full = result
                        # 注册 group_by 实际输出的字段名，供后续操作做字段动态解析
                        gb_field_registry = self._get_groupby_output_fields(op)
                    aggregated_data = result
                    logger.info(f"[AggregationAgent] {operation} 后数据条数: {len(aggregated_data)}")

            elif operation in _SUMMARY_OPS:
                # 汇总型：result 是含 result 键的 dict
                agg_result_val = result.get('result') if isinstance(result, dict) else result
                summary_item = {
                    'operation': operation,
                    'field': field,
                    'result': agg_result_val,
                    'count': result.get('count') if isinstance(result, dict) else None,
                }
                # max/min 操作：传递极值对应的字段名、时刻标签和所在记录
                if operation in ('max', 'min') and isinstance(result, dict):
                    if result.get('best_fields'):
                        summary_item['best_fields'] = result['best_fields']
                    if result.get('best_moments'):
                        summary_item['best_moments'] = result['best_moments']
                    if result.get('best_times'):
                        summary_item['best_times'] = result['best_times']
                    if result.get('record'):
                        summary_item['record'] = result['record']
                summary.append(summary_item)

                if operation == 'count_if' and isinstance(result, dict):
                    for key in ('matched_fields', 'matched_values', 'businessTime', 'nameAbbreviation'):
                        if key in result:
                            summary_item[key] = result[key]
                    # 传递元数据（非时间点字段，如 businessTime、nameAbbreviation 等）
                    for key, value in result.items():
                        if key not in summary_item and not re.match(r'^v\d{4}$', key):
                            summary_item[key] = value

                # 将汇总结果注册到 dynamic_refs，供后续 filter 的动态阈值引用
                # 支持的引用格式："avg_xxx" / "sum_xxx" / "max_xxx" / "min_xxx" / "count_xxx"
                if isinstance(field, str) and agg_result_val is not None:
                    ref_key = f"{operation}_{field}"
                    dynamic_refs[ref_key] = agg_result_val
                    logger.info(f"[AggregationAgent] 注册动态引用: {ref_key}={agg_result_val}")

                # 如果是 sum 操作，保存结果供后续逐行操作使用
                if operation == 'sum':
                    if isinstance(field, list):
                        result_field_name = 'sum_fields_sum'
                    else:
                        result_field_name = f'sum_{field}'
                    sum_results[result_field_name] = agg_result_val
                    logger.info(f"[AggregationAgent] 保存 sum 结果: {result_field_name}={agg_result_val}")
            else:
                # 逐行型：result 是追加了新列的完整列表
                if isinstance(result, list):
                    aggregated_data = result

        return {
            'aggregated_data': aggregated_data,
            'summary': summary,
            'grouped_data_full': grouped_data_full,
        }

    def invoke(
        self,
        user_query: str,
        operations: list,
        data: Any,
        context: dict | None = None,
        domain: str = "",
    ) -> dict:
        """同步执行操作（纯 Python，无大模型调用）

        Args:
            user_query: 用户原始问题（保留接口兼容，暂未使用）
            operations: 操作列表（filter/top_n/bottom_n/sum/average/...）
            data: 待处理的数据（list 或 dict）
            context: 上下文信息（保留接口兼容，暂未使用）
            domain: 业务域（transmission/transaction），用于模糊匹配

        Returns:
            {"aggregated_data": ..., "summary": [...]}
        """
        logger.info(f"[AggregationAgent] 开始执行操作, ops={[op.get('operation') for op in operations]}, domain={domain}")
        return self._run(operations, data, domain, user_query)

    async def ainvoke(
        self,
        user_query: str,
        operations: list,
        data: Any,
        context: dict | None = None,
        domain: str = "",
    ) -> dict:
        """异步执行操作（纯 Python，无大模型调用）

        Args:
            user_query: 用户原始问题（保留接口兼容，暂未使用）
            operations: 操作列表（filter/top_n/bottom_n/sum/average/...）
            data: 待处理的数据（list 或 dict）
            context: 上下文信息（保留接口兼容，暂未使用）
            domain: 业务域（transmission/transaction），用于模糊匹配

        Returns:
            {"aggregated_data": ..., "summary": [...]}
        """
        logger.info(f"[AggregationAgent] 开始执行操作（异步）, ops={[op.get('operation') for op in operations]}, domain={domain}")
        # 纯 CPU 计算，直接同步执行即可
        return self._run(operations, data, domain, user_query)
