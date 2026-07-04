# src/workflow/question_split_workflow.py

import json
import re
from typing import List, Tuple, Dict, Any, Optional
from src.utils.logging_setup import logger
from langchain_core.language_models import BaseChatModel


class QuestionSplitWorkflow:
    MAX_SUB_ITEMS = 8

    def __init__(self, model: BaseChatModel):
        self.model = model

    async def split(self, user_query: str, intent_result: Dict[str, Any]) -> List[Tuple[str, Optional[str]]]:
        """拆分问题，返回子问题及对应的意图列表

        Returns:
            列表，每个元素为 (子问题, 意图或None)
        """
        fallback_intent: Optional[str] = None
        try:
            # 获取意图列表
            intents = intent_result.get("intents", [])
            if not intents:
                old_intent = intent_result.get("intent")
                if old_intent and old_intent != "待澄清":
                    intents = [old_intent]

            fallback_intent = intents[0] if intents else None
            if not intents:
                return [(user_query, None)]

            prompt = self._build_split_prompt_with_intent(
                user_query, intents, intent_result.get("time_range")
            )
            response = await self.model.ainvoke([{"role": "user", "content": prompt}])
            content = self._extract_response_text(response)

            sub_items = self._parse_split_result_with_intent(content, intents)
            sub_items = self._dedupe_sub_items(sub_items)
            # 暂不限制子问题数量，后续根据联调结果再决定是否恢复上限截断。
            # if len(sub_items) > self.MAX_SUB_ITEMS:
            #     logger.warning(
            #         f"[问题拆分] 子问题数量过多({len(sub_items)}), 截断为前{self.MAX_SUB_ITEMS}条"
            #     )
            #     sub_items = sub_items[:self.MAX_SUB_ITEMS]

            if sub_items and len(sub_items) > 1:
                logger.info(f"[问题拆分] 拆分为 {len(sub_items)} 个子问题: {sub_items}")
                return sub_items
            else:
                # 未拆分或拆分失败，返回原始问题，意图取第一个
                return [(user_query, fallback_intent)]
        except Exception as e:
            logger.error(f"[问题拆分] 拆分失败: {e}", exc_info=True)
            return [(user_query, fallback_intent)]

    def _build_split_prompt_with_intent(self, user_query: str, intents: List[str], time_range: Any) -> str:
        intent_list_str = ", ".join(intents)
        with open("data/env_variables/data_standard.json", "r", encoding="utf-8") as f:
            data_standard = json.load(f)
        return f"""你是一个问题拆分专家。用户的问题可能包含一个或多个独立查询，每个查询对应以下意图之一：{intent_list_str}。

请将用户问题拆分为若干独立的子问题，并为每个子问题指定准确的意图（必须从上述意图列表中选择）。如果无法明确判断意图，可将 intent 字段设为 null。

【拆分判断原则】

1. **单业务多实体拆分**
   若问题属于同一业务接口，但同时涉及多个不同实体（如"发电企业A和发电企业B""冀北和蒙东"），应拆分为多个子问题（每个实体一个子问题）。

2. **多业务不同实体拆分**
   若问题同时涉及不同业务对象或不同业务接口（如"企业信息"和"合同信息"），应按业务维度拆分为多个子问题；若每个业务内仍包含多个实体，再在该业务子问题内继续按实体拆分。

3. **单一实体同意图多指标不拆分 / 跨意图多指标必须拆分**
   若问题仅针对单一实体，查询的是多个指标或属性，且这些指标**都属于同一意图**（例如"A线路的功率和电压"都属于同一输电查询意图），默认不拆分，由同一子问题保留完整约束后交给单工作流处理。
   若多个指标分属不同意图（如"电量"属电量查询意图、"电价"属电价查询意图），则必须按意图拆分为多个子问题，每个子问题仅保留与自己意图相关的指标词，删除其他意图的指标词。例如：
   - "查询2025年3月12日18点时段购方节点省间日内现货出清电量与电价的对应数据"（intents: 省间日内现货电量查询, 省间日内现货电价查询）
     → ① {{"question": "查询2025年3月12日18点时段购方节点省间日内现货出清电量数据", "intent": "省间日内现货电量查询"}}
     → ② {{"question": "查询2025年3月12日18点时段购方节点省间日内现货出清电价数据", "intent": "省间日内现货电价查询"}}
     
3.5. **同实体多极值统计拆分**
   若问题只针对单一实体，但明确要求分别统计同一指标的最高值和最低值（即同时出现“最高”和“最低”、“最大”和“最小”、“峰值”和“谷值”等对立极值词），属于逻辑上必须分别执行的两次聚合，应拆分为两个子问题：
   - 子问题1：去掉“最低/最小”相关表述，保留“最高/最大”及时间、实体等约束；
   - 子问题2：去掉“最高/最大”相关表述，保留“最低/最小”及时间、实体等约束。
   示例：
   “查询内蒙古西部售方2025年9月1日至9月20日省间日前现货出清电量明细，并统计最高分时电量和最低分时电量”
   → ① “查询内蒙古西部售方2025年9月1日至9月20日省间日前现货出清电量明细，并统计最高分时电量”
   → ② “查询内蒙古西部售方2025年9月1日至9月20日省间日前现货出清电量明细，并统计最低分时电量”

4. **多时间点同一指标不拆分（特殊规则）**
   仅当问题同时满足以下全部条件时，视为一个整体查询，不进行拆分：
   - 只涉及单一实体（例如"A线路"）；
   - 只查询该实体的**同一个指标**（例如"功率"）；
   - 仅时间点在变化，且为并列/连续的时间点（例如"t1、t2、t3时刻的功率"）。

   **特别注意——同比/环比/同期对比不拆分**：当问题属于"单实体 + 单指标 + 不同时间段"的模式，且包含"同比""环比""同期""变化""趋势"等时间对比语义时，即使问句中出现"与""和""比较""对比"等词，也**必须视为一个整体查询，不进行拆分**。因为此类问题的两个时间段共享同一实体和同一指标，仅时间维度不同。
   示例：
   - "比较2026年2月与2025年2月冀北售方的日前现货平均出清电价同比变化" → 不拆分，返回单元素数组，保留原问题全部约束
   - "查询冀北2026年2月与2025年2月省间日前现货出清平均电价的同比变化，购售方为售方" → 不拆分，返回单元素数组

5. **多日期分别列出拆分**
   若问题要求"分别列出/查询"多个不同日期下满足条件的实体集合（如"分别列出X日和Y日参与交易的省份""X日、Y日分别有哪些机组"等），且各日期查询相互独立（不存在跨日期计算如平均/合计/差值/排序），则应**按日期拆分为多个子问题**，每个子问题只保留单个日期及该日期的全部原始约束（实体、指标、筛选条件等）。

   示例：
   - "分别列出参与了25年7月10日和7月20日的日前现货交易的网省有哪些"
     → ① "参与了25年7月10日日前现货交易的网省有哪些"
     → ② "参与了25年7月20日日前现货交易的网省有哪些"

   注意：本规则仅适用于问"哪些实体满足条件"的枚举型问题。若问题是查询某个确定实体在多个日期的指标取值（如"冀北7月10日和7月20日的日前现货出清电价"），则不按此规则拆分，仍走规则4。

6. **多步依赖拆分 + 约束传播**
   若问题由多个步骤组成（常见连接词:"并""然后""再""接着""同时"），且后续步骤通过指示代词（如"该""此""其""上述""前述""对应"等）引用了前面步骤的结果，则应拆分为多个子问题。
   **约束传播规则**: 原始问题末尾的全局约束条件（如"（购售方为购方）""（交易类型为现货）"等括号标注，或者"其中购售方为购方"等尾部限定）若同时适用于所有子问题，则必须将其**追加到每个子问题的末尾**。
   例子:
   - "查询10月总电量最高的省份及其对应日期，并查询该日期前三天全国总电量（购售方为购方）" → 拆为 2 个子问题:
     ① "查询10月总电量最高的省份及其对应日期（购售方为购方）"
     ② "查询该日期前三天全国总电量（购售方为购方）"

7. **同类跨类别对比拆分**
   若问题属于同一业务接口（单意图），但要求对**同一分组维度下的两个对立/互补类别**进行比较，且系统单次查询只能针对一个类别，则必须拆分为两个子问题分别查询。

   **识别条件（需同时满足）**：
   a) 问题中出现了两个对立或互补的角色/类别，且它们在数据中由同一个分类字段的不同取值表示。常见对立类别对：
      - 交易/电量场景：购方 ↔ 售方、买方 ↔ 卖方、购电 ↔ 售电
      - 输电/联络线场景：送方 ↔ 受方、输入 ↔ 输出、送端 ↔ 受端
      - 调节/调度场景：上调 ↔ 下调、增发 ↔ 减发
      - 发电/用电场景：发电 ↔ 用电、出力 ↔ 负荷
   b) 两个类别作用于**同一实体维度**，要求按该实体维度分组后再做跨类别比较（如"同一省份的购方与售方""同一联络线的送方与受方""同一区域的发电与用电"）。
   c) 存在比较语义：大于、小于、高于、低于、超过、不足、对比、差异、差值、差额、哪个多/少、倒挂、悬殊、更大/更小、最多/最少（指跨类别差值最大）等。

   **不适用此规则的情况**：
   - 若问题已明确指定了某一类别（如"购方电量最高的省份"），不存在跨类别比较，则不拆分。
   - 若两个类别不是对立互补关系，而是普通的多个实体列举（如"北京和上海"），应走规则1（多实体拆分）。

   **拆分方式**：生成两个子问题，分别去掉原始问题中的跨类别比较条件，改为只查询单一类别在该实体维度下的汇总。原始问题中的时间范围等全局约束须完整保留。

   **拆分示例**：
   - "购方总电量大于售方总电量的省份有哪些" → ① "各省份购方总电量" ② "各省份售方总电量"
   - "送方电量高于受方电量的联络线有哪些" → ① "各联络线送方电量" ② "各联络线受方电量"
   - "同一省份购售双方电量差额最大的省份" → ① "各省份购方电量" ② "各省份售方电量"
   - "哪些直流的输入功率小于输出功率" → ① "各直流输入功率" ② "各直流输出功率"

8. **默认不拆分（单意图）**
   除上述明确列出的拆分场景外（多意图、单意图多主体对比、多步依赖、同类跨类别对比等），其余场景默认不拆分，返回单元素数组。

【原始问题】
{user_query}

【时间范围】
{time_range}

【输出格式】
必须严格输出一个 JSON 数组，每个元素为一个对象，包含 `question` 和 `intent` 字段。例如：
```json
[
  {{"question": "2025年1月的发电企业信息", "intent": "发电企业查询"}},
  {{"question": "2025年1月的合同信息", "intent": "合同信息查询"}}
]
```

【必须遵守的标准词库】
name_abbreviation：{data_standard.get('name_abbreviation', [])}
device_name：{data_standard.get('device_name', [])}
plant_name：{data_standard.get('plant_name', [])}
outage_type：{data_standard.get('outage_type', [])}
"sysName": {data_standard.get("sysName", [])}
"sendrecv": {data_standard.get("sendrecv", [])}


【强制要求】
1. 每个子问题必须语义完整、独立，包含所有必要的限定条件（如时间、实体名称等）。
2. 若原始问题为单一意图，且不属于"多主体/多对象对比""多时间点同一指标""多步依赖""同类跨类别对比"场景，则返回单元素数组。
3. 严格遵守上述拆分判断原则，尤其注意特殊规则的严格适用条件。
4. 若无法确定是否应拆分，默认返回单元素数组（保留原问题与原意图）。
5. 严格原样保留原始问题中每个主体/对象的名称（如"冀北""蒙东"等），禁止改写、扩写、补全、追加后缀、语义归一化。
6. 若按意图拆分，子问题必须根据其 intent 进行语义聚焦——仅保留与该意图直接相关的指标关键词（如电量查询子问题只提"电量"、电价查询子问题只提"电价"），删除其他意图的指标词。时间、实体等公共约束原样保留。禁止在子问题中保留"对应""以及""与"等跨意图关联词。
7. **只输出 JSON 数组，不要包含任何解释、说明或额外文本。**
"""

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        content = response.content if hasattr(response, "content") else response
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "\n".join(parts)
        return str(content)

    @staticmethod
    def _extract_first_json_array_fragment(text: str) -> Optional[str]:
        if not isinstance(text, str):
            return None
        start = text.find("[")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escape = False
        for idx, ch in enumerate(text[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_string = False
                continue

            if ch == "\"":
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start:idx + 1]
        return None

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()

    @staticmethod
    def _dedupe_sub_items(sub_items: List[Tuple[str, Optional[str]]]) -> List[Tuple[str, Optional[str]]]:
        result: List[Tuple[str, Optional[str]]] = []
        seen = set()
        for q, intent in sub_items:
            key = (re.sub(r"\s+", "", q), intent)
            if key in seen:
                continue
            seen.add(key)
            result.append((q, intent))
        return result

    def _parse_split_result_with_intent(self, content: str, valid_intents: List[str]) -> List[
        Tuple[str, Optional[str]]]:
        try:
            content = self._strip_code_fence(content)
            data = json.loads(content)
            if not isinstance(data, list):
                return []
            result: List[Tuple[str, Optional[str]]] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                q = item.get("question")
                if not q or not isinstance(q, str):
                    continue
                intent = item.get("intent")
                if intent is not None and intent not in valid_intents:
                    # 意图不在列表中，置为 None（后续快速匹配兜底）
                    intent = None
                result.append((q, intent))
            return result
        except json.JSONDecodeError:
            fallback_content = self._extract_first_json_array_fragment(content)
            if fallback_content:
                try:
                    data = json.loads(fallback_content)
                    if isinstance(data, list):
                        result: List[Tuple[str, Optional[str]]] = []
                        for item in data:
                            if not isinstance(item, dict):
                                continue
                            q = item.get("question")
                            if not q or not isinstance(q, str):
                                continue
                            intent = item.get("intent")
                            if intent is not None and intent not in valid_intents:
                                intent = None
                            result.append((q, intent))
                        return result
                except json.JSONDecodeError:
                    pass
            logger.warning(f"[问题拆分] JSON解析失败: {content}")
            return []
