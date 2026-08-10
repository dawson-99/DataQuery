# 规则审查训练数据编写指南

本指南面向**手工编写规则种子数据集**（Qwen3-3B 级模型工具调用训练），说明三个核心问题：**怎么从问题定工具链、怎么定决策、怎么写句式才能撑起数据量**。

数据链路：`rule_review_seed_spec.json`（句式模板 + 槽位词表）→ `scripts/build_seed_dataset.py`（组合生成 1000 条完整种子）→ `scripts/validate_seed_dataset.py`（校验 + 覆盖统计）。

---

## 1. 工具链选择决策表（从问题定工具）

拿到一个 query，按顺序自问：**① 可计算/可定位/可核验吗 → ② 数据在哪 → ③ 有陷阱吗 → ④ 要几步 → ⑤ 都不需要？**

| 问题特征（prompt 线索） | 工具链 |
|---|---|
| 「价格表/上限表/取数」 | `extract_table_data` |
| 「是否超过/低于 X」 | `extract_table_data` + `arithmetic_compare` |
| 单位混用（kWh/MWh/千度） | 前置 `unit_converter` 再比较 |
| 「第X条/第X款原文」 | `locate_clause` |
| 「引用第X页/出处准确吗」 | `verify_citation` |
| 「正文规定的数值是多少」 | `extract_numeric_fact` |
| 「两规则是否冲突/一致」 | `locate_clause` ×2 + `detect_rule_conflict` |
| 「X日期时规则是否生效」 | `validate_date_applicability` |
| 「第X条参照第Y条，Y的内容」 | `resolve_cross_reference` + `locate_clause` |
| 模糊参数（「华北」不在表内） | 工具未命中 → **修正参数重试** |
| 纯主观/流程/解释/知识题 | **无工具**（负样本） |

**参数纪律**：args 只从 query 原文提取实体（地区/数值/日期/条款号），不自行编造。

## 2. 决策标注规则

| 问题类型 | expected_decision |
|---|---|
| 比较题（超没超上限） | 按数值 vs 基线重判 → 符合 / 不符合 |
| 取数题（上限是多少） | **无法判断**（工具返回事实，不构成判断） |
| 定位/核验/交叉引用题 | **无法判断** |
| 冲突题（数值冲突） | 冲突 → 不符合；一致 → 符合；语义待定 → 无法判断 |
| 负样本 | 无法判断 |

> **最常见的错误**：把取数题标成「符合」——模型会学到不调工具就下结论。只有比较题能给 符合/不符合。

## 3. 句式改写方法论（撑起数据量的第一步）

同一语义场景必须写成**多种问法**（本项目每个场景 10-15 个句式模板）。改写维度：

| 改写维度 | 示例（比较类基句：800元/MWh 是否超过上限） |
|---|---|
| 书面/口语 | 「是否超过」「有没有超过」「超没超」「高于吗」 |
| 结构变化 | 「价格为800元/MWh，请问合规吗」「800元/MWh这个价格，符合上限要求吗」 |
| 省略式 | 「冀北的价格上限是多少」「冀北上限多少」 |
| 前置式 | 「根据省间现货规则，冀北的出清电价上限是？」 |
| 时间位置 | 「2025年3月15日冀北的日前现货出清电价」vs「冀北2025年3月15日的日前现货出清电价」 |
| 单位混用 | 「800元/MWh」「0.8元/kWh」「0.8元每千瓦时」「500元/千度」 |
| 疑问词变化 | 「是否高于」「有没有超过」「超过了没有」「合规吗」「是高了还是低了」 |
| 取数/比较区分 | 取数类不给数值（「上限是多少」）；比较类必须给数值 |

**句式写作要点**：
- 一条句式 = 一个 `query_template`（含 `{槽位}` 占位符）+ 固定 `expected_tools` + `expected_workflow`（工具链提示）
- 句式设计时**定死工具链**——变体只换词不改链，质量不稀释
- 槽位只填实体词（地区/数值/日期/条款号/文档名），句式结构保持

## 4. 槽位词表设计（撑起数据量的第二步）

词表**全局定义**在 `slot_vocab`，与句式模板组合生成。原则：

- **实体多样**：region 20 个省区电力主体；price 覆盖 超限(>760)/合规(<760)/等于(760) 三档
- **单位混用**：同一数值多种单位写法（800元/MWh = 0.8元/kWh = 800元/千度），训练模型做单位归一化
- **数值带换算基准**：词表条目 `{"text": "0.8元/kWh", "mwh": 800}` 携带归一化后的数值，生成时**自动重判决策**
- **分组合词表**：`price_above` / `price_below` / `price_equal` / `unit_price` 按句式语义选用（「低于上限」句式只用 `price_below` 组）
- **边界与错误值**：`out_region`（表外地区）、`no_such_article`（不存在的条款）——训练「未命中 → 修正/反馈」路径

## 5. 种子规格文件结构（v2）

```json
{
  "version": 2,
  "facts": { "price_cap_mwh": 760, "min_quantity_kwh": 10000 },
  "slot_vocab": { "region": [...], "price": [{"text": "800元/MWh", "mwh": 800}, ...], ... },
  "templates": [
    {
      "template_id": "compare_001",
      "category": "compare",
      "query_template": "{date}{region}的日前现货出清电价{price}是否超过价格上限？",
      "slots": ["date", "region", "price"],
      "expected_tools": ["extract_table_data", "arithmetic_compare"],
      "expected_workflow": "extract_table_data({region}, 列=价格上限) → arithmetic_compare({price} gt 上限) → 结论",
      "decision_rule": "gt760_bad",
      "expected_decision": ""
    }
  ]
}
```

`decision_rule` 取值：
- `gt760_bad`：槽位 mwh > 760 → 不符合，否则符合（价格上限语义）
- `gte_10000_good`：kwh ≥ 10000 → 符合，否则不符合（交易量下限语义）
- `fixed`：直接用 `expected_decision`

**约束**：`query_template` 中的 `{占位符}` 必须与 `slots` 声明一致（校验脚本强制）；`expected_workflow` 中的工具名必须以 `工具名(...)` 形式出现（校验正则依赖）。

## 6. 场景矩阵与分布目标

| 类别 | 模板数 | 覆盖内容 |
|---|---|---|
| 单工具 × 9 | ~25 | 每工具 2-3 句式（正例 + 边界：表外地区/未命中条款） |
| 工具链 | ~20 | extract→compare、unit→compare、locate→conflict、cross_ref→locate、date 前置等 |
| 多文档 | ~5 | 跨文档定位 + 冲突/一致性判断 |
| 错误恢复 | ~5 | 模糊参数（「华北」）→ 修正参数重试 |
| 负样本 | ~25 | 纯规则判断、主观解释、流程题、诱惑场景——**占比 10-15%**（防工具滥用） |

**分布目标**（`validate_seed_dataset.py` 输出核对）：
- 9 工具全覆盖（每个工具 ≥ 30 条）
- 负样本占比 10-15%
- 1000 条 query 去重率 100%
- 单工具参数多样性：region ≥ 15、price ≥ 8

## 7. 生成与校验流程

```bash
# 1. 编辑 data/evaluation/rule_review_seed_spec.json（加句式/加词表）
# 2. 生成 1000 条完整种子（无占位符残留）
conda run -n dataquery python scripts/build_seed_dataset.py
# 3. 校验 + 覆盖统计
conda run -n dataquery python scripts/validate_seed_dataset.py
# 4. 单元测试
conda run -n dataquery python -m pytest tests/test_rule_review_seed_dataset.py -q
```

## 8. 扩量原则（从 1000 条继续扩）

1. **先加句式再加词表**：同一语义新问法（如倒装、反问、省略）比加一个地区更有价值
2. **每模板组合数设上限**（`TEMPLATE_VARIANT_CAP=25`）：防组合爆炸的模板挤占小模板分布，保持 9 工具均衡
3. **决策由脚本重判**：比较类换数值后 decision 必须重算，人工改词表后重跑生成
4. **负样本动态平衡**：扩句式时保持无工具题占 10-15%
