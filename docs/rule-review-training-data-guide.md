# 规则审查训练数据编写指南

本指南面向**手工编写规则种子数据集**（Qwen3-3B 级模型工具调用训练），说明三个核心问题：**怎么从问题定工具链、怎么定决策、怎么写句式才能撑起数据量**。

数据链路：`rule_review_seed_spec.json`（v3：句式模板 + 槽位词表 + 判分信号字段）→ `scripts/build_seed_dataset.py`（组合生成 1800 条完整种子）→ `scripts/validate_seed_dataset.py`（三层校验 + 覆盖统计）。v3 起每条种子携带 `expected_keywords` / `expected_sources` 证据信号字段（RL 纯规则 verifier 的判分真值），派生逻辑在 `scripts/seed_signal_fields.py`（build 与 validate 共用，防规则漂移）。

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

## 5. 种子规格文件结构（v3）

```json
{
  "version": 3,
  "facts": { "price_cap_mwh": 760, "min_quantity_kwh": 10000, "docs": [...pdf...] },
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
      "expected_decision": "",
      "expected_keywords": ["价格上限", "760"],
      "expected_sources": ["省间电力现货交易规则"]
    }
  ]
}
```

`decision_rule` 取值：
- `gt760_bad`：槽位 mwh > 760 → 不符合，否则符合（价格上限语义）
- `gte_10000_good`：kwh ≥ 10000 → 符合，否则不符合（交易量下限语义）
- `fixed`：直接用 `expected_decision`

**判分信号字段（v3，RL verifier 真值）**：
- `expected_keywords`：静态关键词——规则主题词（1-2 个）+ 基线数值（"760"/"10000"）；短实体（≤12 字符、无标点），与 `slot_vocab` 槽位派生合并后写入变体
- `expected_sources`：静态来源——∈ 来源白名单（`facts.docs` 去 `.pdf` ∪ `doc_name` ∪ `other_doc_name`，当前 5 个）；单文档模板 1 个、multi_doc/conflict 2 个
- **负样本**（`expected_tools` 为空）：两字段必须为空列表 `[]`——与 `expected_tools` 空双向强制（防模型编造证据；`evaluation.py` 对空 expected 的 keyword/source recall 恒 1.0，负样本只判「零工具 + 无法判断 + 无幻觉」）
- 静态可空的模板（如 locate 类）：必须存在**非排除槽位**提供派生（条款号/文档名），否则校验拦截

**槽位派生（自动，build 时合并）**：每个变体携带 `expected_keywords` + `keyword_evidence`（{static, derived}）与 `expected_sources` + `source_evidence`——与 `decision_evidence` 同构，validator 按派生规则重算比对防手改漂移。
- keyword 派生：每个槽位填充文本；带 `mwh` 的条目附加数值字符串（"850元/MWh" → 派生 "850元/MWh" + "850"）
- **排除槽位**（不派生关键词）：`out_region` / `no_such_article`（未命中→修正语义，强制复述会误罚 recovery）、`citation_text`（整句引用超长，核验由工具结果驱动）
- source 派生：`doc_name` / `other_doc_name` 填充文本

**约束**：`query_template` 中的 `{占位符}` 必须与 `slots` 声明一致（校验脚本强制）；`expected_workflow` 中的工具名必须以 `工具名(...)` 形式出现（校验正则依赖）。

## 6. 场景矩阵与分布目标

| 类别 | 模板数 | 覆盖内容 |
|---|---|---|
| 单工具 × 9 | ~50 | 每工具 3-6 句式（正例 + 边界：表外地区/未命中条款） |
| 工具链 | ~35 | extract→compare、unit→compare、locate→conflict、cross_ref→locate、date 前置等 |
| 判定型 | ~40 | compare 23（超限判定）、conflict 13（冲突/一致）、date 15（含「已废止/不适用」判定）、unit 判定 6 |
| 多文档 | ~7 | 跨文档定位 + 冲突/一致性判断 |
| 错误恢复 | ~7 | 模糊参数（「华北」）→ 修正参数重试 |
| 负样本 | ~34 | 纯规则判断、主观解释、流程题、诱惑场景——**占比 10-15%**（防工具滥用） |

**分布目标**（`validate_seed_dataset.py` 输出核对）：
- 9 工具全覆盖（每个工具 ≥ 90 条）
- 负样本占比 10-15%
- **判定型（符合/不符合/部分符合）占比 25-30%**——RL 决策 reward（R3）的训练信号；当前 1800 条产出 28.4%
- 1800 条 query 去重率 100%
- 单工具参数多样性：region ≥ 15、price ≥ 8、含单位混用变体

## 7. 种子数据集评估方法

评估分**三层**：① 自动化结构校验（脚本强制，全量）→ ② 自动化决策一致性校验（脚本强制，全量）→ ③ 人工语义抽检（抽样）。前两层由 `scripts/validate_seed_dataset.py` 全量执行。

### 7.1 规格层校验（模板与词表本身）

| 检查项 | 说明 |
|---|---|
| 槽位引用存在 | 模板 `slots` 声明的每个槽位都必须在 `slot_vocab` 中定义 |
| 占位符一致性 | `query_template` 中出现的 `{占位符}` 集合 == `slots` 声明集合（多写/漏写都报错） |
| decision_rule 合法 | ∈ {fixed, gt760_bad, gte_10000_good}；fixed 时 `expected_decision` ∈ 4 枚举 |
| 工具名合法 | `expected_tools` 每个工具 ∈ `tools_config.json` 白名单 |
| 信号字段声明 | 模板必须声明 `expected_keywords` / `expected_sources`（v3 强制；负样本可为空列表） |
| 静态关键词 | 非负样本静态关键词为空时必须有非排除槽位提供派生；关键词 ≤12 字符、无标点 |
| 来源白名单 | 静态 `expected_sources` 每个元素 ∈ 来源白名单（facts.docs 去 .pdf ∪ doc_name ∪ other_doc_name） |

### 7.2 数据集层校验（每条生成种子）

| 检查项 | 说明 |
|---|---|
| 占位符残留 | query 中不得出现 `{` / `}`（生成时替换必须彻底） |
| 工具名合法 | 同规格层 |
| 决策枚举 | `expected_decision` ∈ {符合, 不符合, 部分符合, 无法判断} |
| workflow 一致性 | `expected_workflow` 中提及的工具名集合（按 `工具名(` 正则提取）== `expected_tools` 集合——防「写了工具链但没声明工具」或反之 |
| **决策与数值基线一致性** | 每条变体携带 `decision_evidence`（{rule, mwh, baseline}，生成时记录），校验时按规则重算比对：`gt760_bad`：mwh > 760 → 不符合；`gte_10000_good`：kwh ≥ 10000 → 符合。**决策与依据不符即拦截**——防止人工改词表/决策时引入系统性标注错误 |
| **信号字段互斥** | `expected_tools` 空 ↔ `expected_keywords`/`expected_sources` 空，双向强制（防负样本误标关键词导致模型编造证据） |
| **信号字段质量** | 关键词 ≤12 字符、无标点；来源 ∈ 白名单 |
| **信号一致性重算** | 每条变体携带 `keyword_evidence` / `source_evidence`（{static, derived}，生成时记录），校验时按「static ∪ derived 去重」重算比对——**手改字段与派生规则不符即拦截** |

### 7.3 覆盖统计（多样性指标，防「数据全面但集中」）

| 指标 | 目标 |
|---|---|
| 总条数 | 1800（RL 种子；SFT 阶段曾为 1000） |
| 9 工具全覆盖 | 每工具 ≥ 90 条（主工具 extract/compare/locate 越多越好） |
| 负样本占比 | 10-15%（无工具题，防工具滥用） |
| **判定型占比** | **25-30%**（符合/不符合/部分符合——RL 决策 reward 信号） |
| 句式模板数 | ≥ 160（句式多样性——同一语义问法越多，模型泛化越强） |
| query 去重率 | 100%（模板×槽位组合冲突会暴露为重复） |
| 词表实体多样性 | region ≥ 15、price ≥ 8、含单位混用变体 |

### 7.4 人工抽检（语义层，脚本无法覆盖）

- 从 12 个类别各抽 1-2 条，重点核对：**工具链选择是否正确**（该用 unit_converter 却只调 compare？）、**决策标注是否正确**、**query 是否自然**（真实用户不会这么问）
- 抽检发现的问题回改 `rule_review_seed_spec.json`（模板/词表），**重跑生成**（决策自动重判）而非手改数据集——保证「规格是唯一事实源」

### 7.5 后续阶段的扩展评估（本次未做）

- **执行回测**：把每条种子的 `expected_tools` 参数用 `ToolExecutor` 真实执行，验证「工具链能跑通、结果与决策一致」——语义正确性的自动化延伸
- **模型评测**：冻结测试集（`test_cases.json`），三档模型（qwen3-max / SFT / GRPO）对比 decision_accuracy 与工具调用率——数据集有效性的最终检验

## 8. 生成与校验流程

```bash
# 1. 编辑 data/evaluation/rule_review_seed_spec.json（加句式/加词表/补信号字段）
# 2. 生成 1800 条完整种子（无占位符残留，决策自动重判，信号字段自动合并）
conda run -n dataquery python scripts/build_seed_dataset.py
# 3. 三层评估：规格层 + 数据集层 + 覆盖统计（含判定型占比、来源白名单）
conda run -n dataquery python scripts/validate_seed_dataset.py
# 4. 人工抽检 12 类代表样本（工具链/决策标注/信号字段合理性）
# 5. 单元测试（含真实 spec 集成：生成→校验全过 + 判定型占比断言）
conda run -n dataquery python -m pytest tests/test_rule_review_seed_dataset.py -q
```

**信号字段的一致性保证**：`scripts/seed_signal_fields.py` 是派生规则的唯一实现，build（生成）与 validate（重算比对）共用——改模板/词表后重跑生成与校验，规则漂移即被拦截。

## 9. 扩量原则（从 1800 条继续扩）

1. **先加句式再加词表**：同一语义新问法（如倒装、反问、省略）比加一个地区更有价值
2. **每模板组合数设上限**（`TEMPLATE_VARIANT_CAP=25`）：防组合爆炸的模板挤占小模板分布，保持 9 工具均衡
3. **决策由脚本重判**：比较类换数值后 decision 必须重算，人工改词表后重跑生成
4. **负样本动态平衡**：扩句式时保持无工具题占 10-15%
