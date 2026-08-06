# 规则审查子系统 — 工作流程详解(带端到端示例)

> 配套文档:完整设计见 `docs/rule-review-design.md`(v2,2243 行);本文是"流程怎么走"的速读版,所有模块注释均按设计文档 § 节引用。
> 适用场景:理解系统运行方式、面试讲解、故障排查。

## 1. 系统概览

规则审查子系统回答「某交易/申报是否符合电力交易规则」类合规判断问题,走 **9 阶段流水线**:

```
用户问题
   │
   ▼
┌────────────────────────────────────────────────────────────────────┐
│ 0. 问题改写  →  1. 澄清判断  →  2. 多文档拆分  →  3. Query 优化     │
│                    (可能追问)          (拆分/不拆分)                  │
│                                                                      │
│ 4. RAG 检索  →  5. LLM 生成  →  6. Tool 调用  →  7. Judge 校验      │
│    (混合检索)    (第一轮推理)    (最多3轮循环)     (第二模型校验)     │
│                                                                      │
│ 8. SSE 输出(逐阶段进度标签 + 最终结果 JSON)                          │
└────────────────────────────────────────────────────────────────────┘
```

**关键代码位置**:

| 模块 | 文件 | 职责 |
|---|---|---|
| 编排器 | `src/rule_review/pipeline.py` | 9 阶段编排 + SSE 输出 + 分支兜底 |
| 问题改写 | `src/rule_review/query_rewriter.py` | 时间/实体/术语标准化(纯 Python) |
| 混合检索 | `src/rule_review/retriever.py` | Query 优化 + BM25/向量/RRF/精排 |
| 文档存储 | `src/rule_review/document_store.py` | 解析/切块/入库(Milvus 或 FAISS) |
| LLM 推理 | `src/rule_review/generator.py` | Prompt 组装 + 第一轮生成 |
| 工具系统 | `src/rule_review/tool_executor.py` | 工具注册表 + 调用循环(代码计算) |
| 结果校验 | `src/rule_review/judge.py` | 第二个 LLM 校验(幻觉/自洽/遗漏) |
| 审计 | `src/rule_review/audit.py` | 全程留痕 + 抽样质检 |

---

## 2. 端到端示例:走一遍完整流程

**用户问题**:

> 2025年3月15日冀北的日前电价800元/MWh是否超过上限?

假设检索库中已入库《省间应急调度交易规则(日前)》文档,其中第 12 条含表格:

| 地区 | 电价上限(元/MWh) |
|------|-----------------|
| 冀北 | 760 |

---

### 阶段 0:问题改写(`pipeline.py:271` → `query_rewriter.py`)

**做什么**:纯 Python 规则,把口语化问题标准化,让后续正则判断、检索、LLM 推理拿到干净输入。不调 LLM。

| 处理 | 规则 | 输入 → 输出 |
|---|---|---|
| 绝对日期标准化 | `_normalize_absolute_dates` | `2025年3月15日` → `2025-03-15` |
| 实体名归一化 | `_normalize_entities`(rule_terms.json 别名) | `冀北` 已是标准名,不变 |
| 术语别名替换 | `_normalize_terms`(rule_terms.json) | `日前电价` → `日前现货出清电价`、`上限` → `价格上限` |

**结果**:

```
输入: 2025年3月15日冀北的日前电价800元/MWh是否超过上限?
输出: 2025-03-15 冀北的日前现货出清电价800元/MWh是否超过价格上限?
```

---

### 阶段 1:澄清判断(`pipeline.py:276` → `check_clarification_needed`,`pipeline.py:118`)

**做什么**:纯正则判断问题三要素是否齐全,**缺什么追问什么**。

- 有时间?`_has_time_info()` → `2025-03-15` ✅
- 有实体?`_has_entity_info()` → `冀北` ✅
- 有比较意图?`_has_comparison_intent()` → `是否/超过` ✅

三要素齐全 → `needs_clarification=False`,进入下一阶段。

> 若缺要素(比如只问"冀北申报电价符合规则吗"),则 SSE 返回 `clarification` 事件直接结束,附建议句式(见 §3 分支 1)。

---

### 阶段 2:多文档拆分(`pipeline.py:287` → `split_if_multi_document`,`pipeline.py:161`)

**做什么**:用书名号正则 `《([^》]+)》` 提取问题中提到的文档名,与已入库文档匹配;提到 **≥2 个文档**时拆分为多个子问题,每个子问题限定检索对应文档。

本例问题未提及任何 `《文档名》` → 返回单文档任务:

```python
[{"sub_query": "2025-03-15 冀北的日前现货出清电价800元/MWh是否超过价格上限?",
  "doc_name": None, "doc_id": None}]   # None = 检索全部文档
```

---

### 阶段 3 + 4:Query 优化 + RAG 混合检索(`pipeline.py:293` → `retriever.py`)

**Query 优化**(`QueryOptimizer`,`retriever.py:71`):地名/术语归一化 + **同义词扩展**,生成检索变体列表:

```
变体1: 2025-03-15 冀北 日前现货出清电价 800元/MWh 价格上限
变体2: ...(同义词:日前价格 / 最高限价 / 上限值 ...)
```

**混合检索**(`retrieve`,`retriever.py:507`):

```
对每个变体:
  ├─ BM25 稀疏召回        (bm25s, top 30)
  └─ bge-m3 向量稠密召回   (Milvus 或 FAISS 本地索引, top 30)
        ↓
RRF 融合所有变体结果 → top 20
        ↓
Cross-Encoder 精排 → 最终 top 10
```

**返回**:

```
Chunk #1: 来源《省间应急调度交易规则(日前)》/ 第12条 电价上限 / 页码 5
  text: "……省间日前现货出清电价不得超过价格上限。| 地区 | 电价上限(元/MWh) |
         |------|-----------------| | 冀北 | 760 | ……"
```

> 兜底链(`retrieve_with_fallback`):正常检索无结果 → 用**原始 query**(不经优化)再向量检索(扩大 k)→ 再 BM25 → 仍无则 `not_found=True`,SSE 返回 `not_found` 事件(见 §3 分支 2)。

---

### 阶段 5:LLM 生成 — 第一轮(`pipeline.py:338` → `generator.generate`)

**Prompt 构成**(`build_messages`,`prompts.py:264`):

- **SystemMessage**:`SYSTEM_PROMPT_V2` = 角色设定 + 核心原则(必须基于原文、不得编造、证据必须原文引用、**精确计算必须调工具**)+ 严格 JSON 输出格式 + 决策指南 + 工具规则(从 `tools_config.json` 动态生成)
- **HumanMessage**:检索到的 chunk 原文 + 术语映射表 + 用户问题

**第一轮输出**(LLM 识别到需要精确比较,但按规则**不自己算**):

```json
{
  "decision": "",
  "reason": "",
  "evidence": [],
  "confidence": 0.0,
  "tool_calls": [
    {"tool": "extract_table_data",
     "args": {"table_text": "| 地区 | 电价上限(元/MWh) |\n|------|-----------------|\n| 冀北 | 760 |",
              "filter_column": "地区", "filter_value": "冀北",
              "select_column": "电价上限(元/MWh)"}},
    {"tool": "arithmetic_compare",
     "args": {"actual": 800, "operator": "gt", "threshold": 760}}
  ]
}
```

**关键**:第一轮 LLM 可以"先说不会,再回来答"——它只负责**决定调哪个工具、传什么参数**,精确比较本身交给代码。

> 若 LLM 输出 `not_found: true`(认为文档中没有相关规则)→ 短路,直接输出结果,不进入后续阶段。

---

### 阶段 6:Tool 调用循环(`pipeline.py:377` → `execute_with_tool_loop`,`tool_executor.py:512`)

```
第 1 轮: 执行 tool_calls(纯 Python 代码)
  ├─ extract_table_data → 从表格文本定位"冀北"行 → {"value": 760, "unit": "元/MWh"}
  └─ arithmetic_compare(800, "gt", 760)
       → {"success": true,
          "data": {"result": false, "expression": "800 > 760",
                   "detail": "实际值800未超出阈值760"}}
  │
  ↓ 工具结果格式化注入上下文,追加指令"请基于工具执行结果重新生成完整审查结果 JSON"
第 2 轮: LLM 基于真实计算结果生成最终答案
```

**第二轮输出**(工具循环正常终止,tool_calls 清空):

```json
{
  "decision": "符合",
  "reason": "根据《省间应急调度交易规则(日前)》第12条,冀北日前现货出清电价上限为760元/MWh。"
             "经工具计算,实际值800元/MWh未超过上限760元/MWh(800 > 760 = false),符合规则。",
  "evidence": [
    {"source": "省间应急调度交易规则(日前)", "section": "第12条 电价上限", "page": 5,
     "text": "| 冀北 | 760 |"}
  ],
  "confidence": 0.95,
  "tool_calls": []
}
```

**终止与降级条件**(防死循环):

| 条件 | 行为 |
|---|---|
| tool_calls 为空 | ✅ 正常终止 |
| 达到 3 轮(`MAX_TOOL_ROUNDS`) | 降级:让 LLM 尽力判断,标记 `tool_unsolved: true` |
| 总超时 30s(`TOOL_TOTAL_TIMEOUT`) | 同上 |
| 单轮全部工具执行失败 | 降级(注入失败原因) |

---

### 阶段 7:Judge 校验(`pipeline.py:392` → `verify_with_fallback`,`judge.py:365`)

**做什么**:第二个 LLM(独立 `JUDGE_MODEL` 配置,与生成模型不同)**对照原文复核**第一轮结果。输入四样:原始问题 + 全部检索 chunk 原文 + LLM 审查结果 + **工具调用日志**。

三项检查(`JUDGE_SYSTEM_PROMPT`,`judge.py:38`):

| 检查 | 本例判定 |
|---|---|
| ① 幻觉检测:evidence 的 text 是否真在原文? | `| 冀北 | 760 |` 可在 chunk 原文找到 → 真实 |
| ② 逻辑自检:decision 与 reason 是否自洽? | "800 未超 760" 与 decision="符合" 一致;工具日志 `result: false` 佐证 → 自洽 |
| ③ 遗漏检查:关键规则是否未引用? | 第 12 条已引用,无其他适用条款 → 无遗漏 |

**输出**(Judge 修正后的最终结果会**覆盖**原 LLM 输出):

```json
{
  "verified": true,
  "corrections": [],
  "hallucinated_evidence": [],
  "missing_rules": [],
  "final_decision": "符合",
  "final_reason": "……(同上)",
  "confidence": 0.95
}
```

**兜底原则**:Judge 超时(60s)/API 异常/输出解析失败 → **跳过校验**,绝不阻断主流程,`judge_skipped` 标记写入审计。

---

### 阶段 8:SSE 输出(`pipeline.py` 各 `yield`)

前端按序收到(每阶段一条进度标签 + 最终结果):

```text
data: {"type":"messageLabel","answer":"- <span>查询预处理中...</span>","stage":"rewrite"}

data: {"type":"messageLabel","answer":"- <span>问题分析中...</span>","stage":"clarification"}

data: {"type":"messageLabel","answer":"- <span>检索相关知识中...</span>","stage":"retrieval"}

data: {"type":"messageLabel","answer":"- <span>规则推理中...</span>","stage":"generation"}

data: {"type":"messageLabel","answer":"- <span>工具调用中...</span>","stage":"tool"}

data: {"type":"messageLabel","answer":"- <span>结果校验中...</span>","stage":"judge"}

event: message
data: {"answer": "{最终审查结果 JSON}", "type": "content"}

event: done
data: {"done": true, "query_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"}
```

同时,本次请求的全过程(改写后 query、检索 chunks、工具日志、Judge 结果、各阶段耗时)写入 `AuditStore`(`audit.py`),支持按 query_id 追溯与抽样质检。

---

## 3. 分支场景速查

| # | 触发条件 | 走向 |
|---|---|---|
| 1 | 澄清判断:缺时间/实体/比较意图 | 返回 `clarification` 事件 + 建议句式,流程结束 |
| 2 | 检索无结果(含扩大策略后) | 返回 `not_found` 事件,流程结束 |
| 3 | LLM 认为文档无相关规则 | 返回 `not_found: true` 的结果,短路 |
| 4 | 问题提到 ≥2 个文档 | 拆分子问题 → 并发检索(`asyncio.to_thread`)→ `_merge_retrieve_results` 合并去重 |
| 5 | 工具循环超时/超轮数/全失败 | `_fallback_generate` 尽力生成,标记 `tool_unsolved` |
| 6 | Judge 不可用/超时/输出异常 | 跳过校验,`judge_skipped` 留痕 |
| 7 | LLM 生成失败(网络/解析) | SSE `error` 事件返回"无法判断",流程结束 |
| 8 | 基础设施降级 | Redis 不可用 → 内存缓存;Milvus 不可用 → FAISS;PG 不可用 → JSON 文件 |

---

## 4. 人机分工总览(理解本系统的钥匙)

| 判断类型 | 负责方 | 为什么 |
|---|---|---|
| 语义判断:条款是否适用、结论、推理过程 | LLM(第一模型) | 语义理解是 LLM 强项 |
| **精确数值比较/单位换算/表格提取** | **纯 Python 代码**(ToolExecutor) | "800 > 760?" 代码 100% 准确 |
| 结果复核:幻觉/自洽/遗漏 | LLM(第二模型 Judge) | 双模型互检,避免系统性盲区 |
| 问题是否清晰、是否多文档、检索有无结果、超时降级 | 纯规则代码 | 确定性逻辑不依赖模型 |

**两条保障链**:

```
精确性保障: LLM 提出比较 → 代码执行比较 → LLM 基于结果定论
             (prompts.py 强制"必须调工具,不得自行估算")
可信性保障: LLM 生成结论 → 第二模型对照原文复核 → 修正后输出
             (judge.py 三项检查,修正结果覆盖原输出)
```

---

## 5. 常见问题

**Q:为什么第一轮 LLM 输出是空的 decision?**
因为精确比较未完成,按 Prompt 约定它先把 `tool_calls` 抛给代码,第二轮拿到结果才下结论。这是 Phase 2 工具系统的核心设计。

**Q:流式路径与上面流程有区别吗?**
有。SSE 默认走 `pipeline.execute_stream`,它内部调用的 `generate()` 是**非流式**(单次出完整 JSON);工具循环与 Judge 均挂在此路径。`generate_stream`(`generator.py:229`)是逐 token 流式,用于轻量场景,无工具循环与 Judge。也就是说**线上默认路径具备完整校验链**。

**Q:Judge 会不会改错?**
Judge 的修正也是 LLM 输出,同样有幻觉概率;但它是**对照原文复核**而不是凭空生成,且修正会带 `judge_corrections` 留痕,可供人工质检(`GET /v1/rule-review/audit/...`)。

**Q:审计里能查到什么?**
`AuditStore` 记录:改写前后 query、检索结果摘要、工具调用日志(含每轮耗时)、Judge 判定与修正、各阶段延迟、最终输出。离线评估(`evaluation.py`)可基于这些数据回放计算指标。
