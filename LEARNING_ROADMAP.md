# 智能数据查询系统 — 学习路线

## 项目一句话概括

这是一个**省间电力数据自然语言查询服务**：用户用中文提问（如 "昨天华东地区日前交易电量多少？"），系统自动识别意图 → 提取参数 → 调用下游电力 API → 聚合计算 → 格式化输出 Markdown + ECharts 图表，以 SSE 流式返回结果。

---

## 思维导图：以一次请求为例的完整函数调用树（含每阶段输入/输出）

> **贯穿示例**：`POST /v1/query`  
> **用户提问**："查询2025年6月1日到6月15日华东地区日前应急调度交易电量"

```text
═══════════════════════════════════════════════════════════════════════════════
全局输入:  QueryRequest{question: "查询2025年6月1日到6月15日华东地区日前应急调度交易电量",
                        stream: true, sessionId: "abc-123", ...}
═══════════════════════════════════════════════════════════════════════════════
│
├─ 入口: query_router.py:225  @router.post("")
│   函数: async def query(req: QueryRequest)
│   ┌─────────────────────────────────────────────────────────────────────┐
│   │  输入: QueryRequest (Pydantic Model)                                │
│   │  内部:                                                              │
│   │    conversation_id = req.sessionId or str(uuid4())  → "abc-123"    │
│   │    query_id        = str(uuid4())                   → "q-456"     │
│   │    user_id         = get_user_id(req)               → "anonymous"   │
│   │    workflow = await workflow_factory.create_workflow("abc-123")     │
│   │  输出: StreamingResponse (text/event-stream)                        │
│   └─────────────────────────────────────────────────────────────────────┘
│
├─ query_router.py:257  async def generate():  ← SSE 异步生成器
│   │
│   ▼
╔════════════════════════════════════════════════════════════════════════════╗
║              WorkflowRouter.execute_stream()                               ║
║              src/workflow/workflow_router.py:2439                          ║
║  输入: user_query="查询2025年6月1日...", user_id="anonymous", query_id="q-456"
║  输出: AsyncGenerator (yield {"data": ..., "type": ...})                   ║
╚════════════════════════════════════════════════════════════════════════════╝
│
├─── 阶段0: 问题改写 ───────────────────────────────────────────────────────
│   ┌─────────────────────────────────────────────────────────────────────┐
│   │  输入: "查询2025年6月1日到6月15日华东地区日前应急调度交易电量"        │
│   │                                                                     │
│   │  调用链:                                                            │
│   │    rewrite_workflow.py: RewriteWorkflow.rewrite(user_query)          │
│   │      ├─ 模板匹配  (rewrite_knowledge.json, 172个模板)                │
│   │      ├─ 时间标准化: "6月1日" → "2025-06-01"                          │
│   │      ├─ 实体标准化: "华东" → "华东分部"                              │
│   │      └─ 节假日解析: holidays.json                                    │
│   │                                                                     │
│   │  输出: "查询2025年6月1日到6月15日华东分部日前应急调度交易电量"         │
│   └─────────────────────────────────────────────────────────────────────┘
│
├─── 阶段1: 意图识别 第一阶段 (28类业务域路由) ──────────────────────────────
│   ┌─────────────────────────────────────────────────────────────────────┐
│   │  输入: "查询2025年6月1日到6月15日华东分部日前应急调度交易电量"        │
│   │  + system_prompt: Unified_Intent_Recognition_Prompt (prompts.py)     │
│   │                                                                     │
│   │  调用链:                                                            │
│   │    IntentAgent(model=intent_model, prompt_template=xxx)              │
│   │      └─ agent.ainvoke(content=改写后的问题)                           │
│   │           ├─ @before_model: trim_messages_by_length()  裁剪历史消息   │
│   │           ├─ LLM 推理 (DashScope / 内网代理)                         │
│   │           └─ @after_model: parse_json_block()      提取 JSON 块      │
│   │                                                                     │
│   │  输出 (intent_result):                                              │
│   │    {                                                                 │
│   │      "intents":   ["省间应急调度交易信息（日前）查询"],                │
│   │      "time_range": {"start": "2025-06-01", "end": "2025-06-15"},    │
│   │      "entities":  ["华东分部"],                                      │
│   │      "question":  "查询2025年6月1日到6月15日华东分部日前应急调度交易电量"│
│   │    }                                                                 │
│   └─────────────────────────────────────────────────────────────────────┘
│
├─── 阶段2: 向上层传递 intents ─────────────────────────────────────────────
│   ┌─────────────────────────────────────────────────────────────────────┐
│   │  输入: intents=["省间应急调度交易信息（日前）查询"]                    │
│   │  yield {"intents": intents}   →  query_router 缓存, 后续生成推荐链接  │
│   │  输出: 无 (纯传递信号)                                                │
│   └─────────────────────────────────────────────────────────────────────┘
│
├─── 阶段3: 问题拆分 ───────────────────────────────────────────────────────
│   ┌─────────────────────────────────────────────────────────────────────┐
│   │  输入: user_query + intent_result (含 intents + entities)             │
│   │                                                                     │
│   │  调用链:                                                            │
│   │    _resolve_sub_items(user_query, intent_result)                     │
│   │      ├─ 单意图 + 单主体 → 不拆分                                     │
│   │      └─ 多意图/多主体 → QuestionSplitWorkflow.split() (8条拆分规则)    │
│   │                                                                     │
│   │  输出: sub_items = [(                                                  │
│   │    "查询2025年6月1日到6月15日华东分部日前应急调度交易电量",              │
│   │    "省间应急调度交易信息（日前）查询"                                   │
│   │  )]                                                                  │
│   │  → len=1, 走单子问题分支                                              │
│   └─────────────────────────────────────────────────────────────────────┘
│
├─── 阶段4: 意图识别 第二阶段 (域内专项识别) ───────────────────────────────
│   ┌─────────────────────────────────────────────────────────────────────┐
│   │  输入: user_query + intent="省间应急调度交易信息（日前）查询"          │
│   │  + system_prompt: EmergencyDayhead_Intent_Recognition_Prompt          │
│   │                                                                     │
│   │  调用链:                                                            │
│   │    _recognize_domain_intent(intent, user_query)                      │
│   │      └─ IntentAgent(model, EmergencyDayhead_Intent_Recognition_Prompt)│
│   │           └─ agent.ainvoke()                                         │
│   │                                                                     │
│   │  输出 (detail_result):                                              │
│   │    {                                                                 │
│   │      "intent":     "省间应急调度交易信息（日前）查询",                  │
│   │      "time_range": {"start": "2025-06-01", "end": "2025-06-15"},    │
│   │      "operations": [                                                 │
│   │        {"operation": "filter",   "field": "sendrecv", "value": "送"}, │
│   │        {"operation": "group_by", "field": "businessTime",            │
│   │                                 "date_granularity": "day"},           │
│   │        {"operation": "sum",      "field": "交易电量"}                  │
│   │      ]                                                               │
│   │    }                                                                 │
│   │  ★ 第二阶段比第一阶段多了 operations 字段，这是后续聚合的关键输入 ★    │
│   └─────────────────────────────────────────────────────────────────────┘
│
├─── 阶段5: 获取业务工作流实例 ──────────────────────────────────────────────
│   ┌─────────────────────────────────────────────────────────────────────┐
│   │  输入: intent="省间应急调度交易信息（日前）查询"                       │
│   │                                                                     │
│   │  _get_workflow(intent)                                              │
│   │    ├─ WORKFLOW_REGISTRY 查找 → EmergencyDayaheadWorkflow             │
│   │    └─ 创建实例: EmergencyDayaheadWorkflow(                            │
│   │          conversation_id, parameter_model, format_model,              │
│   │          echarts_model, trend_analysis_model, EMERGENCY_DAYAHEAD_API_URL│
│   │        )                                                             │
│   │                                                                     │
│   │  输出: EmergencyDayaheadWorkflow 实例                                 │
│   └─────────────────────────────────────────────────────────────────────┘
│
│   ┌─ workflow.execute_stream(user_query, detail_result) ───────────────┐
│   │                                                                     │
│   ▼                                                                     │
╔════════════════════════════════════════════════════════════════════════════╗
║              BaseWorkflow.execute_stream()                                 ║
║              src/workflow/base_workflow.py:1784                            ║
║  输入: user_query=改写后的问题, intent_result=detail_result                ║
║  输出: AsyncGenerator (SSE chunks: content / messageLabel / error / ...)  ║
╚════════════════════════════════════════════════════════════════════════════╝
│   │
│   ├─── 步骤A: 参数提取 ───────────────────────────────────────────────────
│   │   ┌─────────────────────────────────────────────────────────────────┐
│   │   │  输入: user_query="..."                                          │
│   │   │       intent_result={..., "operations": [...], ...}             │
│   │   │       + system_prompt: get_parameter_prompt()  (子类覆盖)        │
│   │   │                                                                 │
│   │   │  调用链:                                                        │
│   │   │    _extract_parameters()                                        │
│   │   │      └─ ParameterAgent.ainvoke(user_query, intent_result,        │
│   │   │                                 prompt_template=子类覆盖的提示词) │
│   │   │           ├─ @before_model: trim_messages_by_length()            │
│   │   │           ├─ LLM 提取结构化参数                                  │
│   │   │           └─ @after_model: parse_json_block()                   │
│   │   │      └─ _adjust_time_range_end(): end += 1天                   │
│   │   │      └─ data_matching(): "华东"→"华东分部" (difflib 模糊匹配)     │
│   │   │                                                                 │
│   │   │  输出 (params):                                                 │
│   │   │    {                                                             │
│   │   │      "time_range":       {"start": "2025-06-01",                 │
│   │   │                           "end": "2025-06-16"},   ← 自动 +1 天   │
│   │   │      "name_abbreviation": "华东分部",             ← 标准化匹配    │
│   │   │      "sendrecv":         "送",                                  │
│   │   │      "fields":           ["交易电量"],                            │
│   │   │      "operations":       [...]                                  │
│   │   │    }                                                             │
│   │   └─────────────────────────────────────────────────────────────────┘
│   │
│   ├─── 步骤B: 参数验证 ───────────────────────────────────────────────────
│   │   ┌─────────────────────────────────────────────────────────────────┐
│   │   │  输入: params (上一步输出)                                       │
│   │   │  EmergencyDayaheadWorkflow.validate_params(params)               │
│   │   │    检查: time_range 完整? 实体名称有效? sendrecv 合法?            │
│   │   │                                                                 │
│   │   │  输出: {"valid": true}  或  {"valid": false, "message": "..."}   │
│   │   └─────────────────────────────────────────────────────────────────┘
│   │
│   ├─── 步骤C: API 调用 ───────────────────────────────────────────────────
│   │   ┌─────────────────────────────────────────────────────────────────┐
│   │   │  输入: params                                                   │
│   │   │                                                                 │
│   │   │  调用链:                                                        │
│   │   │    call_api(params)                                             │
│   │   │      └─ _call_api_impl(params)   ← EmergencyDayaheadWorkflow 实现│
│   │   │           ├─ 构造请求体 (映射到网关协议):                         │
│   │   │           │  {startDate, endDate, name_abbreviation, ...}       │
│   │   │           └─ httpx.AsyncClient.post(EMERGENCY_DAYAHEAD_API_URL)  │
│   │   │               共享连接池 (base_workflow.py:43, 全局单例)          │
│   │   │                                                                 │
│   │   │  输出 (api_result):                                             │
│   │   │    {                                                             │
│   │   │      "status": "success",                                       │
│   │   │      "data": [                                                   │
│   │   │        {"businessTime": "2025-06-01T08:00:00Z",                  │
│   │   │         "sendrecv": "送",                                        │
│   │   │         "交易电量": 12345.6,   "交易电力": 800.3,                │
│   │   │         "v0015": 200, "v0030": 210, ..., "v2400": 180},         │
│   │   │        {"businessTime": "2025-06-02T08:00:00Z", ...},           │
│   │   │        ...  (15天的数据, 每天一条记录, 含96个时间点字段)           │
│   │   │      ]                                                           │
│   │   │    }                                                             │
│   │   └─────────────────────────────────────────────────────────────────┘
│   │
│   ├─── 步骤D: 空数据兜底 (30天回溯) ──────────────────────────────────────
│   │   ┌─────────────────────────────────────────────────────────────────┐
│   │   │  输入: raw_data (步骤C输出)                                      │
│   │   │                                                                 │
│   │   │  _retry_with_expanded_time_range_if_empty(raw_data, params, ...) │
│   │   │    ├─ 如果有数据 → 跳过, 原样返回                                 │
│   │   │    └─ 如果为空   → time_range 向前扩展 30 天 → 重新调用步骤C     │
│   │   │                                                                 │
│   │   │  输出: 最终 raw_data (与步骤C格式相同, 但有数据)                    │
│   │   └─────────────────────────────────────────────────────────────────┘
│   │
│   ├─── 步骤E: 操作清洗 + 数据处理 + 聚合 ──────────────────────────────────
│   │   ┌─────────────────────────────────────────────────────────────────┐
│   │   │  输入: raw_data (API原始数据) + operations (来自阶段4)             │
│   │   │                                                                 │
│   │   │  E1. sanitize_operations(operations)                             │
│   │   │      ─→ 丢弃无效操作 / 去重 / 时间点字段误判纠正                   │
│   │   │      输出: 清洗后的 operations (与输入可能不同)                    │
│   │   │                                                                 │
│   │   │  E2. process_data(raw_data, params, user_query)                  │
│   │   │      ─→ EmergencyDayaheadWorkflow.process_data()                  │
│   │   │      输出: 清洗后的数据列表 (字段重命名/类型转换/空值处理)          │
│   │   │                                                                 │
│   │   │  E3. aggregation_agent.ainvoke(user_query, operations, data)     │
│   │   │      ─→ 纯 Python 执行, 不调 LLM ──                               │
│   │   │      执行顺序: filter → group_by → top_n/bottom_n → 聚合          │
│   │   │      │                                                           │
│   │   │      ├─ raw_filter(       data, field="sendrecv",                 │
│   │   │      │                  operator="fuzzy", value="送")             │
│   │   │      │  → 过滤后数据 (只保留送端记录)                              │
│   │   │      │                                                           │
│   │   │      ├─ raw_group_by(     过滤后数据, field="businessTime",       │
│   │   │      │                   date_granularity="day")                  │
│   │   │      │  → 按天分组的数据 [{date, records}, ...]                   │
│   │   │      │                                                           │
│   │   │      └─ raw_sum_field(   分组数据, field="交易电量")               │
│   │   │         → 每组交易电量求和                                         │
│   │   │                                                                 │
│   │   │      输出 (agg_result):                                          │
│   │   │      {                                                           │
│   │   │        "summary": [                                              │
│   │   │          {"operation": "sum", "field": "交易电量",                 │
│   │   │           "result": 987654.3}                                    │
│   │   │        ],                                                        │
│   │   │        "aggregated_data": [                                      │
│   │   │          {"businessTime": "2025-06-01", "交易电量": 65432.1},      │
│   │   │          {"businessTime": "2025-06-02", "交易电量": 67890.5},      │
│   │   │          ...  (15行, 每天一行)                                    │
│   │   │        ]                                                         │
│   │   │      }                                                           │
│   │   └─────────────────────────────────────────────────────────────────┘
│   │
│   ├─── 步骤F: 聚合结果决策 (base_workflow.py:1871-1929, ~100行if-else) ────
│   │   ┌─────────────────────────────────────────────────────────────────┐
│   │   │  输入: agg_result (summary + aggregated_data)                    │
│   │   │       + operations 操作类型组合                                  │
│   │   │                                                                 │
│   │   │  决策规则:                                                       │
│   │   │  ┌─────────────────────────────────────────────────────────┐    │
│   │   │  │ group_by + count + 无 top_n/bottom_n    → summary       │    │
│   │   │  │ group_by + max/min + 无 top_n/bottom_n → 极值所在行      │    │
│   │   │  │ top_n/bottom_n/group_by 或行间运算      → aggregated_data│    │
│   │   │  │ sum/average/max/min/count 单独使用      → summary       │    │
│   │   │  └─────────────────────────────────────────────────────────┘    │
│   │   │                                                                 │
│   │   │  本例: group_by(天) + sum(交易电量) → 命中最后一条规则            │
│   │   │                                                                 │
│   │   │  输出 (processed_data / display_data):                           │
│   │   │    [{"businessTime": "2025-06-01", "交易电量": 65432.1}, ...]     │
│   │   │    即 aggregated_data, 15条记录                                  │
│   │   └─────────────────────────────────────────────────────────────────┘
│   │
│   ├─── 步骤G: 格式化决策 ─────────────────────────────────────────────────
│   │   ┌─────────────────────────────────────────────────────────────────┐
│   │   │  输入: display_data (步骤F输出)                                  │
│   │   │       └─ handle_summary_input()  →  压缩为 summary_data         │
│   │   │                                                                 │
│   │   │  _should_use_format_summary(summary_data)                        │
│   │   │    json.dumps(summary_data) → len(...)                           │
│   │   │      ≤ 28000 字符 → True  : 走 FormatAgent LLM 生成 Markdown    │
│   │   │      > 28000 字符 → False : 跳过格式化, 走趋势分析路径            │
│   │   │                                                                 │
│   │   │  本例: 15条记录, 远小于 28000 → 走 LLM 格式化                     │
│   │   │                                                                 │
│   │   │  输出: should_use_format_summary = True                          │
│   │   │        canEcharts() = True  (检测到可绘图数据)                   │
│   │   └─────────────────────────────────────────────────────────────────┘
│   │
│   ├─── 步骤H: 格式化 + 图表 (并发执行) ────────────────────────────────────
│   │   ┌─────────────────────────────────────────────────────────────────┐
│   │   │                                                                 │
│   │   │  ┌─ (并发任务1) 格式化流 ─────────────────────────────────┐     │
│   │   │  │  输入: user_query + summary_data (15条汇总数据)          │     │
│   │   │  │                                                        │     │
│   │   │  │  _format_result_stream(user_query, summary_data)        │     │
│   │   │  │    └─ FormatAgent.astream(data=json.dumps(15条),         │     │
│   │   │  │                           prompt_template=get_format_prompt())│
│   │   │  │         └─ LLM 流式生成 Markdown 文本                     │     │
│   │   │  │              └─ filter_think_tags_async() 过滤<think>标签 │     │
│   │   │  │                                                        │     │
│   │   │  │  输出 (流式, 逐chunk yield):                             │     │
│   │   │  │    "## 查询结果\n"                                       │     │
│   │   │  │    "| 日期 | 交易电量(万kWh) |\n"                        │     │
│   │   │  │    "|------|--------------|\n"                          │     │
│   │   │  │    "| 2025-06-01 | 65432.1 |\n"                        │     │
│   │   │  │    ... (15行表格 + 趋势描述)                              │     │
│   │   │  └────────────────────────────────────────────────────────┘     │
│   │   │                                                                 │
│   │   │  ┌─ (并发任务2) ECharts 图表 ─────────────────────────────┐     │
│   │   │  │  输入: user_query + intent + data_for_chart (15条原始数据)│     │
│   │   │  │                                                        │     │
│   │   │  │  create_echarts(user_query, intent, data)                │     │
│   │   │  │    ├─ [代码路径优先]                                      │     │
│   │   │  │    │  build_dense_time_series_echarts(data)              │     │
│   │   │  │    │    ├─ is_dense_time_series_records()  → 检测96点数据│     │
│   │   │  │    │    ├─ sort_time_fields_batch()         → v0015..v2400│     │
│   │   │  │    │    ├─ has_non_zero_series()            → 过滤全零   │     │
│   │   │  │    │    └─ 构造 {xAxis:["v0015"..], series:[...], ...}  │     │
│   │   │  │    │                                                │     │
│   │   │  │    └─ [模型路径兜底] (代码路径失败时)                  │     │
│   │   │  │         EChartsAgent → LLM 生成 echarts option          │     │
│   │   │  │                                                        │     │
│   │   │  │  输出:                                                    │     │
│   │   │  │    "\n\n```echarts\n{                                       │     │
│   │   │  │      \"title\": {\"text\": \"交易电量趋势\"},                 │     │
│   │   │  │      \"xAxis\": {\"data\": [\"06-01\",\"06-02\",...]},      │     │
│   │   │  │      \"series\": [{\"name\": \"华东分部\", \"data\": [...]}] │     │
│   │   │  │    }\n```"                                                  │     │
│   │   │  └────────────────────────────────────────────────────────┘     │
│   │   │                                                                 │
│   │   │  输出 (yield 给上游 query_router):                               │
│   │   │    {"data": "## 查询结果\n...", "type": "content"}    ← 格式化文本│
│   │   │    {"data": "", "type": "Placeholder_True"}        ← 图表占位    │
│   │   │    {"data": "\n\n```echarts\n{...}\n```", "type": "content"}    │
│   │   └─────────────────────────────────────────────────────────────────┘
│   │
│   └─── 步骤I: 结果截断 ───────────────────────────────────────────────────
│       ┌─────────────────────────────────────────────────────────────────┐
│       │  输入: display_data (格式化/聚合后的最终数据)                     │
│       │                                                                 │
│       │  · MAX_DISPLAY_ITEMS = 24   →  最多保留24条记录                  │
│       │  · MAX_V_FIELDS      = 8    →  最多保留8个时间点字段              │
│       │                                                                 │
│       │  本例: 15条记录, 不超过24 → 不做条数截断                           │
│       │  如超出: 截断部分缓存到 shared_cache (user_id + query_id), 前端可  │
│       │    GET /v1/query/full-data/{query_id} 获取完整数据                │
│       │                                                                 │
│       │  输出: 截断后的数据 (本例未截断, 原样返回)                          │
│       └─────────────────────────────────────────────────────────────────┘
│
│   ← workflow.execute_stream() 结束, AsyncGenerator 耗尽
│
├─ query_router.py:258  normalize_stream_chunk()      标准化每个 chunk
├─ query_router.py:270  isIntents(chunk)?            提取 intents (后续生成推荐链接)
├─ query_router.py:274  emit_answer_start()           "<think></think><answer>"
├─ query_router.py:280  format_output_result_sse()    SSE 帧格式化
│     └─ seeNow(event, content)  →  f"event: {event}\ndata: {json.dumps(content)}\n\n"
│
└─ query_router.py:361  return StreamingResponse(generate(), media_type="text/event-stream")

═══════════════════════════════════════════════════════════════════════════════
全局输出 (SSE 流, 客户端逐条收到):
═══════════════════════════════════════════════════════════════════════════════
  event: message | data: {"answer":"- <span>问题改写中...</span>"}
  event: message | data: {"answer":"- <span>意图识别中 (阶段一)...</span>"}
  event: message | data: {"answer":"- <span>问题拆解中...</span>"}
  event: message | data: {"answer":"- <span>意图识别中 (阶段二)...</span>"}
  event: message | data: {"answer":"- <span>参数提取中...</span>"}
  event: message | data: {"answer":"- <span>数据查询中...</span>"}
  event: message | data: {"answer":"- <span>数据处理中...</span>"}
  event: message | data: {"answer":"<think></think><answer>"}
  event: message | data: {"answer":"## 查询结果\n\n| 日期 | 交易电量(万kWh) |\n|------|--------------|\n| 2025-06-01 | 65432.1 |\n...","traceId":"abc-123"}
  event: message | data: {"echart_holder":true}                          ← 图表占位
  event: message | data: {"answer":"\n\n```echarts\n{\"title\":{\"text\":\"交易电量趋势\"},...}\n```"}
  event: message | data: {"answer":"</answer>"}
  event: message | data: {"answer":"\n\n您可以点击下方链接进行相关数据查询。\n"}
  event: message | data: {"answer":"{\"fileList\":[{...推荐文档链接...}]}"}
  event: done    | data: {"done":true,"conversation_id":"abc-123","query_id":"q-456"}
═══════════════════════════════════════════════════════════════════════════════
```

---

## 实战示例：一次完整请求的函数调用链（含每步输入/输出）

> **贯穿示例**："查询2025年6月1日到6月15日华东地区日前应急调度交易电量"

---

### 第 0 步：请求到达

```
┌─────────────────────────────────────────────────────────────────────┐
│  所在文件: app.py                                                    │
│  函数: log_requests() 中间件 (app.py:136)                             │
│                                                                     │
│  输入: HTTP Request (POST /v1/query, body=QueryRequest{...})          │
│                                                                     │
│  处理:                                                               │
│    trace_id = str(uuid4())        →  "abc-123-def"                  │
│    set_trace_id(trace_id)         →  注入 contextvars, 后续日志带此ID │
│    response = await call_next(request) → 进入 query_router            │
│                                                                     │
│  输出: response.headers["X-Process-Time"] = "0.523s"                 │
│        (trace_id 已注入上下文)                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 第 1 步：query_router 接收并分发

```
┌─────────────────────────────────────────────────────────────────────┐
│  所在文件: query_router.py:225                                       │
│  函数: async def query(req: QueryRequest)                            │
│                                                                     │
│  输入: QueryRequest {                                                │
│          "question":     "查询2025年6月1日到6月15日华东地区日前应急调度交易电量", │
│          "stream":       True,                                       │
│          "sessionId":    "abc-123",                                  │
│          "showThinkProcess": False,                                  │
│          "userInfo":     {"userId": "user_001"},                     │
│          ...                                                         │
│        }                                                             │
│                                                                     │
│  处理:                                                               │
│    user_id         = get_user_id(req)         → "user_001"           │
│    conversation_id = req.sessionId or uuid4() → "abc-123"            │
│    query_id        = str(uuid4())             → "q-456-789"          │
│    workflow = await workflow_factory.create_workflow("abc-123")       │
│              └─ WorkflowFactory (单例)                                │
│                 ├─ 共享模型实例 (进程级复用 HTTP 客户端)               │
│                 ├─ ProxyChatModel (model 在 GATEWAY_MODELS 列表)      │
│                 └─ ChatQwen       (否则直连 DashScope)                │
│                                                                     │
│  输出: StreamingResponse(text/event-stream)                          │
│        └─ 内部 generate() 异步生成器逐 chunk yield                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 第 2 步：问题改写

```
┌─────────────────────────────────────────────────────────────────────┐
│  所在文件: workflow_router.py:2448                                   │
│  函数: RewriteWorkflow.rewrite() → src/workflow/rewrite_workflow.py   │
│                                                                     │
│  输入: "查询2025年6月1日到6月15日华东地区日前应急调度交易电量"          │
│                                                                     │
│  处理:                                                               │
│    ├─ 模板匹配    (data/knowledge/rewrite_knowledge.json, 172个模板)   │
│    ├─ 时间标准化:  "6月1日"  →  "2025-06-01"                          │
│    ├─ 实体标准化:  "华东"    →  "华东分部"                              │
│    └─ 节假日解析:  holidays.json (补全缺失的年份信息)                   │
│                                                                     │
│  输出: "查询2025年6月1日到6月15日华东分部日前应急调度交易电量"          │
│        (实体名标准化为"华东分部"，时间格式统一为"YYYY-MM-DD")            │
└─────────────────────────────────────────────────────────────────────┘
```

### 第 3 步：意图识别 — 第一阶段 (28类业务域路由)

```
┌─────────────────────────────────────────────────────────────────────┐
│  所在文件: workflow_router.py:2458                                   │
│  函数: _recognize_intent() → IntentAgent.ainvoke()                   │
│        → src/agents/intent_agent.py:120                              │
│                                                                     │
│  输入: ┌────────────────────────────────────────────────────────┐   │
│        │ system_prompt: Unified_Intent_Recognition_Prompt        │   │
│        │   (src/agents/prompts.py, 定义28类业务域的描述和区分规则) │   │
│        │ user_message:  "查询2025年6月1日到6月15日华东分部         │   │
│        │                 日前应急调度交易电量"                     │   │
│        │ context:        {"thread_id": "abc-123"}                │   │
│        └────────────────────────────────────────────────────────┘   │
│                                                                     │
│  处理 (IntentAgent 内部):                                            │
│    create_agent(model, system_prompt, middleware, checkpointer)       │
│      ├─ @before_model: trim_messages_by_length()  裁剪历史消息       │
│      ├─ model.ainvoke([HumanMessage(content)])  → LLM 推理            │
│      │   └─ ChatQwen / ProxyChatModel → DashScope / 内网代理         │
│      └─ @after_model:  parse_json_block()  从 markdown 提取 JSON      │
│           └─ utils/output_parser.py                                  │
│                                                                     │
│  输出: ┌────────────────────────────────────────────────────────┐   │
│        │ {                                                        │   │
│        │   "intents":    ["省间应急调度交易信息（日前）查询"],       │   │
│        │   "time_range": {"start": "2025-06-01",                  │   │
│        │                   "end": "2025-06-15"},                  │   │
│        │   "entities":   ["华东分部"],                             │   │
│        │   "question":   "查询2025年6月1日到6月15日华东分部         │   │
│        │                  日前应急调度交易电量"                      │   │
│        │ }                                                        │   │
│        └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 第 4 步：问题拆分

```
┌─────────────────────────────────────────────────────────────────────┐
│  所在文件: workflow_router.py:2489                                   │
│  函数: _resolve_sub_items() → workflow_router.py:485                 │
│                                                                     │
│  输入: ┌────────────────────────────────────────────────────────┐   │
│        │ user_query:    "查询2025年6月1日到6月15日华东分部..."     │   │
│        │ intent_result: {intents: [...], time_range: {...}, ...} │   │
│        └────────────────────────────────────────────────────────┘   │
│                                                                     │
│  处理:                                                               │
│    ├─ 单意图 + 单实体  →  不拆分, 直接返回 [(user_query, intent)]      │
│    └─ 多意图 / 多主体  →  QuestionSplitWorkflow.split() 8条拆分规则   │
│                                                                     │
│  输出: ┌────────────────────────────────────────────────────────┐   │
│        │ sub_items = [                                             │   │
│        │   ("查询2025年6月1日到6月15日华东分部日前应急调度交易电量",   │   │
│        │    "省间应急调度交易信息（日前）查询")                       │   │
│        │ ]                                                         │   │
│        │ len(sub_items) == 1  →  走单子问题分支                      │   │
│        └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 第 5 步：意图识别 — 第二阶段 (域内专项识别)

```
┌─────────────────────────────────────────────────────────────────────┐
│  所在文件: workflow_router.py:2536                                   │
│  函数: _recognize_domain_intent("省间应急调度交易信息（日前）查询", ...) │
│        → _get_domain_intent_agent() → workflow_router.py:199         │
│                                                                     │
│  输入: ┌────────────────────────────────────────────────────────┐   │
│        │ user_query: "查询2025年6月1日到6月15日华东分部..."         │   │
│        │ intent:     "省间应急调度交易信息（日前）查询"              │   │
│        │ system_prompt: EmergencyDayhead_Intent_Recognition_Prompt│   │
│        │   (src/agents/prompts.py, 域内专项提示词)                 │   │
│        └────────────────────────────────────────────────────────┘   │
│                                                                     │
│  处理:                                                               │
│    prompt_map = {                                                    │
│      '省间应急调度交易信息（日前）查询':                                 │
│           EmergencyDayhead_Intent_Recognition_Prompt                 │
│    }                                                                 │
│    IntentAgent(intent_model, EmergencyDayhead_Intent_Recognition_Prompt)│
│      → domain_agent.ainvoke()                                       │
│                                                                     │
│  输出: ┌────────────────────────────────────────────────────────┐   │
│        │ {                                                        │   │
│        │   "intent": "省间应急调度交易信息（日前）查询",              │   │
│        │   "time_range": {"start": "2025-06-01",                  │   │
│        │                   "end": "2025-06-15"},                  │   │
│        │   "operations": [                                         │   │
│        │     {"operation":"filter",   "field":"sendrecv",         │   │
│        │                              "value":"送"},               │   │
│        │     {"operation":"group_by", "field":"businessTime",     │   │
│        │                              "date_granularity":"day"},   │   │
│        │     {"operation":"sum",      "field":"交易电量"}            │   │
│        │   ]                                                       │   │
│        │ }                                                         │   │
│        │ ★ 比第一阶段多了 operations, 这是后续聚合计算的关键输入 ★    │   │
│        └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 第 6 步：获取业务工作流实例

```
┌─────────────────────────────────────────────────────────────────────┐
│  所在文件: workflow_router.py:2555                                   │
│  函数: _get_workflow(intent) → workflow_router.py:414                │
│                                                                     │
│  输入: "省间应急调度交易信息（日前）查询"                               │
│                                                                     │
│  处理:                                                               │
│    WORKFLOW_REGISTRY[intent] → EmergencyDayaheadWorkflow             │
│    API_URL_MAPPING[intent]   → EMERGENCY_DAYAHEAD_API_URL            │
│    创建实例:                                                          │
│      EmergencyDayaheadWorkflow(                                      │
│        conversation_id="abc-123",                                    │
│        parameter_model=qwen3-max,                                    │
│        format_model=qwen3-max,                                       │
│        echarts_model=qwen3-max,                                      │
│        trend_analysis_model=qwen3-max,                               │
│        api_base_url="http://..."                                     │
│      )                                                               │
│                                                                     │
│  输出: EmergencyDayaheadWorkflow 实例 (继承自 BaseWorkflow)            │
│        内部已初始化: ParameterAgent, FormatAgent, AggregationAgent,   │
│                      EChartsAgent, TrendAnalysisAgent, PythonSandbox  │
└─────────────────────────────────────────────────────────────────────┘
```

### 第 7 步：BaseWorkflow.execute_stream — 参数提取

```
┌─────────────────────────────────────────────────────────────────────┐
│  所在文件: base_workflow.py:1794                                     │
│  函数: _extract_parameters() → base_workflow.py:2079                 │
│                                                                     │
│  输入: ┌────────────────────────────────────────────────────────┐   │
│        │ user_query:    "查询2025年6月1日到6月15日华东分部..."     │   │
│        │ intent_result: (阶段5的输出, 含 operations)               │   │
│        │ system_prompt: get_parameter_prompt()  ←  子类覆盖        │   │
│        │   EmergencyDayaheadWorkflow 自己的参数提取提示词           │   │
│        └────────────────────────────────────────────────────────┘   │
│                                                                     │
│  处理:                                                               │
│    ParameterAgent.ainvoke(user_query, intent_result, prompt_template) │
│      ├─ @before_model: trim_messages_by_length()                     │
│      ├─ LLM 提取: time_range / name_abbreviation / sendrecv / fields │
│      └─ @after_model: parse_json_block()                            │
│    _adjust_time_range_end(params)  →  end += 1天                    │
│    data_matching()  →  "华东" → "华东分部" (difflib 模糊匹配)         │
│                                                                     │
│  输出: ┌────────────────────────────────────────────────────────┐   │
│        │ {                                                        │   │
│        │   "time_range":        {"start":"2025-06-01",            │   │
│        │                         "end":"2025-06-16"}, ← 自动+1天  │   │
│        │   "name_abbreviation": "华东分部",        ← 标准化匹配     │   │
│        │   "sendrecv":          "送"                             │   │
│        │ }                                                        │   │
│        └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 第 8 步：参数验证

```
┌─────────────────────────────────────────────────────────────────────┐
│  所在文件: base_workflow.py:1800                                     │
│  函数: EmergencyDayaheadWorkflow.validate_params(params)              │
│                                                                     │
│  输入: params (步骤7输出)                                             │
│                                                                     │
│  处理:                                                               │
│    检查: time_range.start/end 存在? name_abbreviation 有效?          │
│          sendrecv 是否为 "送"/"受"?                                  │
│                                                                     │
│  输出: {"valid": true}  或  {"valid": false, "message": "请补充..."}  │
│        (本例校验通过)                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 第 9 步：API 调用

```
┌─────────────────────────────────────────────────────────────────────┐
│  所在文件: base_workflow.py:1811                                     │
│  函数: call_api() → EmergencyDayaheadWorkflow._call_api_impl(params) │
│                                                                     │
│  输入: params (步骤7输出, 校验通过)                                   │
│                                                                     │
│  处理:                                                               │
│    构造请求体 (映射到下游网关协议):                                    │
│      POST http://gateway/api/emergency-dayahead                     │
│      Body: {                                                         │
│        "startDate": "2025-06-01",                                    │
│        "endDate":   "2025-06-16",                                    │
│        "nameAbbreviation": "华东分部",                                 │
│        "sendrecv":  "送"                                             │
│      }                                                               │
│    通过 httpx.AsyncClient 共享连接池发送 (base_workflow.py:43)         │
│    超时: 180s                                                        │
│                                                                     │
│  输出: ┌────────────────────────────────────────────────────────┐   │
│        │ {                                                        │   │
│        │   "status": "success",                                    │   │
│        │   "data": [                                                │   │
│        │     {"businessTime":"2025-06-01T08:00:00Z",               │   │
│        │      "sendrecv":"送", "交易电量":12345.6, "交易电力":800.3, │   │
│        │      "v0015":200, "v0030":210, ..., "v2400":180},        │   │
│        │     {"businessTime":"2025-06-02T08:00:00Z", ...},        │   │
│        │     ... (15天数据, 每天一条记录, 每条含96个时间点字段)       │   │
│        │   ]                                                       │   │
│        │ }                                                         │   │
│        └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 第 10 步：空数据兜底 — 30天回溯

```
┌─────────────────────────────────────────────────────────────────────┐
│  所在文件: base_workflow.py:1822                                     │
│  函数: _retry_with_expanded_time_range_if_empty(...)                 │
│                                                                     │
│  输入: raw_data (步骤9输出) + params + user_query                     │
│                                                                     │
│  处理:                                                               │
│    _is_empty_result_data(raw_data)?                                  │
│      ├─ 有数据 → 跳过, 原样返回                                       │
│      └─ 空数据 → time_range.start -= 30天 → 重新调用步骤C API         │
│                                                                     │
│  输出: raw_data (有数据的版本, 格式同步骤9)                             │
│        (本例首次就有数据, 未触发回溯)                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 第 11 步：数据处理 + 聚合 (纯 Python, 不调 LLM)

```
┌─────────────────────────────────────────────────────────────────────┐
│  所在文件: base_workflow.py:1848-1860                                │
│  函数: sanitize_operations() → process_data() → aggregation_agent.ainvoke() │
│        → src/agents/aggregation_agent.py:698                        │
│                                                                     │
│  输入: ┌────────────────────────────────────────────────────────┐   │
│        │ raw_data:   (步骤9/10输出, 15条原始记录)                   │   │
│        │ operations: [filter(sendrecv=送), group_by(day),         │   │
│        │              sum(交易电量)]   ← 来自步骤5                  │   │
│        │ user_query: "查询2025年6月1日到6月15日华东分部..."         │   │
│        └────────────────────────────────────────────────────────┘   │
│                                                                     │
│  处理 (★ 纯 Python 执行, 不调LLM ★):                                │
│    ① sanitize_operations(): 丢弃无效操作 / 去重 / 纠正误判           │
│       输出: 清洗后的 operations (与输入可能不同)                       │
│                                                                     │
│    ② process_data(raw_data, params, ...): EmergencyDayaheadWorkflow  │
│       实现 → 字段重命名、类型转换、空值处理                            │
│       输出: 清洗后的数据列表 (15条)                                   │
│                                                                     │
│    ③ aggregation_agent.ainvoke(operations, data):                   │
│       执行顺序: filter → group_by → top_n/bottom_n → 聚合             │
│       ├─ raw_filter(data, field="sendrecv", operator="fuzzy", "送") │
│       │    → 过滤出送端记录                                           │
│       ├─ raw_group_by(filtered_data, field="businessTime", "day")   │
│       │    → 按天分组, 每天一组                                       │
│       └─ raw_sum_field(grouped_data, field="交易电量")                │
│           → 每组电量求和                                             │
│                                                                     │
│  输出: ┌────────────────────────────────────────────────────────┐   │
│        │ {                                                        │   │
│        │   "summary": [                                            │   │
│        │     {"operation":"sum","field":"交易电量","result":987654}│   │
│        │   ],                                                      │   │
│        │   "aggregated_data": [                                     │   │
│        │     {"businessTime":"2025-06-01","交易电量":65432.1},      │   │
│        │     {"businessTime":"2025-06-02","交易电量":67890.5},      │   │
│        │     ... (15行, 每天一行)                                    │   │
│        │   ]                                                       │   │
│        │ }                                                         │   │
│        └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 第 12 步：聚合结果决策 (100行的 if-else 链)

```
┌─────────────────────────────────────────────────────────────────────┐
│  所在文件: base_workflow.py:1871-1929                                │
│                                                                     │
│  输入: ┌────────────────────────────────────────────────────────┐   │
│        │ agg_result: {summary: [...], aggregated_data: [...]}    │   │
│        │ operations: [filter, group_by, sum]                     │   │
│        └────────────────────────────────────────────────────────┘   │
│                                                                     │
│  决策规则 (按优先级):                                                 │
│    ┌─────────────────────────────────────────────────────────────┐  │
│    │ ① group_by + count + 无 top_n/bottom_n       → summary     │  │
│    │ ② group_by + max/min  + 无 top_n/bottom_n    → 极值所在行   │  │
│    │ ③ top_n / bottom_n / group_by / 行间运算      → aggregated  │  │
│    │ ④ sum / average / max / min / count 单独使用  → summary     │  │
│    └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  本例匹配: operations 有 group_by + sum, 无 top_n/bottom_n            │
│          但 sum 是独立操作 → 命中规则④ → 使用 summary ?                │
│          实际: group_by 先命中规则③ → 最终使用 aggregated_data         │
│                                                                     │
│  输出: processed_data = aggregated_data                              │
│        [{"businessTime":"2025-06-01","交易电量":65432.1}, ...]       │
│        共 15 条记录                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 第 13 步：格式化决策

```
┌─────────────────────────────────────────────────────────────────────┐
│  所在文件: base_workflow.py:1987 ↔ base_workflow.py:754              │
│  函数: _should_use_format_summary(summary_data)                      │
│                                                                     │
│  输入: ┌────────────────────────────────────────────────────────┐   │
│        │ display_data: [15条聚合结果]                              │   │
│        │   → handle_summary_input() → summary_data (压缩版)       │   │
│        └────────────────────────────────────────────────────────┘   │
│                                                                     │
│  处理:                                                               │
│    json.dumps(summary_data) 序列化                                    │
│    len(serialized) ≤ 28000 ?                                         │
│      → True  : 调用 FormatAgent LLM 流式生成 Markdown                │
│      → False : 跳过格式化, 走 TrendAnalysisAgent 趋势分析路径         │
│                                                                     │
│  输入: 15条记录 ≈ 3KB  << 28000                                      │
│                                                                     │
│  输出: should_use_format_summary = True  (走 LLM 格式化)              │
│        canEcharts()              = True  (数据适合绘图)              │
└─────────────────────────────────────────────────────────────────────┘
```

### 第 14 步：格式化 + 图表 并发执行

```
┌─────────────────────────────────────────────────────────────────────┐
│  所在文件: base_workflow.py:1990-2066                                │
│                                                                     │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║  (并发任务1) 格式化流 — 与图表并发执行                          ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                     │
│  输入: ┌────────────────────────────────────────────────────────┐   │
│        │ user_query:   "查询2025年6月1日到6月15日华东分部..."     │   │
│        │ summary_data: [15条聚合数据]                             │   │
│        │ system_prompt: get_format_prompt() ← 子类覆盖             │   │
│        └────────────────────────────────────────────────────────┘   │
│                                                                     │
│  处理:                                                               │
│    _format_result_stream(user_query, summary_data)                   │
│      └─ FormatAgent.astream()                                       │
│           └─ LLM 流式生成 Markdown → filter_think_tags_async()       │
│                                                                     │
│  输出 (逐 chunk):                                                     │
│    "## 查询结果\n"                                                    │
│    "根据查询，2025年6月1日至6月15日华东分部日前应急调度..."              │
│    "| 日期 | 交易电量(万kWh) |\n"                                     │
│    "|------|--------------|\n"                                       │
│    "| 2025-06-01 | 65432.1 |\n"                                      │
│    "| 2025-06-02 | 67890.5 |\n"                                      │
│    ...                                                               │
│                                                                     │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║  (并发任务2) ECharts 图表                                       ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                     │
│  输入: user_query + intent + data_for_chart (15条, 含96时间点字段)    │
│                                                                     │
│  处理:                                                               │
│    create_echarts() → base_workflow.py:2492                          │
│      ├─ [代码路径优先]                                               │
│      │    build_dense_time_series_echarts(data)                      │
│      │    ├─ is_dense_time_series_records()   检测96点时间序列        │
│      │    ├─ sort_time_fields_batch()         v0015 → v2400 排序     │
│      │    ├─ has_non_zero_series()            过滤全零序列            │
│      │    └─ 构造 ECharts option (多系列折线图)                       │
│      │                                                                │
│      └─ [模型路径兜底] (代码路径失败时)                                │
│           EChartsAgent → LLM 生成 echarts option                      │
│                                                                     │
│  输出: "\n\n```echarts\n{  \"title\":{\"text\":\"交易电量趋势\"},     │
│           \"xAxis\":{\"data\":[\"06-01\"..]},                        │
│           \"series\":[{\"name\":\"华东分部\",...}] }\n```"            │
│                                                                     │
│                                                                     │
│  输出 (yield 给上游):                                                 │
│    {"data": "## 查询结果\n...",           "type": "content"}          │
│    {"data": "",                           "type": "Placeholder_True"}│
│    {"data": "\n\n```echarts\n{...}\n```", "type": "content"}          │
└─────────────────────────────────────────────────────────────────────┘
```

### 第 15 步：SSE 输出 + 结果截断

```
┌─────────────────────────────────────────────────────────────────────┐
│  所在文件: query_router.py:257-360                                   │
│  函数: generate() → normalize_stream_chunk() → format_output_result_sse() │
│                                                                     │
│  输入: BaseWorkflow 产出的原始 chunk 流                               │
│                                                                     │
│  处理:                                                               │
│    ① normalize_stream_chunk(raw_chunk)   标准化每个 chunk 格式         │
│    ② isIntents(chunk)?                  提取 intents (后续推荐链接)    │
│    ③ emit_answer_start()                 "<think></think><answer>"   │
│    ④ format_output_result_sse()          SSE 帧格式化                 │
│        └─ seeNow(event, content)                                      │
│            → f"event: {event}\ndata: {json.dumps(content)}\n\n"       │
│    ⑤ 结果截断:                                                        │
│        · MAX_DISPLAY_ITEMS = 24  →  最多保留24条记录                   │
│        · MAX_V_FIELDS      = 8   →  最多保留8个时间点字段               │
│        · 超出部分缓存到 shared_cache (user_id + query_id)               │
│    ⑥ 推荐链接: get_file_list_by_intents(captured_intents)             │
│    ⑦ yield {"data": {"done":true, ...}, "type": "done"}              │
│                                                                     │
│  输出: StreamingResponse(text/event-stream)                           │
│                                                                     │
│  客户端收到的完整 SSE 流:                                              │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ event: message | data: {"answer":"<span>问题改写中...</span>"}    ││
│  │ event: message | data: {"answer":"<span>意图识别中...</span>"}    ││
│  │ event: message | data: {"answer":"<span>问题拆解中...</span>"}    ││
│  │ event: message | data: {"answer":"<span>意图识别中...</span>"}    ││
│  │ event: message | data: {"answer":"<span>参数提取中...</span>"}    ││
│  │ event: message | data: {"answer":"<span>数据查询中...</span>"}    ││
│  │ event: message | data: {"answer":"<span>数据处理中...</span>"}    ││
│  │ event: message | data: {"answer":"<think></think><answer>"}      ││
│  │ event: message | data: {"answer":"## 查询结果\n\n..."}           ││
│  │ event: message | data: {"echart_holder":true}                    ││
│  │ event: message | data: {"answer":"\n\n```echarts\n{...}\n```"}  ││
│  │ event: message | data: {"answer":"</answer>"}                    ││
│  │ event: message | data: {"answer":"{\"fileList\":[...]}"}        ││
│  │ event: done    | data: {"done":true,"conversation_id":"abc-123"}││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 分阶段学习路线

### 第一阶段：跑起来 + 理解入口（1～2 天）

| 序号 | 任务 | 关键文件 | 要点 |
|------|------|----------|------|
| 1.1 | 配置环境、启动服务 | `.env`, `app.py`, `requirements.txt` | 理解 `load_env()` 如何加载配置；理解 `run()` vs 开发模式的差异 |
| 1.2 | 发一个请求，跟踪完整链路 | `app.py:225` → `query_router.py:225` | 用 `POST /v1/query` 发一个真实请求，在日志中观察各阶段耗时 |
| 1.3 | 理解 SSE 流式响应格式 | `query_router.py:116-150` | `seeNow()` 如何构造 SSE 帧；`format_output_result_sse()` 如何分发不同 type（content/metadata/done/error） |
| 1.4 | 理解 Pydantic 数据模型 | `src/api/schemas/schemas.py` | `QueryRequest`（输入）/ `ReturnQuery`（输出）/ `SSEPayload` |

**验证**：能自己构造一个 curl 请求，看懂每一步的 SSE 事件流。

---

### 第二阶段：掌握核心编排引擎（3～4 天）

| 序号 | 任务 | 关键文件 | 要点 |
|------|------|----------|------|
| 2.1 | 读懂 WorkflowRouter 的 execute_stream | `src/workflow/workflow_router.py:2439-2518` | 7 个阶段的顺序编排；yield messageLabel 做前端状态提示 |
| 2.2 | 理解两阶段意图识别 | `workflow_router.py:225-358` | 第一阶段 28 域路由 → 第二阶段域内专项；`_parse_intent_result()` JSON 解析 + 兜底逻辑 |
| 2.3 | 理解问题改写 | `src/workflow/rewrite_workflow.py` | 172 个模板的 slot 填充机制；时间标准化；实体名标准化 |
| 2.4 | 理解问题拆分 | `src/workflow/question_split_workflow.py` | 8 条拆分规则；多意图/多主体拆解 |
| 2.5 | 理解多子问题并发 | `workflow_router.py:_execute_multi_questions_stream` | 共享数据模式（同意图子问题共享一次 API 调用）；DenseConnect 冲突检测 |

**验证**：能口述 "华东地区昨天日前交易电量" 这个请求在整个 WorkflowRouter 里经历了哪些阶段。

---

### 第三阶段：深入 BaseWorkflow 模板方法（3～5 天）

| 序号 | 任务 | 关键文件 | 要点 |
|------|------|----------|------|
| 3.1 | 读懂 execute_stream 全流程 | `base_workflow.py:1784-2077` | 7 个阶段（参数提取→验证→API调用→处理→聚合→格式化→图表）的顺序和异常处理 |
| 3.2 | 理解聚合结果决策链 | `base_workflow.py:1871-1929` | ~100 行的 if-else：什么情况下用 summary，什么情况下用 aggregated_data。这是最容易出 bug 的地方 |
| 3.3 | 理解操作清洗 | `base_workflow.py:1112` `sanitize_operations()` | 丢弃无效操作、去重、时间点字段误判纠正 |
| 3.4 | 理解格式化阈值 | `base_workflow.py:754` `_should_use_format_summary()` | `FORMAT_SUMMARY_MAX_DATA_CHARS`（28000）阈值如何决定走 FormatAgent 还是 TrendAnalysisAgent |
| 3.5 | 理解显示截断 | `base_workflow.py` 中 `MAX_DISPLAY_ITEMS=24` + `MAX_V_FIELDS=8` | 超出部分如何缓存、如何通知前端 |
| 3.6 | 读懂一个具体子类 | `src/workflow/emergency_dayahead_workflow.py` | `get_parameter_prompt()`, `get_format_prompt()`, `validate_params()`, `_call_api_impl()`, `process_data()` 五个必须实现的方法 |

**验证**：能画出 BaseWorkflow.execute_stream 的流程图，标注每个步骤的输入/输出。

---

### 第四阶段：理解 Agent 层（2～3 天）

| 序号 | 任务 | 关键文件 | 要点 |
|------|------|----------|------|
| 4.1 | IntentAgent — LangGraph Agent 模式 | `src/agents/intent_agent.py` | `create_agent()` + `@before_model` / `@after_model` middleware；InMemorySaver checkpointer；`ainvoke` vs `astream` |
| 4.2 | ParameterAgent — 参数提取 | `src/agents/parameter_agent.py` | 提示词模板由 BaseWorkflow 子类注入；JSON 解析 + 标准化匹配 |
| 4.3 | AggregationAgent — 纯 Python 聚合引擎 | `src/agents/aggregation_agent.py` | 不调 LLM！18 种操作的 dispatch 逻辑；`filter → group_by → top_n/bottom_n → 聚合` 执行顺序；分组均值回填；日期粒度自动检测 |
| 4.4 | FormatAgent — LLM 格式化 | `src/agents/format_agent.py` | 流式生成 Markdown；提示词模板注入 |
| 4.5 | EChartsAgent + TrendAnalysisAgent | `echarts_agent.py`, `trend_analysis_agent.py` | ECharts 模型路径兜底；趋势分析两阶段（代码生成 + 总结） |

**验证**：能说清楚每个 Agent 的输入/输出类型，以及哪个 Agent 调 LLM、哪个不调。

---

### 第五阶段：理解工具层（2～3 天）

| 序号 | 任务 | 关键文件 | 要点 |
|------|------|----------|------|
| 5.1 | aggregation_tools — 18 个纯函数 | `src/utils/aggregation_tools.py` | 每个函数（`raw_filter`, `raw_sum_field`, `raw_group_by`…）的输入/输出/边界条件；Decimal 精度处理 |
| 5.2 | echarts_utils — 代码路径优先 | `src/utils/echarts_utils.py` | `build_dense_time_series_echarts()` 生成 96 点时间序列折线图；`sort_time_fields_batch()`；`has_non_zero_series()` |
| 5.3 | model_proxy — 网关代理模型 | `src/utils/model_proxy.py` | `ProxyChatModel` 实现 LangChain 兼容接口，SSE 流式解析 |
| 5.4 | python_sandbox — 安全代码执行 | `src/utils/python_sandbox.py` | AST 剥离 import + 白名单 builtins + 超时控制 |
| 5.5 | data_standard + fuzzy_match | `data_standard.py`, `fuzzy_match.py` | `difflib` 实体名标准化；多策略模糊匹配（精确/前缀/包含/数字模糊） |
| 5.6 | 其他工具 | `filter_think_tags.py`, `output_parser.py`, `summary_util.py`, `logging_setup.py`, `get_file_list_by_intents.py` | `<think>` 标签过滤（4 种实现）；JSON 块解析；数据摘要压缩；trace_id 日志追踪 |

**验证**：能独立调用 `raw_group_by` + `raw_sum_field` 完成一次聚合。

---

### 第六阶段：理解服务层 + 配置层（1～2 天）

| 序号 | 任务 | 关键文件 | 要点 |
|------|------|----------|------|
| 6.1 | AgentFactory + WorkflowFactory 单例 | `src/service/agent_factory.py`, `workflow_factory.py` | 进程级共享模型实例（避免每次创建 HTTP 客户端）；内网模型开关 |
| 6.2 | SessionManager | `src/service/session_manager.py` | InMemorySaver 会话管理；1000 会话上限；1 小时过期 |
| 6.3 | Settings 配置中心 | `src/config.py` | 所有环境变量的集中管理；模型选择、超时、并发控制 |
| 6.4 | JSON 配置文件 | `data/env_variables/` | `time_point_mapping.json`（96 时间点）、`operations_config.json`、`echarts_config.json`、`emergency_dispatch_fields_config.json` |
| 6.5 | 缓存管理 | `shared_cache.py` | async Lock 保护；5 分钟过期；按 user_id + query_id 隔离 |

**验证**：能说清楚 `INNER_MODEL_ENABLE=TRUE` 时模型调用路径与默认路径的区别。

---

### 第七阶段：实战 — 添加新业务意图（3～5 天）

**任务**：按 CLAUDE.md 指引，新增一个完整业务意图。

| 序号 | 任务 | 文件 |
|------|------|------|
| 7.1 | 在 `WORKFLOW_REGISTRY` + `API_URL_MAPPING` 中注册 | `workflow_router.py` |
| 7.2 | 创建 `XxxWorkflow` 子类，实现 5 个抽象方法 | `src/workflow/xxx_workflow.py` |
| 7.3 | 添加参数提取 + 格式化提示词模板 | `src/agents/prompts.py` |
| 7.4 | 如需域内专项识别，注册二级 prompt | `workflow_router.py` 的 `_get_domain_intent_agent()` |
| 7.5 | 在 `.env` 配置对应 API URL | `.env` |
| 7.6 | 端到端测试 | curl / Swagger |

---

## 关键概念速查表

| 概念 | 位置 | 一句话解释 |
|------|------|----------|
| SSE 流式响应 | `query_router.py` | `text/event-stream` 格式，前端逐条接收 |
| 两阶段意图识别 | `workflow_router.py:225-358` | 先路由到 28 类业务域，再域内专项识别 |
| 问题拆分 | `question_split_workflow.py` | 多意图/多主体自动拆成独立子问题 |
| 共享数据模式 | `workflow_router.py` | 同意图多子问题共享一次 API 调用 |
| 聚合决策链 | `base_workflow.py:1871-1929` | 决定展示 summary 还是 aggregated_data |
| 格式化阈值 | `base_workflow.py:754` | 28000 字符分界线：小数据走 LLM 格式化，大数据走趋势分析 |
| 代码路径优先 | `echarts_utils.py` | ECharts 先生成代码，失败才用 LLM 兜底 |
| 30天回溯 | `base_workflow.py` | API 返回空时自动扩展时间范围重试 |
| 分组均值回填 | `aggregation_agent.py` | filter 引用 group_by 结果时的特殊处理 |
| ProxyChatModel | `model_proxy.py` | 通过内网代理调用 LLM，兼容 LangChain 接口 |
| InnerModelAgent | `inner_model_agent.py` | 通过 aiohttp 直连内网模型服务 |

---

## 学习建议

1. **必装工具**：VSCode + Python 插件 + 设置断点调试。在 `execute_stream` 各阶段打断点，单步跟踪一次完整请求。
2. **先宏观后微观**：先理解请求从入口到出口经过了哪些阶段（7 个阶段），再深入每个阶段的实现细节。
3. **重点关注 base_workflow.py:1871-1929**：这是聚合结果决策逻辑，最容易出错，也最能体现业务复杂度。
4. **读懂提示词**：`src/agents/prompts.py` 虽然长（814 行），但它是连接用户自然语言和结构化参数的桥梁，理解它才能理解系统为什么能"智能"。
5. **动手实践**：最后阶段一定要亲手添加一个新意图，这是检验理解深度的最好方式。
