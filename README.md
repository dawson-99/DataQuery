# 智能数据查询系统

省间电力数据的自然语言查询服务。用户用中文自然语言提问，系统自动识别意图、提取参数、调用下游电力数据 API，经聚合计算和格式化后，以 SSE 流式返回结果（含 Markdown 文本 + ECharts 图表）。

## 快速开始

### 环境要求

- Python 3.11+
- conda（推荐）或 virtualenv

### 安装与运行

```bash
# 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 开发模式启动（默认端口 6066）
python app.py

# 或生产模式启动
python -c "from app import run; run()"
```

启动后访问：
- API 根路径：http://localhost:6066/
- Swagger 文档：http://localhost:6066/docs
- ReDoc 文档：http://localhost:6066/redoc

### 配置

所有配置通过 `.env` 文件管理（由 `src/config.py` 自动加载）。关键环境变量：

| 变量 | 说明 |
|------|------|
| `SERVER_HOST` / `SERVER_PORT` | 服务监听地址和端口 |
| `DASHSCOPE_API_KEY` | 阿里 DashScope API 密钥（直连模式） |
| `GATEWAY_BASE_URL` | 模型请求中转服务地址 |
| `GATEWAY_MODELS` | 走中转代理的模型名列表（逗号分隔） |
| `COMMON_API_URL` | 统一网关入口，所有业务 API 通过此地址透传 |
| `LLM_MODEL` / `INTENT_MODEL` / `FORMAT_MODEL` 等 | 各阶段使用的 LLM 模型名 |
| `INNER_MODEL_ENABLE` | 设为 `TRUE` 时，图表和格式化 Agent 切到内网模型 |
| `FORMAT_SUMMARY_MAX_DATA_CHARS` | 格式化阶段字符数阈值（默认 28000），超出则跳过格式化 |
| `MAX_SUBQUESTION_CONCURRENCY` | 子问题并发数 |

## 核心接口

### `POST /v1/query` — 智能查询

请求体：

```json
{
    "question": "2025年3月15日冀北的日前现货出清电量是多少",
    "aiModel": "qwen3-max",
    "sessionId": "uuid-session-id",
    "showThinkProcess": false,
    "stream": true,
    "userInfo": { "userId": "user123" }
}
```

- `stream=true`：SSE 流式响应，实时返回思考过程、数据查询状态、格式化结果和 ECharts 图表
- `stream=false`：一次性返回 JSON 结果

### 缓存管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/v1/query/full-data/{query_id}?user=xxx` | 获取被截断的完整数据 |
| DELETE | `/v1/query/cache/{query_id}?user=xxx` | 清理单条查询缓存 |
| DELETE | `/v1/query/cache/user/{user}` | 清理用户所有缓存 |
| GET | `/v1/query/cache/stats` | 查看缓存统计 |

## 架构概览

```
POST /v1/query
  └── query_router.py          ← 接收请求，分发流式/非流式
      └── WorkflowRouter        ← 核心编排器
          ├── RewriteWorkflow   ← 问题改写
          ├── IntentAgent × 2   ← 两阶段意图识别（业务域路由 → 域内专项识别）
          ├── QuestionSplitWorkflow ← 问题拆分（多意图/多主体场景）
          └── BaseWorkflow 子类 ← 业务工作流执行
              ├── ParameterAgent     ← 参数提取
              ├── call_api           ← API 调用（统一网关透传）
              ├── AggregationAgent   ← 纯 Python 聚合（filter/sum/group_by/max/min 等）
              ├── FormatAgent        ← 流式 Markdown 格式化输出
              └── EChartsAgent       ← ECharts 图表生成（代码优先 → 模型兜底）
```

## Agent 层详解

`src/agents/` 目录包含 **7 个 Agent 类** + **2 个辅助文件**，覆盖从意图识别到结果输出的完整流水线：

### 1. `intent_agent.py` — IntentAgent（意图识别 Agent）

识别用户自然语言查询背后的业务意图，判断属于哪个业务场景（如交易结果查询、应急调度查询等）。

- 接收 LLM 模型 + 业务域提示词模板，通过 LangChain `create_agent` 构建 Agent
- `@before_model` 中间件：消息历史裁剪，防止超出 context 限制
- `@after_model` 中间件：统一解析模型输出的 JSON 块
- 对应流水线阶段：**两阶段意图识别**（先路由到 28 类业务域 → 再域内专项识别）

### 2. `parameter_agent.py` — ParameterAgent（参数提取 Agent）

从用户问题中提取业务查询所需的结构化参数（时间范围、主体名称、数据类型等）。

- 与 IntentAgent 架构类似，使用 LangChain Agent + before/after 中间件
- 输入的 `prompt_template` 由各业务 Workflow 子类的 `get_parameter_prompt()` 提供
- 若未提取到时间范围，会触发**时间回溯 30 天兜底重试**机制

### 3. `aggregation_agent.py` — AggregationAgent（聚合计算 Agent）

**纯 Python 实现的数据聚合引擎，不调用 LLM**。速度快、结果准确、不受上下文长度限制。

| 类别 | 支持的操作 |
|------|------------|
| 过滤/排序/分组 | `filter`、`top_n`、`bottom_n`、`group_by` |
| 汇总统计 | `sum`、`average`、`max`、`min`、`count`、`count_if` |
| 算术运算 | `subtract`、`multiply`、`divide` |
| 环比/同比 | `mom_change`、`yoy_change` |
| 加权/行级 | `weighted_average`、`row_sum`、`count_records` |

**执行顺序**：`filter → group_by → top_n/bottom_n → 聚合运算`，内含“分组均值回填”逻辑用于 filter 引用 group_by 结果的复杂场景。此外会自动识别用户问题中的日期粒度关键词（“每天”“逐月”“按周”等），修正分组粒度。

### 4. `format_agent.py` — FormatAgent（结果格式化 Agent）

将查询结果转换为用户易读的 **Markdown 格式**文本，通过 SSE 流式输出。

- 通过 `_build_prompt()` 将 `user_query` 和 `data`（JSON）填入业务提示词模板
- 支持 `invoke()`（同步非流式）和 `astream()`（异步流式）两种模式
- 格式化阈值：数据序列化后超过 `FORMAT_SUMMARY_MAX_DATA_CHARS`（默认 28000）时跳过格式化，改用趋势分析

### 5. `echarts_agent.py` — EChartsAgent（图表生成 Agent）

根据查询数据生成 **ECharts 图表配置（option）**，用于前端渲染可视化图表。

- 直接调用 LLM（不走 LangChain Agent），使用 `SystemMessage + HumanMessage` 模式
- 输入格式：`USER_QUERY=...` + `QUERY_DATA=...`
- 作为**模型路径兜底**；正常情况下优先走代码路径（`echarts_utils.py` 的 `build_dense_time_series_echarts`）

### 6. `inner_model_agent.py` — InnerModelAgent（内网模型 Agent）

通过**内网 HTTP 流式 API** 调用 LLM 模型，作为阿里 DashScope 直连的替代方案。

- 基于 `aiohttp` 实现异步流式 SSE 调用
- 自动解析 `data: {...}` 格式的 SSE 事件，提取 `delta.content`
- 内置 `filter_think_tags` 过滤模型思考标签（`...`）
- 当 `.env` 中 `INNER_MODEL_ENABLE=TRUE` 或模型在 `GATEWAY_MODELS` 列表中时切换至此 Agent
- 使用共享的 `aiohttp.ClientSession`（由 `agent_factory.py` 单例工厂管理）

### 7. `trend_analysis_agent.py` — TrendAnalysisAgent（趋势分析 Agent）

当数据量超过格式化阈值时，**跳过逐条格式化，改为趋势分析**。

- **阶段 1（`generate_code`）**：根据数据概要和用户问题，让 LLM 生成可执行的 Python 分析代码，由 `python_sandbox.py` 安全执行
- **阶段 2（`summarize` / `summarize_stream`）**：基于沙箱统计结果，生成简要分析总结

### 辅助文件

- **`prompts.py`**（814 行）：所有 Agent 的提示词模板集，包括统一意图识别、各业务域专项识别、参数提取、格式化、ECharts 生成、趋势分析等
- **`utils.py`**：提供 `trim_messages_by_length()` 工具函数，根据 `MAX_CONTEXT_CHARS` 限制裁剪消息历史，优先保留系统提示词和最新用户消息

### 流水线全景

```
用户问题
  │
  ├─ 1. IntentAgent（两阶段意图识别：业务域路由 → 域内专项）
  ├─ 2. ParameterAgent（提取时间/主体/数据类别等参数）
  ├─ 3. API 调用（通过 COMMON_API_URL 统一网关透传下游数据源）
  ├─ 4. AggregationAgent（纯 Python 聚合：filter → group_by → 统计计算）
  ├─ 5. FormatAgent（Markdown 格式化，SSE 流式输出）
  └─ 6. EChartsAgent（图表配置生成，代码路径优先 → 模型兜底）

  超阈值分支：FormatAgent 跳过 → TrendAnalysisAgent 趋势分析替代
  内网模型分支：IntentAgent / FormatAgent / EChartsAgent → InnerModelAgent 运输
```

## 项目结构

```
DataQuery/
├── app.py                    # FastAPI 应用入口
├── requirements.txt          # Python 依赖
├── .env                      # 环境变量配置（含密钥，勿提交）
├── data/
│   ├── env_variables/        # 配置文件（时间点映射、图表配置、字段定义等）
│   └── knowledge/            # 知识库（节假日、改写规则等）
└── src/
    ├── config.py             # 配置加载与 Settings 单例
    ├── agents/               # LLM Agent 层
    │   ├── intent_agent.py   # 意图识别 Agent（LangGraph + middleware）
    │   ├── parameter_agent.py# 参数提取 Agent
    │   ├── aggregation_agent.py # 聚合引擎（纯 Python，不依赖 LLM）
    │   ├── format_agent.py   # 结果格式化 Agent
    │   ├── echarts_agent.py  # ECharts 图表生成 Agent
    │   ├── trend_analysis_agent.py # 趋势分析 Agent（Python 沙箱）
    │   ├── prompts.py        # 所有提示词模板
    │   └── utils.py          # Agent 工具函数
    ├── api/
    │   ├── routers/
    │   │   ├── query_router.py   # 查询路由（SSE 流式响应 + 缓存管理）
    │   │   └── shared_cache.py   # 缓存实现
    │   └── schemas/
    │       └── schemas.py        # Pydantic 请求/响应模型
    ├── service/
    │   ├── agent_factory.py      # Agent 工厂（单例，进程级共享）
    │   ├── workflow_factory.py   # 工作流工厂（懒加载模型实例）
    │   └── session_manager.py    # 会话注册表（过期清理、容量限制）
    ├── workflow/
    │   ├── workflow_router.py    # 核心编排器（意图路由、问题拆分、共享数据模式）
    │   ├── base_workflow.py      # 业务工作流抽象基类
    │   ├── emergency_dayahead_workflow.py # 应急调度交易查询
    │   ├── clarification_workflow.py      # 问题澄清
    │   ├── rewrite_workflow.py   # 问题改写
    │   └── question_split_workflow.py     # 问题拆分
    └── utils/
        ├── aggregation_tools.py  # 聚合底层函数（raw_filter、raw_sum 等）
        ├── echarts_utils.py      # ECharts 工具（代码路径图表生成）
        ├── model_proxy.py        # HTTP 代理模型（兼容 LangChain 接口）
        ├── data_standard.py      # 数据标准化
        ├── fuzzy_match.py        # 模糊匹配
        ├── python_sandbox.py     # Python 沙箱（趋势分析用）
        ├── summary_util.py       # 摘要数据处理
        └── logging_setup.py      # 日志配置
```

## 添加新业务意图

1. 在 `src/workflow/workflow_router.py` 的 `WORKFLOW_REGISTRY` 和 `API_URL_MAPPING` 中注册
2. 创建 `src/workflow/xxx_workflow.py`，继承 `BaseWorkflow`，实现：
   - `get_parameter_prompt()` — 参数提取提示词
   - `get_format_prompt()` — 结果格式化提示词
   - `validate_params()` — 参数验证规则
   - `_call_api_impl()` — 具体 API 调用逻辑
   - `process_data()`（可选）— 自定义数据处理
3. 在 `src/agents/prompts.py` 中添加对应的提示词模板
4. 如需域内专项意图识别，在 `_get_domain_intent_agent()` 中注册二级 prompt
5. 在 `.env` 中配置对应的业务 API URL
