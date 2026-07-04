# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

智能数据查询系统 — 省间电力数据的自然语言查询服务。用户用中文自然语言提问，系统自动识别意图、提取参数、调用下游电力数据 API，经聚合/格式化后以 SSE 流式返回结果（含 Markdown 文本 + ECharts 图表）。

当前仅实现了 **省间应急调度交易信息（日前）查询** 这一个业务意图；意图识别 Prompt 虽然按多业务域设计，但 `src/workflow/workflow_router.py` 的 `WORKFLOW_REGISTRY` 中只注册了该意图与澄清工作流。

## 技术栈

- **Python**: 3.11+
- **Web**: FastAPI + Uvicorn（SSE 流式响应）
- **LLM**: 阿里 DashScope（qwen3-max / qwen3-32b）+ 内网中转代理（可选）
- **Agent 框架**: LangChain / LangGraph（`create_agent` + `@before_model`/`@after_model` 中间件），含 `langchain-qwq`（通义千问 LangChain 集成）
- **数据计算**: pandas + 纯 Python 聚合（不依赖 LLM）
- **HTTP**: httpx（共享连接池，对外 API 调用）+ aiohttp（内网模型流式调用）

> **注意**: `httpx` 在代码中使用但**未列入 `requirements.txt`**，它作为 `langchain-core` 的传递依赖被安装。如遇导入错误，手动 `pip install httpx`。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 开发模式启动（热重载，单进程；默认端口以 .env 的 SERVER_PORT 为准，项目使用 6066）
python app.py

# 生产模式启动
python -c "from app import run; run()"

# 或直接 uvicorn
uvicorn app:app --host 0.0.0.0 --port 6066 --reload
```

### 测试

项目**目前没有自动化测试**，但 `src/utils/aggregation_tools.py` 等纯函数非常适合优先补单元测试。补充测试后（建议放在 `tests/` 目录）：

```bash
# 运行全部测试
pytest

# 运行单个测试文件
pytest tests/test_aggregation_tools.py

# 运行单个测试函数
pytest tests/test_aggregation_tools.py::test_raw_filter
```

### 手动端到端验证

```bash
curl -X POST http://localhost:6066/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "2025年3月15日冀北的日前现货出清电量是多少",
    "aiModel": "qwen3-max",
    "sessionId": "uuid-session-id",
    "showThinkProcess": false,
    "stream": true,
    "userInfo": { "userId": "user123" }
  }'
```

启动后访问：
- Swagger UI：http://localhost:6066/docs
- ReDoc：http://localhost:6066/redoc

环境变量全部在 `.env` 文件中管理，由 `src/config.py` 的 `load_env()` 自动加载。**注意：`.env` 包含 API 密钥，切勿提交到版本控制。**

**学习资源**：`LEARNING_ROADMAP.md` 以一次完整请求为例，详细跟踪了从 HTTP 入站到 SSE 出站的完整函数调用链（含每步输入/输出），是理解系统的首选参考文档。

## 核心架构

请求处理流水线：

```
POST /v1/query
  → query_router.py: 接收 QueryRequest，分流式/非流式
  → WorkflowRouter.execute_stream(): 核心编排
      ├── 问题改写（RewriteWorkflow）
      ├── 意图识别第1阶段：业务域路由（IntentAgent + Unified_Intent_Recognition_Prompt）
      ├── 意图识别第2阶段：域内专项识别（EmergencyDispatch_Intent_Recognition_Prompt）
      ├── 问题拆分（QuestionSplitWorkflow，仅多意图/多主体场景）
      └── 对每个意图/子问题：
          └── BaseWorkflow.execute_stream()
              ├── 参数提取（ParameterAgent + 子类 get_parameter_prompt()）
              ├── 参数验证（子类 validate_params()）
              ├── API 调用（统一网关 COMMON_API_URL 透传 → 子类 _call_api_impl()）
              ├── 数据处理（子类 process_data() + AggregationAgent 纯 Python 聚合）
              ├── 结果格式化（FormatAgent 流式输出 Markdown）
              └── ECharts 图表生成（代码路径优先 → 模型路径兜底）
```

### 应用生命周期（app.py）

`app.py` 使用 FastAPI 的 `lifespan` 管理启动/关闭：

- 启动时创建后台异步日志写入协程（`log_writer`），通过 `asyncio.Queue` 非阻塞写日志，保留 `trace_id` 上下文。
- 启动缓存清理任务。
- 关闭时依次：
  1. 调用 `agent_factory.shutdown()` 关闭 `aiohttp.ClientSession`；
  2. 停止缓存清理任务；
  3. 发送停止信号并等待日志写入协程结束。

HTTP 层包含 CORS 中间件、自定义异步请求日志中间件（记录方法/路径/状态码/耗时）以及全局异常处理器（`DEBUG=false` 时不对外暴露详细错误）。

## 关键模块

### 编排层

#### `src/workflow/workflow_router.py`（核心编排器，3079 行）
最重要的文件。负责：
- **两阶段意图识别**：先路由到业务域，再执行域内专项识别
- **问题拆分**：多意图 / 多主体（"A和B的..."）场景自动拆分子问题
- **共享数据模式**：同一意图下的多个子问题共享一次 API 调用，按 `targetType` / `nameAbbreviation` / `caseId` 过滤分发
- **ECharts 统一生成**：代码路径（`echarts_utils.py` 的 `build_dense_time_series_echarts`）优先，模型路径兜底
- **时间范围兜底**：首次查询为空时自动回溯 30 天重试

工作流注册表 `WORKFLOW_REGISTRY` 和 API 映射表 `API_URL_MAPPING` 均位于此文件，新增业务意图需在此注册。**当前注册表中只有 `省间应急调度交易信息（日前）查询` 与 `待澄清` 两个工作流。**

#### `src/workflow/base_workflow.py`（抽象基类，2541 行）
定义所有业务工作流的模板方法：
- **聚合结果决策逻辑**（约 100 行的 if-else 链）：根据操作类型组合（聚合/逐行/分组/排序）决定最终展示 `summary` 还是 `aggregated_data`，这是最容易出错的地方
- **操作清洗**（`sanitize_operations()`）：丢弃无效操作、去重、时间点字段误判 filter 纠正为 time_points
- **格式化阈值**（`_should_use_format_summary()`）：当数据序列化后 > `FORMAT_SUMMARY_MAX_DATA_CHARS`（默认 28000）时跳过格式化，改为趋势分析
- **显示截断**：最多展示 24 条记录 + 8 个时间点字段

#### 其他工作流文件
- `src/workflow/rewrite_workflow.py` — 问题改写（172 个模板 + 时间/实体标准化）
- `src/workflow/question_split_workflow.py` — 问题拆分（多意图/多主体）
- `src/workflow/emergency_dayahead_workflow.py` — 应急调度交易查询（BaseWorkflow 的参考实现）
- `src/workflow/clarification_workflow.py` — 问题澄清（参数不足时追问用户）

### Agent 层（`src/agents/`）

#### `intent_agent.py` — IntentAgent（意图识别）
基于 LangGraph `create_agent` + `@before_model`/`@after_model` 中间件。两阶段使用：第一阶段用 `Unified_Intent_Recognition_Prompt` 做业务域路由，第二阶段用域内专项 prompt 提取 operations。

#### `parameter_agent.py` — ParameterAgent（参数提取）
从用户问题中提取结构化参数（时间范围、主体名称、数据类型等）。提示词模板由各 BaseWorkflow 子类的 `get_parameter_prompt()` 注入。

#### `aggregation_agent.py` — AggregationAgent（聚合引擎，720 行）
**纯 Python 实现，不调用 LLM**。支持 18 种操作：`filter` / `top_n` / `bottom_n` / `group_by` / `sum` / `average` / `max` / `min` / `count` / `count_if` / `subtract` / `multiply` / `divide` / `mom_change` / `yoy_change` / `weighted_average` / `row_sum` / `count_records`

执行顺序：filter → group_by → top_n/bottom_n → 聚合。内含「分组均值回填」逻辑用于 filter 引用 group_by 结果的场景。

#### `format_agent.py` — FormatAgent（Markdown 格式化）
将数据转为用户可读的 Markdown，支持 `invoke()`（同步）和 `astream()`（SSE 流式输出）。

#### `echarts_agent.py` — EChartsAgent（图表生成）
LLM 直接调用（不走 LangChain Agent），使用 `SystemMessage + HumanMessage` 模式。作为图表生成的**模型路径兜底**；正常情况下优先走代码路径。

#### `inner_model_agent.py` — InnerModelAgent（内网模型代理）
通过 `aiohttp` 直连内网模型服务的异步 SSE 流式调用。内置 `filter_think_tags` 过滤模型思考标签。

#### `trend_analysis_agent.py` — TrendAnalysisAgent（趋势分析）
当数据量超过格式化阈值时的替代路径。两阶段：① LLM 生成 Python 分析代码 → `python_sandbox.py` 安全执行；② 基于沙箱统计结果生成简要总结。

#### `prompts.py`（814 行）
所有 Agent 的提示词模板集，包括统一意图识别、域内专项识别、参数提取、格式化、ECharts 生成、趋势分析等。这是系统的"大脑"——连接自然语言和结构化参数的核心。

### 工具层（`src/utils/`）

- `aggregation_tools.py`（1620 行）— 聚合底层纯函数（`raw_filter`、`raw_sum_field` 等），每个操作一个函数，可独立测试
- `echarts_utils.py` — ECharts 代码路径实现（`build_dense_time_series_echarts`、96 点时间序列检测与排序）
- `model_proxy.py` — `ProxyChatModel`，兼容 LangChain 接口的 HTTP 代理模型
- `python_sandbox.py` — 趋势分析的安全代码执行沙箱（AST 剥离 import + 白名单 builtins + 超时控制）
- `filter_think_tags.py` — `<think>...</think>` 标签过滤（4 种实现：同步/异步 × 流式/非流式）
- `fuzzy_match.py` — 多策略实体名模糊匹配（精确/前缀/包含/数字模糊）
- `data_standard.py` — 数据标准化
- `output_parser.py` — JSON 块解析（从 LLM markdown 输出中提取 JSON）
- `summary_util.py` — 摘要数据压缩处理
- `logging_setup.py` — trace_id 日志追踪
- `get_file_list_by_intents.py` — 根据识别的意图生成推荐文档链接

### 服务层（`src/service/`）

#### `agent_factory.py` + `workflow_factory.py`
单例工厂，进程级共享模型实例（避免每个请求创建独立的 HTTP 客户端）：
- `WorkflowFactory` 为意图识别、参数提取、格式化、问题拆分、ECharts、趋势分析等阶段提供共享模型实例。
- `AgentFactory` 负责创建 `FormatAgent` / `EChartsAgent`；当 `INNER_MODEL_ENABLE=TRUE` 时，这两个 Agent 会切换为 `InnerModelAgent`。

#### `session_manager.py`
会话注册表，`InMemorySaver` checkpointer 管理。上限 1000 会话，1 小时过期自动清理。

### API 层（`src/api/`）

- `routers/query_router.py` — SSE 流式响应格式化层 + 缓存管理 API
  - `POST /v1/query` — 智能查询（流式/非流式）
  - `GET /v1/query/full-data/{query_id}?user=xxx` — 获取被截断的完整数据
  - `DELETE /v1/query/cache/{query_id}?user=xxx` — 清理单条查询缓存
  - `DELETE /v1/query/cache/user/{user}` — 清理用户所有缓存
  - `GET /v1/query/cache/stats` — 查看缓存统计
- `routers/shared_cache.py` — 缓存实现（async Lock 保护，5 分钟过期）
- `schemas/schemas.py` — Pydantic 请求/响应模型（`QueryRequest`、`ReturnQuery`、`SSEPayload`）

## 环境变量要点

- `GATEWAY_MODELS`: 逗号分隔的模型名，在此列表中的模型走中转代理而非直连 DashScope
- `COMMON_API_URL`: 统一网关入口，所有业务 API 调用均通过此 URL 透传
- `INNER_MODEL_ENABLE`: 为 `TRUE` 时，`FormatAgent` 与 `EChartsAgent` 切换到内网模型服务
- `FORMAT_SUMMARY_MAX_DATA_CHARS`: 格式化阶段的字符数阈值（默认 28000），超出则跳过格式化
- `MAX_SUBQUESTION_CONCURRENCY` / `MAX_SHARED_SUBQUESTION_CONCURRENCY`: 子问题并发数控制
- `SERVER_HOST` / `SERVER_PORT`: 服务监听地址和端口（`config.py` 默认端口为 6060，项目 `.env` 使用 6066）

> **关于业务 API URL 的读取方式**：`src/config.py` 的 `Settings` 类只读 `COMMON_API_URL` 和 `EMERGENCY_DAYAHEAD_API_URL` 两个业务 URL。`.env` 中定义的其他业务 API URL（如 `REALTIME_MARKET_API_URL`、`LONGTERM_CONTRACT_API_URL` 等）是直接在对应的 workflow 文件中通过 `os.getenv()` 读取的。新增业务意图时，这两种方式均可使用。

**注意：没有 `.env.example` 模板文件**。如需分享项目配置结构而不泄露密钥，可手动创建。

## 配置文件

`data/env_variables/` 下的 JSON 配置文件是系统行为的关键数据源：
- `time_point_mapping.json`: 96 个标准时间点字段（v0000-v2345）及其别名
- `echarts_config.json`: 图表配置
- `emergency_dispatch_fields_config.json`: 应急调度字段定义
- `fuzzy_match_config.json`: 模糊匹配配置
- `operations_config.json`: 操作配置
- `data_standard.json`: 数据标准化配置

`data/knowledge/` 下的知识库文件：
- `rewrite_knowledge.json`: 172 个改写模板，用于问题改写阶段的 slot 填充
- `holidays.json`: 节假日数据，用于补全缺失的年份信息

`.claude/settings.local.json`: Claude Code 的本地设置文件（已存在，修改时注意兼容）。

## 添加新业务意图

1. 在 `src/workflow/workflow_router.py` 的 `WORKFLOW_REGISTRY` 和 `API_URL_MAPPING` 中注册
2. 创建 `src/workflow/xxx_workflow.py` 继承 `BaseWorkflow`，实现 `get_parameter_prompt()`、`get_format_prompt()`、`validate_params()`、`_call_api_impl()`
3. 在 `src/agents/prompts.py` 中添加参数提取和格式化提示词模板
4. 如需域内专项意图识别，在 `_get_domain_intent_agent()` 中注册二级 prompt
5. 在 `.env` 中配置对应的业务 API URL

## 模型调用路径

系统存在**三条 LLM 调用路径**，由配置文件和环境变量共同决定：

| 路径 | 触发条件 | 实现 | 用途 |
|------|---------|------|------|
| **DashScope 直连** | 默认，`GATEWAY_MODELS` 为空 | `ChatQwen`（langchain-qwq） | 直接调用阿里 DashScope API |
| **网关代理** | 模型名在 `GATEWAY_MODELS` 列表中 | `ProxyChatModel`（`model_proxy.py`） | 通过内网中转服务 `GATEWAY_BASE_URL` 代理调用 |
| **内网模型直连** | `INNER_MODEL_ENABLE=TRUE` | `InnerModelAgent`（`inner_model_agent.py`） | 通过 aiohttp 直连内网模型服务的 SSE 流式 API |

模型实例由 `workflow_factory.py` / `agent_factory.py` 单例工厂管理，进程级共享。各阶段（意图识别、参数提取、格式化、图表生成、趋势分析）可使用**不同模型**，由 `src/config.py` 中对应的 `*_MODEL` / `*_API_KEY` / `*_API_BASE` 环境变量独立配置。

**路由细节**：
- 意图识别、参数提取、问题拆分、ECharts、趋势分析模型：由 `WorkflowFactory._create_chat_model()` 根据 `GATEWAY_MODELS` 决定使用 `ProxyChatModel` 还是 `ChatQwen`。
- 格式化与 ECharts 输出： additionally 受 `INNER_MODEL_ENABLE` 控制；为 `TRUE` 时由 `AgentFactory` 返回 `InnerModelAgent`。

## User Rules

1. 写完一个功能必须进行单元测试，测试不通过就继续修改，直到测试通过
2. 每个功能完成后，必须提交代码到 GitHub
3. 必须提交到 GitHub 后才能够继续下一个功能的开发
