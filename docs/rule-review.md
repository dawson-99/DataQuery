# 规则审查子系统 — 总文档(设计 + 流程 + 面试优化)

> 说明:本文件由原 `rule-review-design.md` / `rule-review-workflow.md` / `rule-review-optimization.md`
> 三份文档合并重构而来,是规则审查子系统的**唯一权威入口**。
> 章节编号体系沿用设计文档 v2(§1-§14),**代码注释中的 § 引用继续有效**。
> 三部分:① 详细设计(技术细节)② 工作流程详解(带端到端示例)③ 优化方向清单(面试素材)。

## 总目录

- **第一部分 详细设计**(原 rule-review-design.md):系统概述 / 架构 / 数据流 / 模块详细设计 / 接口 / 数据模型 / Prompt / Tool 系统 / RAG 策略 / 集成 / 依赖 / 实施计划 / 完整工作流(12 分支)/ 异常处理
- **第二部分 工作流程详解**(原 rule-review-workflow.md):系统概览 / 端到端示例(9 阶段走一遍)/ 分支场景速查 / 人机分工 / 常见问题
- **第三部分 优化方向清单**(原 rule-review-optimization.md):回答框架 / 方向一~十(评估闭环、Judge 接线、基础设施、真流式、多轮记忆、可观测性、检索增强、工具增强、RAGAS 评测、Corrective-RAG 回环)/ 工具候选 / 回答节奏

---

# 第一部分 详细设计(原 rule-review-design.md)

## 电力规则审查系统 — 详细设计文档 v2

> 版本：v2.0 | 日期：2026-07-05 | 状态：Phase 1 设计（API 模式）

---

## 目录

1. [系统概述](#1-系统概述)
2. [架构设计](#2-架构设计)
3. [数据流设计](#3-数据流设计)
4. [模块详细设计](#4-模块详细设计)
5. [接口设计](#5-接口设计)
6. [数据模型设计](#6-数据模型设计)
7. [Prompt 设计](#7-prompt-设计)
8. [Tool 系统设计（重点）](#8-tool-系统设计)
9. [RAG 检索策略](#9-rag-检索策略)
10. [与现有系统的集成](#10-与现有系统的集成)
11. [技术依赖与配置](#11-技术依赖与配置)
12. [实施计划](#12-实施计划)
13. [完整工作流设计（12 个分支场景）](#13-完整工作流设计12-个分支场景)
14. [异常处理策略](#14-异常处理策略)

---

## 1. 系统概述

### 1.1 业务背景

电力交易涉及大量规则文档（交易规则、监管办法、实施细则等），当前业务人员需人工查阅 PDF 来判断交易行为是否符合规则。本系统用大模型 + RAG 实现自动化规则审查。

### 1.2 核心能力

```
用户输入：自然语言规则审查问题
    ↓
系统输出：结构化 JSON {decision, reason, evidence}
```

### 1.3 设计原则

- **证据驱动**：所有判断必须引用原文，不得编造
- **API 优先**：Phase 1 全部走 API 调用，后续替换为本地模型
- **并行子系统**：新增代码不侵入现有的电力数据查询逻辑
- **逐步增强**：Phase 1 先跑通 RAG + LLM + Judge 核心链路（不含 Tool），Phase 2 增加 Tool 系统

---

## 2. 架构设计

### 2.1 整体架构

```
┌──────────────────────────────────────────────────┐
│                   app.py (FastAPI)                │
│                                                   │
│  ┌──────────────────┐  ┌───────────────────────┐ │
│  │  /v1/query       │  │  /v1/rule-review      │ │
│  │  (现有数据查询)    │  │  (新增规则审查)        │ │
│  │  query_router.py  │  │  rule_review/router.py│ │
│  └────────┬─────────┘  └───────────┬───────────┘ │
│           │                         │              │
│  ┌────────▼─────────┐  ┌───────────▼───────────┐ │
│  │ WorkflowRouter   │  │  RuleReviewPipeline   │ │
│  │ BaseWorkflow     │  │  (独立编排器)          │ │
│  └──────────────────┘  └───────────┬───────────┘ │
│                                     │              │
│  ┌──────────────────────────────────▼───────────┐ │
│  │           共享基础设施层                       │ │
│  │  日志 / 沙箱 / 模型管理 / 会话 / SSE 流式     │ │
│  └──────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

### 2.2 规则审查子系统内部架构

```
POST /v1/rule-review
  │
  ▼
RuleReviewPipeline.execute()
  │
  ├── 0. 问题改写 → 时间标准化 + 实体名标准化 + 术语映射
  │                 （复用 rewrite_workflow.py 的知识库加载模式）
  │
  ├── 1. 问题澄清判断 → 问题是否明确？
  │     ├── 是 → 继续
  │     └── 否 → 返回澄清追问（suggestions），不继续后续流程
  │
  ├── 2. 问题拆分判断 → 涉及多个文档？
  │     ├── 是 → 拆分为子问题，每个子问题独立检索
  │     └── 否 → 单文档检索 → 继续
  │
  ├── 3. Query 优化 → 术语标准化 + 同义词扩展 + 多 query 变体
  │
  ├── 4. RAG 检索 → BM25 + bge-m3 向量 → RRF 融合排序
  │     ├── 有结果 → 继续
  │     └── 空结果 → 空检索兜底策略（扩大搜索 → 回复"未找到"）
  │
  ├── 5. LLM 生成 → Qwen3B 注入 System Prompt + chunks
  │                  → 流式推理 → 判断"文档中是否有相关规则"
  │                  → 无相关规则 → 直接回复"文档中未找到"，跳过后续步骤
  │
  ├── 6. Tool 调用 [Phase 2] → 按需调用工具（最多 3 轮）
  │     └── 3 轮后仍未解决 → 降级为纯 LLM 判断（附带 tool 未完成标记）
  │
  ├── 7. Judge 校验 → DeepSeek-v4 验证幻觉 + 逻辑自检
  │     └── Judge 失败/超时 → 跳过校验，以 LLM 原始结果输出（标注未校验）
  │
  └── 8. SSE 流式输出 → 每阶段进度 + 最终结果
```

### 2.3 完整工作流树（12 个分支场景）

```
用户输入
  │
  ├── 场景 A：问题不明确（缺少日期、实体名模糊）
  │     → 问题澄清 → 返回 suggestions
  │
  ├── 场景 B：单文档查询，检索有结果
  │     ├── B1：LLM 无需工具 → 直接输出 → Judge → 返回
  │     ├── B2：LLM 需要工具 → 工具执行 1-3 轮 → 最终输出 → Judge → 返回
  │     └── B3：文档中无相关规则 → LLM 回复"未找到" → 跳过 Judge
  │
  ├── 场景 C：多文档查询
  │     → 问题拆分 → 多文档并行检索 → 结果合并 → LLM 推理 → ...（同 B1/B2）
  │
  ├── 场景 D：检索无结果（空检索）
  │     → 扩大搜索 → 仍空？→ 回复"未找到相关文档"
  │
  ├── 场景 E：Tool 循环 3 轮后未解决
  │     → 降级为纯 LLM 判断 → 输出附带 tool_unsolved: true 标记 → Judge
  │
  └── 场景 F：Judge 失败 / 超时
        → 跳过校验 → 输出附带 judge_skipped: true 标记

---

## 3. 数据流设计

### 3.1 各阶段数据格式变化

```
阶段              输入                                          输出
─────────────────────────────────────────────────────────────────────────────────
0. 问题改写        原始自然语言问题                                标准化问题
                                                                 (时间格式统一 + 实体名标准化)

1. 问题澄清判断    标准化问题                                     {needs_clarification: bool, suggestions: [...]}

2. 问题拆分判断    标准化问题                                     [{sub_query, target_doc}, ...]
                                                                 (单文档时只有一条)

3. Query优化       子问题                                         优化后的 query 列表
                                                                 [原始query, 变体1, 变体2]

4. RAG检索         优化后的 query 列表                            Top-K=10 chunks
                                                                 [{chunk_id, text, score, section, page, doc_name}, ...]
                                                                 或空列表 []

5. 空检索兜底      空列表 []                                      - 扩大搜索后的 chunks
                                                                 - 或 {"not_found": true}

6. LLM生成         System Prompt + chunks + 用户问题              流式 JSON
                                                                 {"decision","reason","evidence","confidence",
                                                                  "tool_calls":[], "not_found": true|false}

7. Tool调用[Phase2] LLM 输出的 tool_calls 数组                    tool 执行结果
                                                                 或 {"tool_unsolved": true}（3轮后未解决）

8. Judge校验       LLM 输出 + 原始 chunks + query + tool日志      最终结果 + 幻觉标注
                                                                 或 {"judge_skipped": true}（Judge失败时）

9. SSE输出         各阶段结果                                     text/event-stream
```

---

## 4. 模块详细设计

### 4.1 文件结构

```
src/rule_review/
├── __init__.py              # 包初始化 + 全局单例（DocumentStore、模型实例）
├── pipeline.py              # RuleReviewPipeline 编排器（含完整工作流树 + 异常降级）
├── router.py                # FastAPI 路由 + SSE 流式响应格式化
├── schemas.py               # Pydantic 请求/响应模型
├── query_rewriter.py        # 问题改写（时间标准化 + 实体名标准化 + 地名归一化 + 术语映射）
├── document_store.py        # PDF 解析（含 OCR）+ chunk 切分（含标题+表格联合建 chunk）+ 索引管理
├── retriever.py             # BM25 + bge-m3 向量检索 + RRF 融合 + Cross-Encoder 精排 + QueryOptimizer + 空检索兜底
├── generator.py             # Qwen3B LLM 推理（复用 ProxyChatModel）
├── judge.py                 # DeepSeek-v4 结果校验 + Judge 失败兜底
├── tool_executor.py         # Tool 调用运行时 + 5 个工具实现 + 终止条件 [Phase 2]
├── prompts.py               # 规则审查专用 Prompt（动态生成 tool 规则 + 术语映射）
├── sandbox_utils.py         # 沙箱辅助（封装 PythonSandbox，增加工具执行上下文）
└── audit.py                 # 审计追溯: 答案来源追踪 + 日志审计 + 合规核查

data/env_variables/
├── tools_config.json        # 工具定义配置（新增）
└── rule_terms.json          # 术语 + 同义词映射（新增）

# 地名归一化复用现有文件:
#   data/env_variables/data_standard.json → name_abbreviation (43个标准地名)
#   src/utils/data_standard.py → data_matching() (difflib 模糊匹配)
```

### 4.2 `router.py` — API 路由

```python
router = APIRouter(prefix="/v1/rule-review", tags=["规则审查"])

@router.post("")
async def rule_review(req: RuleReviewRequest):
    """规则审查主接口，支持 SSE 流式 / 非流式"""

@router.post("/documents")
async def upload_document(file: UploadFile):
    """上传规则 PDF → 解析 → chunk → 索引"""

@router.get("/documents")
async def list_documents():
    """列出已入库文档"""

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档 + 清理索引"""
```

### 4.3 `document_store.py` — 文档解析与存储（含 OCR）

核心类：`DocumentStore`，负责 PDF → text → chunk → embedding → 索引。

#### 文档解析流水线

```
PDF 文件
  │
  ├── 文本 PDF（可直接提取文本）
  │     └── pymupdf 提取 text + table
  │
  └── 扫描版 PDF（图片，需 OCR）
        ├── Unlimited-OCR 方案: 文字检测 + 文字识别 + 版面分析
        │     参考: https://github.com/baidu/Unlimited-OCR
        │
        ├── 步骤:
        │   1. pymupdf 提取每页为图片 (300 DPI)
        │   2. PaddleOCR 做文字检测 + 识别
        │   3. PP-Structure 做表格检测 + 结构化提取
        │   4. 按版面分析结果重组文本流
        │
        └── 降级: 如果 PaddleOCR 不可用，pymupdf 直接提取（可能丢字）
  │
  ▼
统一文本内容
  │
  ├── 标题检测: 正则匹配 "第X章"、"第X条"、"一、" 等
  ├── 表格检测: 识别表格区域 → 结构化提取 → Markdown table 格式
  └── 章节层级: 自动构建 章→节→条 的层级树
  │
  ▼
Chunk 切分 (见下方策略)
```

#### OCR 实现

```python
class OCRProcessor:
    """扫描版 PDF OCR 处理。

    参考百度 Unlimited-OCR 方案:
    - 文字检测: DB (Differentiable Binarization)
    - 文字识别: SVTR / CRNN
    - 表格识别: PP-Structure (table detection + structure recognition)
    - 版面分析: PicoDet 布局检测

    Phase 1 实现: 使用 PaddleOCR 封装（PaddleOCR 已集成上述所有模块）
    Phase 2 可替换: 部署 Unlimited-OCR 服务（更高精度、更快速度）
    """

    def __init__(self, use_gpu: bool = False):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(
            use_angle_cls=True,    # 文本方向分类
            lang='ch',             # 中英文混合
            use_gpu=use_gpu,
            show_log=False,
        )

    def process_page(self, image_path: str) -> dict:
        """OCR 单页，返回结构化结果"""
        result = self.ocr.ocr(image_path, cls=True)

        # 分类结果: 纯文本块 / 表格块
        text_blocks = []
        table_blocks = []
        for line in result[0]:
            bbox, (text, confidence) = line
            if self._is_table_region(bbox):
                table_blocks.append({"bbox": bbox, "text": text})
            else:
                text_blocks.append({"bbox": bbox, "text": text, "confidence": confidence})

        return {
            "text_blocks": text_blocks,
            "table_blocks": table_blocks,
        }

    def _is_table_region(self, bbox) -> bool:
        """判断区域是否为表格（基于形状 + 内含分隔符）"""
        # 宽高比 + 内部是否含 | 或制表符
        pass
```

#### Chunk 策略

| 参数 | 值 | 说明 |
|------|---|------|
| chunk_size | 800 tokens (~600 中文) | 保证包含完整的规则条文 |
| chunk_overlap | 150 tokens | 跨 chunk 上下文连贯 |
| 切分方式 | 按章节标题自然切分 | 保留文档结构层级 |
| 标题联合建 chunk | 表格内容 + 所在章节标题 + 表名合并 | 避免检索到孤立表格 |
| 表格处理 | OCR 检测 → PP-Structure 结构化 → Markdown table | 保留行列结构，含标题行 |

**标题 + 表格联合建 chunk 规则**：

```
规则: 表格不能单独成 chunk，必须与标题联合。

示例：
  原文:
    第三章 交易规则
    第12条 价格上限
    下表为各省日前现货出清电价上限：
    | 省份 | 电价上限(元/MWh) |
    |------|-----------------|
    | 冀北 | 760             |

  生成的 chunk:
    [第三章 > 第12条 价格上限] 各省日前现货出清电价上限：
    | 省份 | 电价上限(元/MWh) |
    |------|-----------------|
    | 冀北 | 760             |

规则:
  1. 检测到表格时，向上回溯最近的章节标题和条款标题
  2. 将标题 + 表前引导句 + 表格内容合并为一条 chunk
  3. 如果标题+表格总长度超过 chunk_size，优先截断表格（保留标题+引导句+表头+前N行）
```

**Chunk 元数据**：

```python
@dataclass
class Chunk:
    chunk_id: str               # uuid
    doc_id: str                 # 所属文档 ID
    text: str                   # chunk 文本（含 Markdown table）
    tables: list[dict]          # 结构化表格数据 [{headers, rows, caption}]
    section: str                # 章节标题（如 "第三章 交易规则 第12条 价格上限"）
    section_hierarchy: list     # 章节层级 ["第三章 交易规则", "第12条 价格上限"]
    page: int                   # 页码
    ocr_confidence: float       # OCR 置信度（扫描版有，文本版为 1.0）
    is_scanned: bool            # 是否扫描版
    embedding: np.ndarray       # bge-m3 向量 (1024 维)
```

#### 解析分层（PDFParser 抽象 + MinerU 本地 + 手动入库）

**方案演进**：建立向量数据库对解析精度要求非常高。初期尝试调用 API 解析政策 PDF，
随着新业务进来、PDF 数量增多，发现该方案**成本高**（按页/按文档计费，量上来后不可控）且
**表格中的重要内容无法完全准确识别**（如各省电价上限表的单元格错位、丢列），因此弃用
（历史方案，不写代码）。改为分层策略：**相对不重要的文档采用本地 MinerU 解析，
重要的政策问答中的表格采用手动入库**。在测试中，这种处理方式的召回率达到 **98%**
（评测口径见下「98% 双评测」）。

**分层决策表**：

| 文档类型 | importance | parse_mode | 解析路径 | 说明 |
|---|---|---|---|---|
| 相对不重要的规则文档 | low | auto | MinerU 本地解析（未安装降级 pymupdf） | 批量上传，表格可完整识别 |
| 重要的政策问答/表格 | high | manual | 手动入库（`POST /documents/manual`） | 人工整理 Markdown，识别准确率 100% |
| 扫描件/特殊格式 | 任意 | pymupdf | pymupdf 文本提取 + PaddleOCR 兜底 | 显式指定，跳过 MinerU |

**代码结构**（`src/rule_review/parsers.py`，原 document_store.py 解析逻辑迁移）：

- `PDFParser` 抽象基类（镜像 `OCRProcessor` 的可用性降级模式）：`name` + `is_available` + `parse(data, filename) -> list[PageContent]`
- `PymupdfParser`：原 `_parse_document`/`_extract_text_page`/`find_tables()` 逻辑原样迁移，行为零改动
- `MinerULocalParser`：`from mineru import MinerU` 延迟导入，未安装时 `is_available=False` 静默降级 pymupdf；单例复用避免重复加载模型；输出 markdown 经 `markdown_to_page_content` 还原为 `PageContent`
- `markdown_to_page_content`：MinerU 输出与手动入库共用——`#` 标题 → TextBlock（「第X章/条」交给 `_detect_heading` 正则）、连续 `|` 行 → TableBlock（跳过分隔行）、合成 bbox 按行序保序、表格 caption 由 `_build_chunks` 的 recent_texts 回溯覆盖
- `DocumentStore.ingest(..., importance="low", parse_mode="auto")`：auto 时 MinerU 可用则用之否则 pymupdf；mineru 强制（不可用降级）；pymupdf 恒走本地；实际生效模式（含降级后）写入 `DocumentInfo.parse_mode`
- `DocumentStore.ingest_manual(markdown, ...)`：手动入库不生成 PDF，`parse_markdown → _build_chunks` 与 PDF 上传完全同路径；`delete()` 兼容无 PDF 文档

已知限制：非「第X章/条」结构的 markdown 标题会退化为正文；表格单元格内含 `|` 不做转义。

**98% 双评测**（阈值常量：`TABLE_CELL_ACCURACY_TARGET=0.98`、`RETRIEVAL_RECALL_TARGET=0.98`）：

1. **解析精度**（`src/rule_review/parsing_eval.py`）：表格单元格识别准确率 = 正确单元格 / 期望表单元格总数
   （期望表手工标注，缺行/缺列记错，多出的行/列不计入分母）。手动入库 = markdown 原文还原 → 100%；
   MinerU = 对真实 PDF 跑解析后与标注表对比 → 期望 ≥98%。运行：`python -m src.rule_review.parsing_eval`，
   评测数据 `data/evaluation/table_parse_cases.json`。
2. **检索召回**：表格类用例（tag=「表格检索」，tc-015~018）经现有 EvalRunner 的 `avg_recall_at_k`
   （关键词代理）统计，阈值 ≥98%。运行：`python -m src.rule_review.evaluation`。

**配置**（`src/config.py`，默认值保证本地开发零行为变化）：`RULE_REVIEW_MINERU_ENABLED`（默认 false）、
`RULE_REVIEW_PARSE_MODE`（默认 auto）。`requirements.txt` 中 `mineru>=2.0` 为可选依赖（注释标注）。

### 4.4 `retriever.py` — 混合检索引擎（含 Cross-Encoder 精排 + 地名归一化）

**检索流程**：

```
query
  ├── QueryOptimizer.optimize()
  │     ├── 地名归一化: "冀北" → difflib 模糊匹配 data_standard.json（复用 data_standard.py）
  │     ├── 专业词汇归一化: "电价上限" → "出清电价上限"（加载 rule_terms.json）
  │     └── 同义词扩展: 生成 2-3 个变体 query
  │
  ├── [并行]
  │   ├── BM25 召回 (bm25s)        → top 30
  │   └── bge-m3 向量召回 (FAISS)  → top 30
  │
  ├── RRF 融合去重                 → top 20（给精排留余量）
  │
  ├── Cross-Encoder 精排           → top 10
  │     模型: BAAI/bge-reranker-v2-m3
  │     对每个 (query, chunk) 对打分，按分数重排序
  │
  └── 返回 Top-K=10 chunks
```

**嵌入模型选型：bge-m3**

| 属性 | 值 |
|------|---|
| 模型 | BAAI/bge-m3 |
| 维度 | 1024 |
| 大小 | ~2.2 GB |
| 特性 | 多语言、支持 dense + sparse 双向量 |
| Phase 1 | 使用 dense 向量 + 独立 BM25 |
| Phase 2 | 可切换为 bge-m3 自带 sparse（统一模型） |

**精排模型选型：bge-reranker-v2-m3**

| 属性 | 值 |
|------|---|
| 模型 | BAAI/bge-reranker-v2-m3 |
| 大小 | ~1.5 GB |
| 特性 | Cross-Encoder，逐对 (query, chunk) 打分 |
| 精度 | 显著优于 RRF 等无监督融合 |

**融合 + 精排算法**：

```python
async def hybrid_retrieve_with_rerank(
    query: str,
    retriever: HybridRetriever,
    reranker: CrossEncoder,
    top_k: int = 10,
) -> list[dict]:
    """
    BM25 + 向量 → RRF → Cross-Encoder → Top-K
    """
    # Step 1: 粗排（并行）
    bm25_results = retriever.bm25_search(query, k=30)
    vector_results = retriever.vector_search(query, k=30)

    # Step 2: RRF 融合 → top 20
    fused = rrf_fusion(bm25_results, vector_results, k=60)[:20]

    # Step 3: Cross-Encoder 精排
    pairs = [(query, item["text"]) for item in fused]
    scores = reranker.predict(pairs)  # 每对返回一个分数

    # Step 4: 按精排分数重排序 → top_k
    for item, score in zip(fused, scores):
        item["rerank_score"] = float(score)
    ranked = sorted(fused, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    return ranked
```

**Query 优化器（含地名归一化）**：

```python
class QueryOptimizer:
    """查询优化：地名归一化 + 专业词汇归一化 + 同义词扩展。

    地名归一化: 复用现有的 data/env_variables/data_standard.json
    - name_abbreviation: 43 个标准地名（冀北、山西、四川主网、蒙东、蒙西等）
    - 匹配时使用 difflib.get_close_matches（复用现有 data_standard.py 的模式）

    专业词汇归一化: data/env_variables/rule_terms.json（新增）
    """

    def __init__(self):
        # 复用现有的地名标准库
        with open("data/env_variables/data_standard.json", "r") as f:
            data_std = json.load(f)
        self.standard_place_names = data_std.get("name_abbreviation", [])
        # 43个: 冀北、山西、四川主网、四川攀西、蒙东、蒙西、华东、华北...

        with open("data/env_variables/rule_terms.json", "r") as f:
            self.term_map = json.load(f)

    def optimize(self, query: str) -> list[str]:
        """
        返回优化后的 query 列表（含原始 + 变体）

        处理顺序:
        1. 地名归一化: "冀北" → "冀北分部"
        2. 专业词汇归一化: "电价上限" → "出清电价上限"
        3. 同义词扩展: 生成 2-3 个变体 query
        """
        # 1. 地名归一化
        for alias, standard in self.place_names.get("aliases", {}).items():
            if alias in query:
                query = query.replace(alias, standard)

        # 2. 专业词汇归一化
        for term, info in self.term_map.get("terms", {}).items():
            for alias in info.get("aliases", []):
                if alias in query:
                    query = query.replace(alias, term)
                    break

        # 3. 同义词扩展 → 变体
        variants = [query]
        for term, info in self.term_map.get("terms", {}).items():
            if term in query:
                for synonym in info.get("synonyms", [])[:2]:
                    variants.append(query.replace(term, synonym))

        return variants
```

**地名归一化**（复用现有 `data/env_variables/data_standard.json` + `src/utils/data_standard.py`）：

```python
# 现有文件: src/utils/data_standard.py 已实现
# data_matching(target, candidates) → 用 difflib.get_close_matches 模糊匹配

# 现有文件: data/env_variables/data_standard.json 已有
# name_abbreviation: ["冀北", "山西", "蒙东", "蒙西", "四川主网", ...] 共 43 个

# 规则审查中直接复用:
from src.utils.data_standard import data_matching

# 示例: 用户说"冀北" → difflib 模糊匹配到 "冀北"（cutoff=0.6）
# 用户说"冀北电网" → 模糊匹配到 "冀北"
# 用户说"四川" → 模糊匹配到 "四川主网" 或 "四川攀西"（由 cutoff 决定）
```

### 4.5 `generator.py` — LLM 推理

```python
class RuleReviewGenerator:
    """Qwen3B API 推理"""

    def __init__(self):
        self.model = self._create_model()
        # 复用: ProxyChatModel / ChatQwen

    async def generate_stream(self, query, context_chunks, tool_results=None):
        """构建 messages → model.astream() → yield chunk"""

    async def generate(self, query, context_chunks, tool_results=None):
        """非流式版本，返回完整 dict"""
```

### 4.6 `judge.py` — 结果校验

```python
class RuleReviewJudge:
    """DeepSeek-v4 校验 Qwen3B 的输出"""

    async def verify(self, llm_output, original_query, context_chunks) -> dict:
        """
        逐条检查:
        1. evidence 是否真在 context 中（反幻觉）
        2. decision 与 reason 是否逻辑自洽
        3. 是否有遗漏的重要规则

        返回: {verified, corrections, hallucinated_evidence,
               missing_rules, final_decision, final_reason,
               final_evidence, confidence}
        """
```

### 4.7 `audit.py` — 审计追溯（答案来源追踪 + 合规核查）

**目标**：每条审查结果都可以被外部核查人员追溯到原始规则文档，支持合规部门直接质检。

#### 设计思路

规则审查的 ToB 场景下，用户（合规人员）需要：

1. 知道答案的**每一个判断依据**来自哪个文档、哪一页、哪一段
2. 能够**点击追溯**到原始 PDF 的对应位置
3. 审查过程有**完整日志**，可以被第三方审计
4. 具备**质检接口**，合规部门可以对历史审查结果抽样复核

#### 审计追溯信息结构

```python
# schemas.py 新增

class AuditRecord(BaseModel):
    """单次审查的完整审计记录"""
    query_id: str                        # 审查唯一标识
    timestamp: str                       # 审查时间

    # 用户输入
    original_query: str                  # 原始问题
    rewritten_query: str                 # 改写后问题

    # 检索过程
    retrieval: RetrievalAudit            # 检索详情

    # LLM 推理过程
    llm_generation: LLMGenerationAudit   # LLM 推理详情

    # Tool 调用过程 [Phase 2]
    tool_executions: list[ToolCallLog]   # 工具调用日志

    # Judge 校验过程
    judge_verification: Optional[JudgeAudit]  # Judge 校验详情

    # 最终输出
    final_result: RuleReviewResult       # 最终审查结果

    # 溯源信息
    source_traceability: list[SourceTrace]  # 答案→原文的溯源链

class RetrievalAudit(BaseModel):
    """检索阶段审计"""
    bm25_k: int                          # BM25 召回数量
    vector_k: int                        # 向量召回数量
    fusion_method: str                   # "RRF"
    final_k: int                         # 最终送入 LLM 的 chunk 数
    search_expanded: bool                # 是否触发了扩大搜索兜底
    retrieval_latency_ms: float          # 检索耗时

class LLMGenerationAudit(BaseModel):
    """LLM 推理阶段审计"""
    model: str                           # 使用的模型
    tok_input: int                       # 输入 token 数
    tok_output: int                      # 输出 token 数
    latency_ms: float                    # 推理耗时
    not_found: bool                      # 是否判定"文档中无相关规则"

class JudgeAudit(BaseModel):
    """Judge 校验阶段审计"""
    model: str                           # 使用的模型
    verified: bool                       # 校验是否通过
    hallucinated_count: int              # 检测到的幻觉数
    skipped: bool                        # 是否跳过了校验
    skipped_reason: str                  # 跳过原因
    latency_ms: float                    # 校验耗时

class SourceTrace(BaseModel):
    """单条溯源信息：answers 中的某段结论 → 原始文档位置"""
    result_field: str                    # 对应结果的哪个字段（"decision" / "reason" / "evidence[0]"）
    result_excerpt: str                  # 结果中的原文片段
    source_doc: str                      # 来源于哪个文档
    source_section: str                  # 来源于哪个章节
    source_page: int                     # 来源于哪一页
    source_text: str                     # 原始文档中的原文
    source_chunk_id: str                 # 来源于哪个 chunk
    match_type: str                      # "exact" | "fuzzy" | "llm_extracted"
```

#### 审计存储

```python
class AuditStore:
    """审计日志存储。

    Phase 1: JSON 文件存储（data/audit_logs/{date}/{query_id}.json）
    Phase 2: 数据库存储（PostgreSQL / MongoDB）

    每个审查请求自动生成一条审计记录，包含完整的溯源链。
    """

    def __init__(self, storage_dir: str = "data/audit_logs"):
        self.storage_dir = storage_dir

    async def save(self, record: AuditRecord):
        """保存审计记录"""
        date_dir = record.timestamp[:10]
        path = f"{self.storage_dir}/{date_dir}/{record.query_id}.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.model_dump(), f, ensure_ascii=False, indent=2)

    async def load(self, query_id: str, date: str) -> AuditRecord:
        """加载审计记录"""

    async def list_by_date(self, date: str) -> list[str]:
        """列出某天的所有审计记录"""

    async def sample_for_review(self, date: str, count: int = 10) -> list[AuditRecord]:
        """抽样用于质检"""
```

#### 溯源链的生成

```python
def build_source_traceability(
    llm_output: dict,           # LLM 原始输出
    retrieved_chunks: list[dict], # 检索到的 chunks
    final_result: dict,         # 最终结果（经 Judge 后）
) -> list[SourceTrace]:
    """
    为每条 evidence 构建溯源链。

    方法:
    1. 从最终结果的 evidence 列表中逐条取 text
    2. 在 retrieved_chunks 中找最长公共子串匹配
    3. 记录匹配到的 chunk 来源信息
    4. 无法精确匹配的标记为 "llm_extracted"

    这样合规人员可以看到:
    - "这句话来自《省间电力现货交易规则》第12条 第23页"
    - 点击可跳转到原始 PDF 对应位置
    """
    traces = []
    for i, evidence in enumerate(final_result.get("evidence", [])):
        evidence_text = evidence.get("text", "")
        best_match = None
        best_score = 0

        for chunk in retrieved_chunks:
            # 最长公共子串匹配
            lcs_len = longest_common_substring(evidence_text, chunk["text"])
            if lcs_len > best_score:
                best_score = lcs_len
                # 阈值: 至少 30% 匹配才算可追溯
                if lcs_len / max(len(evidence_text), 1) > 0.3:
                    best_match = chunk

        if best_match:
            traces.append(SourceTrace(
                result_field=f"evidence[{i}]",
                result_excerpt=evidence_text[:200],
                source_doc=best_match.get("doc_name", ""),
                source_section=best_match.get("section", ""),
                source_page=best_match.get("page", 0),
                source_text=best_match["text"][:300],
                source_chunk_id=best_match["chunk_id"],
                match_type="exact" if best_score / max(len(evidence_text), 1) > 0.8 else "fuzzy",
            ))
        else:
            traces.append(SourceTrace(
                result_field=f"evidence[{i}]",
                result_excerpt=evidence_text[:200],
                source_doc="",
                source_section="",
                source_page=0,
                source_text="",
                source_chunk_id="",
                match_type="llm_extracted",
            ))

    return traces
```

#### 审计 API

```python
# router.py 新增

@router.get("/audit/{query_id}")
async def get_audit_record(query_id: str, date: str = None):
    """获取审查的完整审计记录（含溯源链）"""
    # 合规人员可以查看某次审查的完整决策过程

@router.get("/audit/sample/{date}")
async def sample_for_quality_check(date: str, count: int = 10):
    """抽样用于质检 -- 合规部门每日随机抽样核查"""

@router.get("/audit/stats")
async def get_audit_stats(start_date: str, end_date: str):
    """审计统计：审查次数、幻觉率、跳过率等"""
```

#### SSE 输出中附加溯源信息

```
event: message
data: {"answer":"{\"decision\":\"不符合\",...}","source_traces":[{...}]}

event: done
data: {"done":true,"query_id":"q-xxx","audit_url":"/v1/rule-review/audit/q-xxx"}
```

---

## 5. 接口设计

### 5.1 `POST /v1/rule-review` — 规则审查查询

**请求**：

```json
{
  "question": "2025年3月15日冀北的日前现货出清电价达到800元/MWh，是否符合价格上限规则？",
  "stream": true,
  "sessionId": "uuid-xxx",
  "userInfo": {"userId": "user123"}
}
```

**SSE 流式响应**：

```
event: message
data: {"type":"messageLabel","answer":"- <span>查询预处理中...</span>"}

event: message
data: {"type":"messageLabel","answer":"- <span>检索相关知识中...</span>"}

event: message
data: {"type":"messageLabel","answer":"- <span>规则推理中...</span>"}

event: message
data: {"answer":"{\n  \"decision\": \"不符合\",\n  \"reason\": \"...\",\n  \"evidence\": [...]\n}"}

event: message
data: {"type":"messageLabel","answer":"- <span>结果校验中...</span>"}

event: done
data: {"done":true}
```

### 5.2 `POST /v1/rule-review/documents` — 上传文档

multipart 表单字段（解析分层新增 `importance` / `parse_mode`，见 §4.3）：

| 字段 | 必填 | 取值 | 说明 |
|---|---|---|---|
| file | 是 | .pdf | 规则 PDF |
| force | 否 | bool | 兼容字段（delete 接口使用） |
| importance | 否 | high \| low（默认 low） | 文档重要程度，显式标注 |
| parse_mode | 否 | auto \| mineru \| pymupdf（默认 auto） | 解析方式；auto 时 MinerU 可用则用之 |

响应：

```json
{
  "status": "success",
  "data": {"doc_id": "doc_abc", "file_name": "规则.pdf", "page_count": 45,
           "chunk_count": 120, "importance": "low", "parse_mode": "pymupdf"}
}
```

### 5.3 `GET /v1/rule-review/documents` — 文档列表

响应 data 每项含 `importance` / `parse_mode` / `source` 三个解析分层字段。

### 5.4 `DELETE /v1/rule-review/documents/{doc_id}` — 删除文档

### 5.5 `POST /v1/rule-review/documents/manual` — 手动入库（重要政策问答表格）

重要文档不经过 PDF 解析，直接提交人工整理的 Markdown（标题 + 段落 + `|` 表格），
走与 PDF 上传完全相同的 chunk 组装与索引路径，识别准确率 100%（见 §4.3「98% 双评测」）。

请求（JSON）：

```json
{
  "markdown": "## 第2条 价格上限\n下表为各省日前现货出清电价上限：\n| 省份 | 电价上限(元/MWh) |\n|---|---|\n| 冀北 | 760 |",
  "filename": "价格上限表.md",
  "importance": "high",
  "source": "政策问答表格人工整理"
}
```

响应：与 5.2 相同结构，`parse_mode="manual"`、`importance="high"`。

---

## 6. 数据模型设计

```python
# schemas.py

class RuleReviewRequest(BaseModel):
    """规则审查请求"""
    question: str
    stream: bool = True
    sessionId: str = ""
    userInfo: Optional[dict] = None
    top_k: int = Field(default=10, ge=1, le=50)

class RuleReviewResult(BaseModel):
    """最终审查结果"""
    decision: str           # "符合" | "不符合" | "部分符合" | "无法判断"
    reason: str             # 推理过程
    evidence: list[EvidenceItem]
    confidence: float       # 0.0 - 1.0

class EvidenceItem(BaseModel):
    source: str             # 文档名
    section: str            # 章节
    page: int               # 页码
    text: str               # 原文引用
    chunk_id: str

class LLMOutput(BaseModel):
    """LLM 输出的 JSON 结构（包含可能的 tool_calls）。
    与现有系统的 operations 数组模式一致：LLM 在 JSON 中输出工具调用列表，
    @after_model 中间件解析后，Python dispatch 执行。"""
    decision: str = ""
    reason: str = ""
    evidence: list[EvidenceItem] = []
    confidence: float = 0.0
    tool_calls: list[dict] = []   # [{"tool": "extract_table_data", "args": {...}}, ...]
                                  # 正常情况为空数组

class ToolCallLog(BaseModel):
    """工具调用日志（调试 + 训练数据收集用）"""
    query_id: str
    round: int              # 第几轮 LLM 推理
    tool_name: str
    args: dict
    result: dict
    timestamp: str
    latency_ms: float

class DocumentUploadResponse(BaseModel):
    doc_id: str
    file_name: str
    page_count: int
    chunk_count: int
    uploaded_at: str
    importance: str = "low"       # 解析分层：high | low
    parse_mode: str = "pymupdf"   # 解析分层：pymupdf | mineru | manual（实际生效模式）

class ManualDocumentUploadRequest(BaseModel):
    """手动入库请求（重要政策问答表格，见 §4.3 解析分层）"""
    markdown: str                 # 规则 Markdown 内容（# 标题、段落、| 表格 |）
    filename: str = ""            # 文档名称，留空自动生成
    importance: str = "high"      # 手动入库面向重要内容，默认 high
    source: str = ""              # 来源说明（如：政策问答表格人工整理）
```

**DocumentInfo 解析分层新增字段**（`src/rule_review/models.py`，均带默认值保证旧 JSON 反序列化兼容）：

```python
@dataclass
class DocumentInfo:
    doc_id: str
    file_name: str
    page_count: int
    chunk_count: int
    created_at: str
    importance: str = "low"      # high | low —— 调用方显式标注
    parse_mode: str = "pymupdf"  # pymupdf | mineru | manual（实际生效模式，含降级后）
    source: str = ""             # 来源说明（手动入库如「政策问答表格人工整理」）
```

`PageContent` 新增 `is_scanned: bool = False`：解析器抽象后由解析器把扫描页判定透传给 chunk 组装。
`Chunk` 不加字段（importance 为文档级属性）。

---

## 7. Prompt 设计

### 7.0 Prompt 管理方式（遵循现有项目模式）

**现有项目怎么管理 Prompt？**

`src/agents/prompts.py` 中有 `_build_operations_section()` 函数，从 `data/env_variables/operations_config.json` 动态生成操作规则文本，嵌入到意图识别 Prompt 中。Pattern 如下：

```
JSON 配置文件 ──→ _build_xxx_section() ──→ 生成的文本 ──→ Prompt 模板
```

**规则审查系统怎么沿用这个模式？**

同样使用「JSON 配置文件 → 动态生成规则文本 → 嵌入 Prompt」：

```
data/env_variables/tools_config.json    → _build_tools_section()    → System Prompt
data/env_variables/rule_terms.json      → _build_terms_section()    → RAG Context Prompt
```

这样做的好处：
- LLM 输出的 `tool_calls` 是一个 JSON 数组（和现有 `operations` 数组一样），Python 解析后 dispatch 执行
- 不需要引入 LangChain tool calling、不需要 `@tool` 装饰器
- 工具规则放在 JSON 配置文件里，修改时不需要改代码，只需更新 JSON 并重启

### 7.1 配置文件：`tools_config.json`

```json
{
  "tools": {
    "extract_table_data": {
      "name": "表格数据提取",
      "description": "从 Markdown 格式表格中精确提取指定行列的数值。用于规则以表格形式呈现（如各地区电价上限表）时，替代 LLM 直接读表避免看错行列。",
      "triggers": ["表格", "下表", "如下表", "价格表", "上限表"],
      "parameters": {
        "table_text": "Markdown 格式的完整表格文本",
        "filter_column": "用于定位目标行的列名",
        "filter_value": "用于定位目标行的值（支持模糊匹配）",
        "select_column": "要提取数值的列名"
      },
      "output": {"value": "number", "unit": "string", "row_number": "int"}
    },
    "arithmetic_compare": {
      "name": "精确算术比较",
      "description": "对实际值与规则阈值进行精确比较。规则审查的核心判断——"800 > 760 吗？"由代码计算而非 LLM 推理，保证 100% 准确。",
      "triggers": ["比较", "对比", "是否超过", "是否低于", "大于", "小于", "等于", "介于"],
      "parameters": {
        "actual": "实际值（数值，必须先通过 unit_converter 统一单位）",
        "operator": "比较运算符：gt(大于) / gte(大于等于) / lt(小于) / lte(小于等于) / eq(等于) / neq(不等于) / between(介于两端之间)",
        "threshold": "阈值",
        "threshold_high": "上限（仅 between 时需要）"
      },
      "output": {"result": "boolean", "expression": "string", "detail": "string"}
    },
    "resolve_cross_reference": {
      "name": "规则交叉引用解析",
      "description": "解析规则文本中的交叉引用（如'参照第5条第2款'、'按照《XXX》第三章执行'），在被引位置找到对应的规则原文。",
      "triggers": ["参照", "按照", "依据", "按照...执行", "第X条", "详见"],
      "parameters": {
        "reference_text": "包含交叉引用语句的文本片段"
      },
      "output": {"references": "array", "each": {"type": "string", "pattern": "string", "resolved_text": "string", "source": "string"}}
    },
    "validate_date_applicability": {
      "name": "规则时效性校验",
      "description": "判断某条规则在给定日期是否有效。检查规则的施行日期和废止日期，防止用旧规则判断新交易。",
      "triggers": ["施行", "废止", "有效期", "生效", "失效", "版本"],
      "parameters": {
        "rule_text": "包含日期信息的规则文本",
        "query_date": "查询日期，格式 YYYY-MM-DD"
      },
      "output": {"is_applicable": "boolean", "effective_date": "string", "expiry_date": "string|null", "version": "string", "reason": "string"}
    },
    "unit_converter": {
      "name": "单位转换",
      "description": "电力交易单位换算。用户说的值和规则写的值单位可能不同（如万kWh vs MWh），转换后统一比较。",
      "triggers": ["万kWh", "亿kWh", "GWh", "kWh", "分/kWh", "元/千度"],
      "parameters": {
        "value": "待转换的数值",
        "from_unit": "原单位",
        "to_unit": "目标单位"
      },
      "output": {"value": "number", "unit": "string"}
    }
  }
}
```

### 7.2 动态生成工具规则文本

```python
# src/rule_review/prompts.py

import json
from datetime import datetime

current_date = datetime.now().strftime("%Y-%m-%d")

# 加载工具配置
with open("data/env_variables/tools_config.json", "r", encoding="utf-8") as f:
    tools_config = json.load(f)

# 加载术语映射
with open("data/env_variables/rule_terms.json", "r", encoding="utf-8") as f:
    rule_terms = json.load(f)


def _build_tools_section() -> str:
    """从 tools_config.json 动态生成工具规则文本（与现有 _build_operations_section() 模式一致）"""
    lines = ["## 可用工具\n"]
    lines.append("当遇到以下场景时，在输出的 `tool_calls` 数组中添加相应的工具调用：\n")
    for tool_name, tool_info in tools_config["tools"].items():
        lines.append(f"### {tool_name}（{tool_info['name']}）")
        triggers = "、".join(f'"{t}"' for t in tool_info["triggers"])
        lines.append(f"触发词：{triggers}")
        lines.append(f"说明：{tool_info['description']}")
        lines.append(f"参数：")
        for param_name, param_desc in tool_info["parameters"].items():
            lines.append(f"  - {param_name}: {param_desc}")
        lines.append(f"返回值：{json.dumps(tool_info['output'], ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)


def _build_terms_section() -> str:
    """从 rule_terms.json 动态生成术语映射表"""
    lines = ["## 术语映射表"]
    lines.append("以下为电力交易领域术语的标准表达及其同义词：\n")
    for term, synonyms in rule_terms.get("terms", {}).items():
        syn_list = "、".join(synonyms)
        lines.append(f"- {term}：{syn_list}")
    return "\n".join(lines)


# 生成动态内容
_tools_section = _build_tools_section()
_terms_section = _build_terms_section()
```

### 7.3 System Prompt（Phase 2 版本）

```
你是电力交易规则审查专家。

## 核心原则
1. 必须基于提供的规则文档原文回答，不得编造任何内容
2. 如果文档中没有相关信息，必须明确说明"文档中未找到相关规则"
3. 每条判断必须附带原文引用作为证据
4. 所有精确计算（数值比较、单位换算、表格提取）必须调用工具完成，不得自行估算

## 输出格式
必须严格输出以下 JSON 格式。工具调用和最终结果整合在同一个 JSON 中：

{
  "decision": "符合 | 不符合 | 部分符合 | 无法判断",
  "reason": "详细的推理过程，包括引用的规则条款",
  "evidence": [
    {
      "source": "文档名称",
      "section": "章节",
      "page": 页码,
      "text": "原文引用"
    }
  ],
  "confidence": 0.0-1.0,
  "tool_calls": []   // 正常情况为空数组；如需调用工具，填入工具调用列表
}

## 决策指南
- "符合"：所有条件满足规则要求
- "不符合"：至少一个条件违反规则
- "部分符合"：部分条件满足但存在问题
- "无法判断"：缺少关键信息或规则不明确

## 工具调用规则
当遇到以下场景时，将需要调用的工具填入 `tool_calls` 数组（从上到下按调用顺序排列）：

{tools_section}

### tool_calls 数组格式
```json
"tool_calls": [
  {
    "tool": "工具名",
    "args": {参数对象}
  }
]
```

### 工具调用示例
假设用户问"冀北电价800元/MWh是否超出上限"，规则文档中的表格为：
| 地区 | 电价上限(元/MWh) |
|------|-----------------|
| 冀北 | 760             |

则应输出：
```json
{
  "decision": "",
  "reason": "",
  "evidence": [],
  "confidence": 0.0,
  "tool_calls": [
    {"tool": "extract_table_data", "args": {"table_text": "| 地区 | 电价上限(元/MWh) |\n|------|-----------------|\n| 冀北 | 760             |", "filter_column": "地区", "filter_value": "冀北", "select_column": "电价上限(元/MWh)"}},
    {"tool": "arithmetic_compare", "args": {"actual": 800, "operator": "gt", "threshold": 760}}
  ]
}
```

系统执行工具后，会将结果追加返回，请基于工具结果重新生成完整的审查结果（此时 tool_calls 为空数组）。
```

### 7.4 RAG Context Prompt

```
## 以下是从规则文档库检索到的相关内容

{context}

## 术语参考
{terms_section}

## 用户问题

{query}

请基于以上规则文档内容回答用户问题。
```

### 7.5 Judge Prompt

```
你是电力交易规则审查结果校验专家。

## 任务
校验以下审查结果：
1. 每条 evidence 是否真正出现在原文中（检查幻觉）
2. decision 与 reason 是否逻辑自洽
3. 是否有遗漏的重要规则

## 原始问题：{original_query}
## 规则原文：{context}
## 审查结果：{llm_output}
## 工具调用日志：{tool_logs}

## 输出格式
{
  "verified": true/false,
  "corrections": [...],
  "hallucinated_evidence": [...],
  "missing_rules": [...],
  "final_decision": "...",
  "final_reason": "...",
  "final_evidence": [...],
  "confidence": 0.0-1.0
}
```

---

## 8. Tool 系统设计

### 8.1 核心理念：沿用现有「声明式 JSON」模式

**现有项目是怎么处理的？**

现有系统（`src/agents/intent_agent.py` + `src/agents/aggregation_agent.py`）的模式：

```
Step 1: JSON 配置文件 + _build_xxx_section() 函数 ──→ 动态生成 Prompt 文本
Step 2: Prompt 文本嵌入 System Prompt → LLM 输出 JSON（含 operations 数组）
Step 3: @after_model 中间件解析 JSON → Python dispatch 执行各操作
Step 4: 结果返回 → LLM 基于结果继续（或直接使用）
```

关键特征：
- `IntentAgent.tools: list = []` — **永远是空列表，不使用 LangChain tool calling**
- LLM 输出的是纯 JSON 字符串，由中间件解析
- 操作定义在 Prompt 文本中（不是 LangChain tool schema）
- Python 侧用 dispatch 模式 `operation_map = {'filter': raw_filter, 'sum': raw_sum_field, ...}` 执行

**规则审查系统怎么沿用这个模式？**

```
Step 1: data/env_variables/tools_config.json ──→ _build_tools_section() ──→ Prompt 文本
Step 2: LLM 输出 JSON，tool_calls 作为其中一个字段（和现有 operations 字段一样）
Step 3: @after_model 解析 → ToolExecutor dispatch 执行各工具
Step 4: 工具结果注入 messages → LLM 继续生成最终结果
```

### 8.2 为什么需要 Tool

大模型有天然短板，工具系统弥补这些短板：

| LLM 短板 | 对应 Tool | 不靠工具的风险 |
|----------|-----------|---------------|
| 数值比较不精确（"800 > 760？"可能出错） | `arithmetic_compare` | 核心判断结果反了 |
| 表格中提取特定值容易错行漏列 | `extract_table_data` | 拿到错误的阈值 |
| 只看到检索到的 chunks，不会主动追交叉引用 | `resolve_cross_reference` | 基于不完整规则判断 |
| 经常忽略时间维度，用旧规则判新交易 | `validate_date_applicability` | 用了已废止的规则 |
| 单位换算数量级错误 | `unit_converter` | 把万kWh当MWh比较 |

### 8.3 工具定义：JSON 配置文件（非 LangChain tool schema）

**关键区别**：工具定义放在 `data/env_variables/tools_config.json` 中，**不在代码中用 `@tool` 装饰器**。

理由：
1. 和现有 `operations_config.json` 模式一致——修改工具规则不需要改 Python 代码
2. 工具描述（触发条件、参数说明、输出格式）也用于生成 Prompt 文本，一处定义、两处使用
3. 非技术人员可以参与工具调优

配置文件结构见 [7.1 节](#71-配置文件tool_configjson)。

### 8.4 工具实现：纯 Python dispatch（不调 LLM）

五个工具都是纯 Python 函数，接收参数 → 计算 → 返回 dict。**不调 LLM，不涉及 LangChain tool calling。**

```python
# src/rule_review/tool_executor.py

from src.utils.python_sandbox import PythonSandbox
from src.utils.fuzzy_match import fuzzy_match  # 复用现有的模糊匹配

class ToolExecutor:
    """工具调用运行时。遵循现有 AggregationAgent 的 dispatch 模式。"""

    # dispatch 映射表：工具名 → 纯 Python 函数
    TOOL_MAP = {
        "extract_table_data": extract_table_data,
        "arithmetic_compare": arithmetic_compare,
        "resolve_cross_reference": resolve_cross_reference,
        "validate_date_applicability": validate_date_applicability,
        "unit_converter": unit_converter,
    }

    @staticmethod
    def execute_tool(tool_name: str, args: dict) -> dict:
        """执行单个工具调用"""
        func = ToolExecutor.TOOL_MAP.get(tool_name)
        if not func:
            return {"success": False, "error": f"未知工具: {tool_name}"}
        try:
            result = func(**args)
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def execute_tool_calls(tool_calls: list[dict]) -> list[dict]:
        """批量执行 tool_calls，返回结果列表"""
        results = []
        for call in tool_calls:
            tool_name = call.get("tool", "")
            args = call.get("args", {})
            results.append({
                "tool": tool_name,
                "args": args,
                "result": ToolExecutor.execute_tool(tool_name, args),
            })
        return results
```

### 8.5 工具详解

#### Tool 1: `extract_table_data` — 表格数据提取

```python
def extract_table_data(
    table_text: str,
    filter_column: str,
    filter_value: str,
    select_column: str,
) -> dict:
    """
    从 Markdown 表格中精确提取数据。

    实现:
    1. 正则解析 Markdown table → List[Dict]
    2. pandas DataFrame 标准化列名
    3. 模糊匹配 filter_value 定位目标行（复用 src/utils/fuzzy_match.py）
    4. 提取 select_column 的值
    5. 返回 {value, unit, matched_filter, row_number}

    示例:
    extract_table_data(
        table_text="| 地区 | 电价上限(元/MWh) |\n| 冀北 | 760 |",
        filter_column="地区", filter_value="冀北",
        select_column="电价上限(元/MWh)"
    ) → {"value": 760, "unit": "元/MWh", "matched_filter": "冀北", "row_number": 1}
    """
```

#### Tool 2: `arithmetic_compare` — 精确算术比较

```python
def arithmetic_compare(
    actual: float,
    operator: str,
    threshold: float,
    threshold_high: float | None = None,
) -> dict:
    """
    精确算术比较。假定输入值已完成单位统一。

    operator 支持: gt / gte / lt / lte / eq / neq / between

    返回:
    {result: bool, expression: "800 > 760", detail: "实际值800超出上限760，超出40"}
    """
```

#### Tool 3: `resolve_cross_reference` — 规则交叉引用解析

```python
def resolve_cross_reference(
    reference_text: str,
    all_chunks: list[dict],
    current_doc_id: str | None = None,
) -> dict:
    """
    解析交叉引用，找到被引条款原文。

    实现:
    1. 正则匹配引用模式:
       - "第X条(第Y款)"、"第X章第Y节" → internal，在当前 doc 中搜索
       - "按照《XXX》第X条"、"参照《XXX》执行" → external，在所有 docs 中搜索
    2. 条款号模糊匹配（"第五条" ↔ "第5条"）
    3. 返回匹配到的原文 + 来源信息
    """
```

#### Tool 4: `validate_date_applicability` — 规则时效性校验

```python
def validate_date_applicability(
    rule_text: str,
    query_date: str,
) -> dict:
    """
    判断规则在给定日期是否有效。

    实现:
    1. 正则提取: "自XXXX年XX月XX日起施行" → effective_date
                "XXXX年XX月XX日废止" → expiry_date
    2. 提取版本号: "（2024年版）"
    3. 比较 query_date 是否在 [effective_date, expiry_date] 区间内
    4. 返回 {is_applicable, effective_date, expiry_date, version, reason}
    """
```

#### Tool 5: `unit_converter` — 单位转换

```python
# 单位换算表（基准：MWh / 元/MWh）
UNIT_TO_MWH = {
    "MWh": 1.0, "万kWh": 10.0, "亿kWh": 10000.0, "GWh": 1000.0, "kWh": 0.001,
}
UNIT_TO_YUAN_PER_MWH = {
    "元/MWh": 1.0, "元/千度": 1.0, "元/万kWh": 0.1, "分/kWh": 10.0,
}

def unit_converter(value: float, from_unit: str, to_unit: str) -> dict:
    """先转到基准单位，再转到目标单位。返回 {value, unit}"""
```

### 8.6 Tool 执行流程（声明式 JSON 模式）

**与现有 AggregationAgent 执行模式完全一致：**

```
LLM 流式推理
  │
  │  LLM 输出 JSON（与现有 operations 数组类似的 tool_calls 数组）:
  │  {
  │    "decision": "",
  │    "reason": "",
  │    "evidence": [],
  │    "confidence": 0.0,
  │    "tool_calls": [
  │      {"tool": "extract_table_data", "args": {...}},
  │      {"tool": "arithmetic_compare", "args": {...}}
  │    ]
  │  }
  │
  ▼
@after_model 中间件（类似现有 parse_block）解析 JSON
  │
  ├── tool_calls 非空？
  │   ├── YES → ToolExecutor.execute_tool_calls(tool_calls)
  │   │        → 每个 tool 校验 schema → dispatch 到纯 Python 函数 → 执行
  │   │        → 结果格式化:
  │   │          {
  │   │            "tool_results": [
  │   │              {"tool": "extract_table_data", "result": {"success": true, "data": {"value": 760, ...}}},
  │   │              {"tool": "arithmetic_compare", "result": {"success": true, "data": {"result": false, ...}}}
  │   │            ]
  │   │          }
  │   │        → 结果注入到 messages（追加 tool 消息）
  │   │        → 重新调用 LLM（让 LLM 基于工具结果生成最终答案）
  │   │        → 最终 LLM 输出（tool_calls 为空数组的完整结果）
  │   │
  │   └── NO  → 直接返回 LLM 结果
  │
  └── 解析最终 JSON → 格式化为 SSE 事件
```

**对比现有 AggregationAgent 的执行流程：**

| 维度 | 现有 AggregationAgent | 规则审查 ToolExecutor |
|------|----------------------|----------------------|
| 操作定义位置 | `operations_config.json` → Prompt 文本 | `tools_config.json` → Prompt 文本 |
| LLM 输出字段名 | `operations` 数组 | `tool_calls` 数组 |
| Python dispatch | `_execute_one()` → `operation_map` | `execute_tool()` → `TOOL_MAP` |
| 是否调 LLM | 否（纯 Python） | 否（纯 Python） |
| 执行后 | 结果直接使用（单轮） | 结果注入 messages → LLM 继续生成（多轮，最多 3 轮） |

**关键差异**：AggregationAgent 是单轮执行——LLM 输出所有 operations → Python 一次执行完 → 返回结果。规则审查的 Tool 是多轮——LLM 输出 tool_calls → Python 执行 → 结果注入 → LLM 基于结果继续推理（最多 3 轮）。多轮是因为规则审查的推理链路比数据聚合复杂得多。

### 8.7 Tool 调用日志

```python
# 每次 tool 调用都在 pipeline 层记录
tool_call_log_entry = {
    "query_id": "q-xxx",
    "round": 1,                         # 第几轮 LLM 推理（最多 3）
    "tool_name": "extract_table_data",
    "args": {"filter_column": "地区", ...},
    "result": {"success": True, "data": {"value": 760, ...}},
    "timestamp": "2026-07-05T10:30:00Z",
    "latency_ms": 12,
}
```

日志用途：
- 调试：看到 LLM 每一步调了什么工具、什么参数、什么结果
- RL 训练数据收集：tool_calls + 参数 + 结果 → 用于后续 GRPO reward 计算

---

## 9. RAG 检索策略

### 9.1 嵌入模型：bge-m3

| 属性 | 值 |
|------|---|
| 模型名 | `BAAI/bge-m3` |
| 维度 | 1024 |
| 大小 | ~2.2 GB |
| 最大长度 | 8192 tokens |
| 特点 | 多语言（中英均优）、支持 dense + sparse 双向量 |
| 加载方式 | `SentenceTransformer("BAAI/bge-m3")` |

**为什么选 bge-m3 而不是 bge-large-zh-v1.5？**

1. bge-m3 是多语言模型，电力规则中常有英文术语混排（"MWh"、"day-ahead"），bge-m3 比纯中文模型处理更好
2. bge-m3 自带 learned sparse 向量，Phase 2 可以替代独立 BM25，统一为一个模型
3. MTEB 中文榜单上 bge-m3 综合表现优于 bge-large-zh-v1.5

### 9.2 检索参数

| 参数 | 值 | 说明 |
|------|---|------|
| BM25 召回数 | 30 | 3 倍 top_k，保证召回 |
| 向量召回数 | 30 | 同上 |
| 融合后 Top-K | 10 | 送入 LLM 的 chunk 数 |
| RRF k 值 | 60 | 标准值 |

### 9.3 Query 优化策略

```
原始 query: "冀北日前电价800元每兆瓦时符合上限吗"

第1层 — 术语表映射:
  "日前" → "日前现货"
  "电价" → "出清电价"
  → "冀北 日前现货 出清电价 800元/MWh 符合 上限"

第2层 — 同义词扩展:
  "符合上限" → "符合上限 超出上限 价格限制 最高限价"
  → 变体1: "冀北 日前现货 出清电价上限 最高限价"
  → 变体2: "冀北 日前现货 价格限制 电价阈值"

第3层 — [Phase 2] LLM 多 query 生成:
  输入: 原始 query + 术语表
  输出: 3 个不同角度的检索 query
```

---

## 10. 与现有系统的集成

### 10.1 `app.py` 变更

```python
# 新增导入
from src.rule_review.router import router as rule_review_router

def create_app():
    # ... 现有代码不变 ...
    app.include_router(query_router)        # 现有
    app.include_router(rule_review_router)  # 新增
```

### 10.2 `config.py` 新增配置

```python
# ====== 规则审查系统 ======
self.RULE_REVIEW_MODEL = os.getenv("RULE_REVIEW_MODEL", "qwen3-max")
self.RULE_REVIEW_API_KEY = os.getenv("RULE_REVIEW_API_KEY", self.DASHSCOPE_API_KEY)
self.RULE_REVIEW_API_BASE = os.getenv("RULE_REVIEW_API_BASE", self.DASHSCOPE_API_BASE)

self.JUDGE_MODEL = os.getenv("JUDGE_MODEL", "deepseek-v4")
self.JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", "")
self.JUDGE_API_BASE = os.getenv("JUDGE_API_BASE", "")

self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

self.RULE_DOCUMENTS_DIR = os.getenv("RULE_DOCUMENTS_DIR", "data/rule_documents")
self.RULE_INDEX_DIR = os.getenv("RULE_INDEX_DIR", "data/rule_index")
```

### 10.3 不影响现有代码

- `src/workflow/` — 完全不动
- `src/agents/` — 不动
- `src/utils/` — 只 import，不修改

### 10.4 复用现有组件

```
src/utils/python_sandbox.py   → Tool 执行沙箱
src/utils/model_proxy.py      → Qwen3B / DeepSeek-v4 API 调用
src/utils/logging_setup.py    → trace_id 日志
src/utils/filter_think_tags.py → <think> 标签过滤
src/utils/fuzzy_match.py      → 表格查询时的实体模糊匹配
src/api/routers/query_router.py → SSE 流式输出模式参考
```

---

## 11. 技术依赖与配置

### 11.1 `requirements.txt` 新增

```
# RAG & 文档解析
pymupdf>=1.24.0              # PDF 文本 + 表格提取
bm25s>=0.9.0                 # BM25 关键词检索
sentence-transformers>=3.0.0 # 向量嵌入（bge-m3）
faiss-cpu>=1.8.0             # 向量相似度索引
```

### 11.2 `.env` 新增

```bash
# ====== 规则审查系统 ======
RULE_REVIEW_MODEL=qwen3-max
RULE_REVIEW_API_KEY=xxx
RULE_REVIEW_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

JUDGE_MODEL=deepseek-v4
JUDGE_API_KEY=xxx
JUDGE_API_BASE=https://api.deepseek.com/v1

EMBEDDING_MODEL=BAAI/bge-m3
RULE_DOCUMENTS_DIR=data/rule_documents
RULE_INDEX_DIR=data/rule_index
```

### 11.3 新增目录

```
data/rule_documents/         # 上传的 PDF 原文
data/rule_index/             # FAISS 索引持久化文件
data/env_variables/
  └── rule_terms.json        # 术语 + 同义词映射（新增）
```

### 11.4 Docker 单机部署

> 当前项目无 Docker 化。以下为新增的容器化方案，单机运行。

#### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 系统依赖：pymupdf 需要 libmupdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预先下载 bge-m3 模型（避免首次启动时的网络延迟）
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# 复制代码
COPY . .

# 创建数据目录
RUN mkdir -p data/rule_documents data/rule_index data/env_variables

EXPOSE 6066

# 启动命令
CMD ["python", "-c", "from app import run; run()"]
```

#### docker-compose.yml（单机，可选 Redis）

```yaml
version: "3.8"

services:
  app:
    build: .
    container_name: data-query
    ports:
      - "6066:6066"
    env_file:
      - .env
    volumes:
      # 持久化数据目录
      - ./data/rule_documents:/app/data/rule_documents
      - ./data/rule_index:/app/data/rule_index
      - ./logs:/app/logs
    restart: unless-stopped
    # 单机模式，1 worker 即可
    environment:
      - UVICORN_WORKERS=1

  # Phase 3 可选的 Redis 缓存
  # redis:
  #   image: redis:7-alpine
  #   container_name: data-query-redis
  #   ports:
  #     - "6379:6379"
  #   volumes:
  #     - redis_data:/data
  #   restart: unless-stopped

# volumes:
#   redis_data:
```

#### 常用运维命令

```bash
# 构建并启动
docker compose up -d --build

# 查看日志
docker compose logs -f app

# 停止
docker compose down

# 进入容器调试
docker compose exec app bash

# 重新构建（依赖变更时）
docker compose build --no-cache
```

#### 关键注意事项

| 关注点 | 说明 |
|--------|------|
| bge-m3 模型 | ~2.2GB，首次构建时下载。Dockerfile 中预下载到镜像层，避免每次启动等待 |
| FAISS 索引持久化 | 通过 volume 挂载 `data/rule_index/`，重启后索引不丢失 |
| PDF 文件持久化 | 通过 volume 挂载 `data/rule_documents/`，重启后文档不丢失 |
| 内存需求 | bge-m3 + FAISS + FastAPI ≈ 建议 8GB+ RAM |
| 磁盘需求 | bge-m3 模型 ~2.2GB + PDF 文件 + FAISS 索引 ≈ 建议 10GB+ 可用空间 |
| .env 安全 | `.env` 通过 `env_file` 注入，不要打包进镜像。加入 `.dockerignore` |

#### .dockerignore

```
.venv
__pycache__
*.pyc
logs/
.git
.env.example
*.md
docs/
CLAUDE.md
LEARNING_ROADMAP.md
```

---

## 12. 实施计划

### Phase 1：核心链路（不含 Tool）

**目标**：完整工作流树 + 所有分支场景 + 异常兜底 → 上传 PDF → chunk → bge-m3 + BM25 检索 → Qwen3B API 推理 → DeepSeek-v4 Judge 校验 → SSE 流式 JSON 输出

| 步骤 | 内容 | 产出文件 |
|------|------|---------|
| 1.1 | 基础设施：config 配置 + Pydantic schemas + app.py 路由注册 | `__init__.py`, `schemas.py`, 修改 `config.py` + `app.py` |
| 1.2 | 问题改写：时间标准化 + 实体名标准化 + 术语映射 | `query_rewriter.py` |
| 1.3 | 文档解析：pymupdf PDF → text → chunk → bge-m3 embedding → FAISS 索引 | `document_store.py` |
| 1.4 | 混合检索：BM25 + 向量检索 + RRF 融合 + QueryOptimizer + 空检索兜底 | `retriever.py` |
| 1.5 | LLM 推理：Prompt 注入 + RAG context + ProxyChatModel 流式调用 | `generator.py`, `prompts.py` |
| 1.6 | 编排器：完整的 9 阶段 pipeline（含澄清 + 拆分 + 异常降级链路） | `pipeline.py` |
| 1.7 | API 路由：SSE 流式响应格式化 + 文档管理接口 | `router.py` |
| 1.8 | Judge 校验：DeepSeek-v4 验证 + Judge 失败兜底 | `judge.py` |
| 1.9 | Docker 化：Dockerfile + docker-compose.yml + .dockerignore | `Dockerfile`, `docker-compose.yml`, `.dockerignore` |
| 1.10 | 端到端联调 + 6 个分支场景测试 | — |

**Phase 1 覆盖的场景**：

| 场景 | 描述 | 测试方法 |
|------|------|---------|
| 场景 A | 问题不明确 → 澄清 | 发送缺少日期的 query → 预期收到 suggestions |
| 场景 B1 | 单文档查询，无需工具 | 发送明确问题 → 预期完整审查 JSON |
| 场景 B3 | 文档中无相关规则 | 发送未入库领域的 query → 预期 "not_found" |
| 场景 C | 多文档查询 | 发送涉及 2 个文档的 query → 预期合并检索结果 |
| 场景 D | 检索无结果（空索引） | 不上传文档直接查询 → 预期 "未找到" |
| 场景 F | Judge 超时 | 模拟慢速 Judge → 预期正常返回（标注 judge_skipped） |
| 正常 | 完整链路 | 完整端到端测试 |

### Phase 2：Tool 系统

**目标**：LLM 推理过程中按需调用 5 个工具 + Tool 终止条件

| 步骤 | 内容 | 产出 |
|------|------|------|
| 2.1 | Tool 实现：5 个工具的纯 Python 函数 | `tool_executor.py` |
| 2.2 | Tool 运行时：tool_call 检测 → schema 校验 → dispatch → 结果注入 | `tool_executor.py` |
| 2.3 | Tool 终止条件：3 轮限制 + 超时 + 降级处理 | `tool_executor.py` + `pipeline.py` |
| 2.4 | LLM tool_call 输出能力验证 | — |
| 2.5 | bge-m3 sparse 向量替代 BM25 实验 | `retriever.py` |

**Phase 2 新增覆盖的场景**：

| 场景 | 描述 |
|------|------|
| 场景 B2 | LLM 需要工具 → 工具执行 1-3 轮 → 最终输出 → Judge |
| 场景 E | Tool 循环 3 轮后未解决 → 降级为纯 LLM 判断 |

### Phase 3：生产增强

- Redis 缓存（docker-compose 中已有可选配置）
- 本地模型替换（Qwen3B API → vLLM 容器）
- 评估体系搭建
- K8s 部署（如需多副本）

---

## 13. 完整工作流设计（12 个分支场景）

### 13.0 工作流总览

规则审查 pipeline 包含 **9 个阶段**：问题改写 → 澄清判断 → 拆分判断 → Query 优化 → RAG 检索 → LLM 生成 → Tool 调用 → Judge 校验 → SSE 输出。每个阶段的输入/输出/异常处理都需明确定义。

### 13.1 阶段 0：问题改写

**目标**：将用户原始问题标准化，统一时间格式、实体名称。

**参照现有系统**：`src/workflow/rewrite_workflow.py` 的模板匹配 + slot 填充模式。

```
实现方式：
1. 时间标准化（正则 + 规则，不调 LLM）：
   - "昨天" → 当前日期 -1 天，格式 YYYY-MM-DD
   - "本月" → 当前月份，如 2026-07-01 至 2026-07-31
   - "2025年3月15日" → 2025-03-15
   - "今年" → 2026 年
   - 复用现有 rewrite_workflow.py 的时间处理规则
   - 节假日处理：引用 data/knowledge/holidays.json

2. 实体名标准化（不调 LLM）：
   - "冀北" → "冀北分部"
   - "华东" → "华东分部"
   - 从 data/env_variables/rule_terms.json 加载实体别名映射
   - 复用现有 data_standard.py 的 difflib 模糊匹配

3. 术语标准化：
   - "日前电价" → "日前现货出清电价"
   - "上限" → "价格上限"
   - 从 data/env_variables/rule_terms.json 加载术语映射
```

```python
# src/rule_review/query_rewriter.py

class QueryRewriter:
    """问题改写器。复用现有系统的知识库加载模式。"""

    def __init__(self):
        # 加载知识库（与 rewrite_workflow.py 相同的模式）
        with open("data/knowledge/holidays.json", "r") as f:
            self.holidays = json.load(f)
        with open("data/env_variables/rule_terms.json", "r") as f:
            self.term_map = json.load(f)

    def rewrite(self, query: str) -> str:
        """改写问题"""
        # 1. 时间标准化
        query = self._normalize_time(query)
        # 2. 实体名标准化
        query = self._normalize_entities(query)
        # 3. 术语标准化
        query = self._normalize_terms(query)
        return query

    def _normalize_time(self, query: str) -> str:
        """时间正则替换"""

    def _normalize_entities(self, query: str) -> str:
        """实体名 difflib 模糊匹配 → 标准名"""

    def _normalize_terms(self, query: str) -> str:
        """术语映射表替换"""
```

### 13.2 阶段 1：问题澄清判断

**目标**：判断用户问题是否足够明确。如果不明确，返回追问，不继续后续流程。

**触发条件**（任一满足即触发澄清）：

| 条件 | 示例 |
|------|------|
| 缺乏时间信息 | "这个交易符合规则吗"（没有说哪天） |
| 实体名模糊 | "某省的电价符合上限吗"（没有说哪个省） |
| 比较对象不明确 | "800 是否符合上限"（没有说哪种交易类型） |
| 问题过于宽泛 | "所有规则有哪些"（意图不明确） |

**澄清方式**（不调 LLM，规则判断）：

```python
def check_clarification_needed(rewritten_query: str) -> dict:
    """
    返回: 
    - {"needs_clarification": False} → 继续后续流程
    - {"needs_clarification": True, "missing": [...], "suggestions": [...]} → 返回追问
    """
    missing = []
    
    # 检查时间信息
    if not has_time_info(rewritten_query):
        missing.append("时间范围")
    
    # 检查实体信息
    entities = extract_entities(rewritten_query)
    if not entities:
        missing.append("查询主体（如省份、节点名称）")
    
    # 检查是否有数值 + 比较意图（规则审查的核心特征）
    if not has_comparison_intent(rewritten_query):
        missing.append("具体数据值（如电价800元/MWh）")
    
    if missing:
        return {
            "needs_clarification": True,
            "missing": missing,
            "suggestions": [
                f"请补充：{'、'.join(missing)}",
                "例如：2025年3月15日冀北的日前现货出清电价达到800元/MWh，是否符合价格上限规则？"
            ]
        }
    return {"needs_clarification": False}
```

**SSE 输出**（澄清场景）：

```
event: message
data: {"type":"messageLabel","answer":"- <span>问题分析中...</span>"}

event: message
data: {"type":"content","answer":"{\"needs_clarification\":true,\"missing\":[\"时间范围\"],\"suggestions\":[\"请补充时间范围。\",\"例如：2025年3月15日冀北的日前现货出清电价800元/MWh是否符合上限？\"]}"}

event: done
data: {"done":true}
```

### 13.3 阶段 2：多文档问题拆分

**目标**：用户问题涉及多个规则文档时，拆分为子问题，每个子问题独立检索。

**触发条件**：
- 用户明确提到多个文档名（"根据《A》和《B》..."）
- 用户问题包含对比意图（"A 和 B 的规则有什么不同"）

**拆分方式**（不调 LLM，规则判断）：

```python
def split_if_multi_document(query: str, document_store: DocumentStore) -> list[dict]:
    """
    检测是否涉及多文档，如果是则拆分。

    返回:
    - 单文档: [{"sub_query": query, "doc_name": None}]  # doc_name=None 表示检索所有文档
    - 多文档: [{"sub_query": "子问题1", "doc_name": "规则A"}, ...]
    """
    doc_names = document_store.list_documents()
    mentioned = [d for d in doc_names if d in query]
    
    if len(mentioned) <= 1:
        return [{"sub_query": query, "doc_name": None}]
    
    # 多文档拆分：为每个文档创建独立检索任务
    sub_items = []
    for doc_name in mentioned:
        sub_query = query.replace(doc_name, "").strip()
        sub_items.append({"sub_query": sub_query, "doc_name": doc_name})
    return sub_items
```

**并发检索**：多个子问题的检索并发执行，用 `asyncio.gather()` 收集结果后合并。

```python
if len(sub_items) > 1:
    # 并发检索每个子问题
    tasks = [retriever.retrieve(item["sub_query"], doc_filter=item["doc_name"]) 
             for item in sub_items]
    all_results = await asyncio.gather(*tasks)
    # 合并 + 去重 + 排序
    merged_chunks = merge_and_deduplicate(all_results)
else:
    chunks = await retriever.retrieve(query)
```

### 13.4 阶段 3：Query 优化（已在 §4.4 和 §9.3 中详述）

参见检索策略章节，不重复。

### 13.5 阶段 4：RAG 检索 + 空检索兜底

**检索流程**（已在 §4.4 和 §9.2 中详述）：

```
BM25 召回 30 + bge-m3 向量召回 30 → RRF 融合 → Top-K=10
```

**空检索兜底策略**（新增）：

```
检索结果为空 (chunks == [])
  │
  ├── Step 1: 扩大召回
  │     BM25 召回 60 + 向量召回 60 → RRF → Top-K=10
  │     ├── 有结果 → 使用
  │     └── 仍空 → Step 2
  │
  └── Step 2: 返回"未找到"
        直接回复用户，不调 LLM:
        {
          "decision": "无法判断",
          "reason": "规则文档库中未检索到与您问题相关的规则内容，无法进行审查判断。请确认是否已上传相关规则文档。",
          "evidence": [],
          "confidence": 0.0,
          "not_found": true
        }
```

```python
async def retrieve_with_fallback(query: str, retriever, top_k=10) -> dict:
    """检索 + 空结果兜底"""
    chunks = await retriever.retrieve(query, top_k=top_k)
    
    if chunks:
        return {"chunks": chunks, "not_found": False}
    
    # Step 1: 扩大召回
    logger.info(f"[检索兜底] 原始检索为空，扩大召回范围")
    chunks = await retriever.retrieve(query, top_k=top_k, 
                                       bm25_k=60, vector_k=60)
    
    if chunks:
        logger.info(f"[检索兜底] 扩大后找到 {len(chunks)} 条结果")
        return {"chunks": chunks, "not_found": False, "search_expanded": True}
    
    # Step 2: 完全无结果
    logger.info(f"[检索兜底] 扩大检索后仍为空，返回 not_found")
    return {"chunks": [], "not_found": True}
```

**SSE 输出**（空检索场景）：

```
event: message
data: {"type":"messageLabel","answer":"- <span>检索相关知识中...</span>"}

event: message
data: {"type":"messageLabel","answer":"- <span>扩大检索范围中...</span>"}

event: message
data: {"type":"content","answer":"{\"decision\":\"无法判断\",\"reason\":\"规则文档库中未检索到与您问题相关的规则内容...\",\"evidence\":[],\"confidence\":0.0,\"not_found\":true}"}

event: done
data: {"done":true}
```

### 13.6 阶段 5：LLM 生成（已在 §4.5 中详述）

参见 generator.py 设计，不重复。

**"文档中未找到相关规则"的判断**：

LLM 在审核检索到的 chunks 后，如果发现所有 chunks 都与用户问题不相关，应在输出中标记 `not_found: true`：

```json
{
  "decision": "无法判断",
  "reason": "检索到的规则文档内容中未找到与「冀北电价上限」相关的条款...",
  "evidence": [],
  "confidence": 0.0,
  "not_found": true,
  "tool_calls": []
}
```

Pipeline 检测到 `not_found: true` 时，**跳过 Tool 调用和 Judge 校验**，直接返回结果。

### 13.7 阶段 6：Tool 调用 + 终止条件（Phase 2）

Tool 执行流程已在 §8.6 中详述。本节补充**终止条件**。

**Tool 循环终止条件**：

| 条件 | 行为 |
|------|------|
| LLM 返回 `tool_calls: []` | 正常终止 → 进入 Judge 校验 |
| 达到 3 轮仍 `tool_calls` 非空 | 强制终止 → 进入降级处理 |
| 某轮 tool 执行全部失败 | 强制终止 → 进入降级处理 |
| 总 Tool 执行时间超 30 秒 | 强制终止 → 进入降级处理 |

**降级处理（3 轮后未解决）**：

```python
MAX_TOOL_ROUNDS = 3
TOOL_TOTAL_TIMEOUT = 30  # 秒

async def execute_with_tool_loop(generator, query, chunks):
    """带终止条件的 Tool 循环"""
    messages = build_initial_messages(query, chunks)
    tool_logs = []
    round_start = time.time()
    
    for round_num in range(1, MAX_TOOL_ROUNDS + 1):
        # 超时检查
        if time.time() - round_start > TOOL_TOTAL_TIMEOUT:
            logger.warning(f"[Tool] 总超时 {TOOL_TOTAL_TIMEOUT}s，强制终止")
            return await fallback_generate(generator, messages, 
                                           tool_unsolved=True, 
                                           reason="tool_timeout")
        
        llm_output = await generator.generate(messages)
        
        # 无 tool_calls → 正常结束
        if not llm_output.get("tool_calls"):
            return llm_output, tool_logs
        
        # 执行工具
        results = ToolExecutor.execute_tool_calls(llm_output["tool_calls"])
        tool_logs.extend(results)
        
        # 全部失败 → 降级
        if all(not r["result"]["success"] for r in results):
            logger.warning(f"[Tool] 第{round_num}轮全部工具失败，降级")
            return await fallback_generate(generator, messages,
                                           tool_unsolved=True,
                                           reason="all_tools_failed")
        
        # 注入结果，继续下一轮
        messages.append({"role": "tool", "content": json.dumps(results)})
    
    # 3 轮后仍未解决 → 降级
    logger.warning(f"[Tool] {MAX_TOOL_ROUNDS}轮后仍未解决，降级")
    return await fallback_generate(generator, messages,
                                   tool_unsolved=True,
                                   reason="max_rounds_exceeded")


async def fallback_generate(generator, messages, tool_unsolved=False, reason=""):
    """降级生成：让 LLM 基于已有信息做最好判断"""
    fallback_prompt = """
## 注意
工具调用未能完成所有计算。请基于目前已检索到的规则原文，
尽力做出最好的判断。如果信息不足以做出确定判断，请标记为"无法判断"。
"""
    messages.append({"role": "user", "content": fallback_prompt})
    output = await generator.generate(messages)
    if tool_unsolved:
        output["tool_unsolved"] = True
        output["tool_unsolved_reason"] = reason
    return output, []
```

**SSE 输出**（降级场景）：

```
event: message
data: {"type":"messageLabel","answer":"- <span>工具调用中（第3轮）...</span>"}

event: message
data: {"type":"messageLabel","answer":"- <span>工具未完成，降级为直接推理...</span>"}

event: message
data: {"answer":"{\"decision\":\"无法判断\",\"reason\":\"...\",\"tool_unsolved\":true,\"tool_unsolved_reason\":\"max_rounds_exceeded\"}"}
```

### 13.8 阶段 7：Judge 校验 + 异常兜底

正常流程已在 §4.6 和 §7.5 中详述。本节补充**异常兜底**。

**Judge 可能遇到的异常**：

| 异常 | 处理 |
|------|------|
| Judge API 调用超时（>60s） | 跳过校验，直接使用 LLM 原始结果，标记 `judge_skipped` |
| Judge API 返回错误（5xx） | 同上 |
| Judge 返回非 JSON 格式 | 重试 1 次 → 仍失败则跳过 |

```python
JUDGE_TIMEOUT = 60  # 秒

async def verify_with_fallback(judge, llm_output, query, chunks, tool_logs):
    """Judge 校验 + 异常兜底"""
    # 不需要校验的场景：直接跳过
    if llm_output.get("not_found"):
        logger.info("[Judge] 文档未找到，跳过校验")
        return llm_output

    try:
        result = await asyncio.wait_for(
            judge.verify(llm_output, query, chunks, tool_logs),
            timeout=JUDGE_TIMEOUT
        )
        
        # 校验返回格式
        if not isinstance(result, dict) or "verified" not in result:
            raise ValueError("Judge 返回格式异常")
        
        return result
    
    except asyncio.TimeoutError:
        logger.warning(f"[Judge] 超时 {JUDGE_TIMEOUT}s，跳过校验")
        llm_output["judge_skipped"] = True
        llm_output["judge_skipped_reason"] = "timeout"
        return llm_output
    
    except Exception as e:
        logger.error(f"[Judge] 校验失败: {e}，跳过校验")
        llm_output["judge_skipped"] = True
        llm_output["judge_skipped_reason"] = str(e)
        return llm_output
```

**SSE 输出**（Judge 跳过场景）：

```
event: message
data: {"type":"messageLabel","answer":"- <span>结果校验中...</span>"}

event: message
data: {"type":"messageLabel","answer":"- <span>校验服务繁忙，已跳过校验...</span>"}

event: message
data: {"answer":"{\"decision\":\"不符合\",\"reason\":\"...\",\"judge_skipped\":true}"}
```

### 13.9 阶段 7.5：Corrective-RAG 回环（Judge 触发扩大检索）

> 本节为设计文档 v2 的新增闭环，位于阶段 7（Judge 校验）与阶段 8（SSE 输出）之间。
> 背景：单向流水线中 Judge 检出幻觉/遗漏时只能在**已有证据范围内**删证据、改结论；
> 而 `missing_rules` 本身就是"context 里没有的东西"，Judge 无法在现状下自我修复。
> 回环让 Judge 的校验信号反向驱动检索层，形成 RAG 自纠错闭环（Corrective-RAG 思路）。

#### 13.9.1 触发条件

| 信号 | 判定 | 说明 |
|---|---|---|
| `judge_hallucinated` 非空 | `_should_run_corrective()` | 证据未能与原文匹配 → 上下文缺支撑 |
| `judge_missing_rules` 非空 | 同上 | 存在未检索到的关键规则 → 补检重点 |
| 任一为真即触发 | — | `judge_skipped` 时**不触发**；配置开关关闭时不触发 |

配置项 `RULE_REVIEW_CORRECTIVE_ENABLED`（默认 `true`，`src/config.py` 读取），
评测或延迟敏感场景可置 `false` 关闭。

#### 13.9.2 回环时序（最多 1 轮）

```
阶段7 Judge（第1轮）→ 检出幻觉/遗漏
  → 1. 构造补充 query：rewritten_query + missing_rules[].rule 文本（≤50字符×3条）
       （无 missing_rules 时改用幻觉证据的 section 标题关键词；与 query 互相包含的跳过；总长≤200）
  → 2. 二次检索：retrieve_with_fallback(corrective_query, top_k=min(top_k×2, 50))
  → 3. 与首轮结果按 chunk_id 合并去重（复用 _merge_retrieve_results），取前 min(top_k×2, 50)
  → 4. 带 Judge 反馈重新生成：generate(query=corrective_query, context_chunks=合并chunks,
       judge_feedback={hallucinated_evidence, missing_rules})
       —— 反馈仅描述问题不下定论，防 LLM 过度服从
  → 5. 二次校验：verify_with_fallback(judge, 第二轮LLM输出, rewritten_query, 合并chunks)
       —— 注意传 rewritten_query（Judge 必须针对用户原问题）
  → 6. 第二轮结果即最终输出，不再循环
```

#### 13.9.3 终止矩阵（第二轮结果一律为终判）

| 第二轮 LLM 结果 | 处理 | terminated_reason |
|---|---|---|
| 二次检索无结果 | 保留首轮 Judge 结果，不掩盖 | `second_retrieve_empty` |
| 二次检索异常 | 同上（warning 日志） | `second_retrieve_error` |
| 生成返回 None（服务故障） | 降级输出首轮结果 | `second_generate_none` |
| `not_found=True` | 输出第二轮 LLM 输出 + `judge_skipped=True, reason="not_found"` | `second_not_found` |
| 带 `tool_calls` | 1 轮预算内不重跑 Tool：清空 tool_calls 直接送 Judge | — |
| 二次校验正常 | 输出 `verify_with_fallback` 结果（含第二轮信号） | `""` |
| 二次校验跳过 | 输出其跳过结果 | `second_judge_skipped` |

#### 13.9.4 关键实现约定

- **合并证据**：首轮（多文档时为合并结果）+ 二次检索结果，按 `chunk_id` 去重、
  按 `score` 降序、截断 `min(top_k×2, 50)`；去重保留首次出现的副本。
- **幻觉证据不重查**：`hallucinated_evidence` 本身是错的，仅当其 section 标题
  （通常含条款号关键词）被用作补充关键词，证据正文不进入二次 query。
- **多文档场景**：二次检索走全局检索（不传 doc_filter），简化为全局补检。
- **审计**：`AuditRecord.corrective` 记录触发原因/补充 query/合并 chunk 数/第二轮
  校验结论/终止原因；`RetrievalAudit.retrieval_rounds`、`LLMGenerationAudit.rounds`、
  `JudgeAudit.rounds` 记录轮次（旧记录 load 时回填默认值，向后兼容）。
- **SSE**：新增 `re_retrieval` / `re_generation` / `re_judge` 进度标签，
  在回环结束后统一 flush（生成本就是非流式内部调用）。
- **评测适配**：`evaluation.py` 的幻觉检测与 RAGAS 上下文改为取**末次**
  （合并后）检索结果（`_latest_retrieval_chunks`），避免把二次检索证据误判为幻觉。

---

## 14. 异常处理策略

### 14.1 全局异常分类与处理

| 异常类型 | 触发条件 | 处理策略 | 用户看到的 |
|---------|---------|---------|----------|
| **LLM API 失败** | DashScope 5xx、网络超时 | 重试 2 次（间隔 2s/4s）→ 仍失败则返回友好提示 | "规则审查服务暂时不可用，请稍后重试" |
| **Judge API 失败** | DeepSeek API 5xx、超时 | 不重试，跳过校验，返回 LLM 原始结果 | 正常结果（只是未校验） |
| **嵌入模型加载失败** | bge-m3 下载超时、OOM | 启动时检查，失败时退出并提示原因 | N/A（系统不可用） |
| **PDF 解析失败** | 文件损坏、加密 PDF | 返回具体错误信息 | "文档解析失败：文件可能已损坏或加密" |
| **BM25 索引构建失败** | 内存不足 | 降级为仅向量检索 | 正常查询（只是少了 BM25 召回） |
| **Tool 执行异常** | 参数格式错误、除零 | 返回错误到 LLM，由 LLM 决定重试或跳过 | 正常结果（tool 失败但 LLM 兜底） |
| **Tool 沙箱异常** | 代码执行超时 10s | 强制终止，返回 timeout 错误 | 同上 |

### 14.2 降级链路（从最优到最差）

```
完整链路 （最优）
  RAG → LLM → Tool(3轮) → Judge
    ↓ Tool 失败
降级1: 跳过 Tool（无工具辅助）
  RAG → LLM → Judge
    ↓ LLM 失败 / 超时
降级2: 跳过 LLM（仅提示）
  返回 "服务暂时不可用"
    ↓ Judge 失败
降级3: 跳过 Judge（无校验）
  RAG → LLM → 返回（标记未校验）
    ↓ RAG 空结果
降级4: 直接回复
  回复 "未找到相关规则文档"
```

### 14.3 超时配置

| 超时项 | 默认值 | 说明 |
|--------|--------|------|
| 请求总超时 | 120s | 从收到请求到返回结果的硬上限 |
| LLM 推理超时 | 60s | Qwen3B 单次 API 调用 |
| Judge 校验超时 | 60s | DeepSeek-v4 单次 API 调用 |
| Tool 总执行超时 | 30s | 所有轮次 Tool 执行的累计时间 |
| 单次 Tool 执行超时 | 10s | 单个工具函数的沙箱执行上限 |
| PDF 解析超时 | 120s | 上传大 PDF 时的解析时间上限 |
| 嵌入向量生成超时 | 60s | 单次 batch embedding 时间上限 |

### 14.4 日志记录规范

```python
# 每个请求的结构化日志（复用现有 logging_setup 的 trace_id 机制）
request_log = {
    "trace_id": "uuid",
    "query_id": "q-xxx",
    "rewritten_query": "...",
    "clarification_needed": False,
    "split_count": 1,
    "retrieval": {
        "bm25_count": 30,
        "vector_count": 30,
        "final_count": 10,
        "search_expanded": False,
    },
    "not_found": False,
    "llm": {
        "model": "qwen3-max",
        "latency_ms": 2500,
        "tok_count": 450,
    },
    "tool": {
        "rounds": 2,
        "tools_called": ["extract_table_data", "arithmetic_compare"],
        "failed": False,
        "total_latency_ms": 45,
    },
    "judge": {
        "model": "deepseek-v4",
        "skipped": False,
        "latency_ms": 1800,
    },
    "total_latency_ms": 5200,
    "degradation_level": 0,  # 0=完整链路，1=跳过Tool，2=跳过Judge，3=跳过LLM
}
```

---

## 附录：Phase 1 vs Phase 2 对比

| 维度 | Phase 1（MVP） | Phase 2（+Tool） |
|------|---------------|-------------------|
| 检索 | bge-m3 + BM25 + RRF | 同左，实验 bge-m3 sparse 替代 BM25 |
| LLM 推理 | Qwen3B API，纯文本推理 | Qwen3B API + Tool Calling |
| 表格处理 | LLM 直接读 Markdown table（有误差） | `extract_table_data` 工具（精确） |
| 数值比较 | LLM 直接判断（有概率出错） | `arithmetic_compare` 工具（100% 准确） |
| 规则引用 | LLM 只能看到检索到的 chunks | `resolve_cross_reference` 自动追踪引用 |
| 时效性 | LLM 可能忽略时间维度 | `validate_date_applicability` 强制校验 |
| 单位 | LLM 可能混淆单位 | `unit_converter` 统一单位后比较 |
| Judge | DeepSeek-v4 校验 | DeepSeek-v4 校验 + Tool 日志交叉验证 |


---

# 第二部分 工作流程详解(原 rule-review-workflow.md)

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

**如果 Judge 检出了幻觉或遗漏**(`hallucinated_evidence` / `missing_rules` 任一非空,且开关 `RULE_REVIEW_CORRECTIVE_ENABLED=true`),不再直接输出,而是进入 **Corrective-RAG 回环**(阶段 7.5,最多 1 轮):

```text
Judge 检出问题
  → 1. 构造补充 query:原改写 query + missing_rules[].rule 文本
       (无遗漏时用幻觉证据的 section 标题关键词;互相包含的跳过;总长≤200)
  → 2. 二次检索:retrieve_with_fallback(corrective_query, top_k=min(top_k×2, 50))
  → 3. 与首轮结果按 chunk_id 合并去重(复用 _merge_retrieve_results)
  → 4. 带 Judge 反馈重新生成:generate(query=corrective_query, context_chunks=合并chunks,
       judge_feedback={hallucinated_evidence, missing_rules})   # 反馈只描述问题,不下定论
  → 5. 二次校验:verify_with_fallback(judge, 第二轮输出, rewritten_query, 合并chunks)
  → 6. 第二轮结果即最终输出(终止矩阵见设计文档 §13.9.3,不再循环)
```

**终止要点**:二次检索为空/生成失败 → 降级输出首轮结果;第二轮 LLM 判 `not_found` → 如实输出
`judge_skipped` 不掩盖;第二轮带 `tool_calls` → 预算内不重跑 Tool 直接送 Judge。
**评测适配**:离线评估的幻觉检测与 RAGAS 上下文改为取**末次(合并后)检索结果**,
避免把二次检索的证据误判为幻觉。

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

# 触发 Corrective-RAG 回环时,阶段 7 与阶段 8 之间追加:
data: {"type":"messageLabel","answer":"- <span>补充检索中...</span>","stage":"re_retrieval"}

data: {"type":"messageLabel","answer":"- <span>重新推理中...</span>","stage":"re_generation"}

data: {"type":"messageLabel","answer":"- <span>二次校验中...</span>","stage":"re_judge"}

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


---

# 第三部分 优化方向清单(原 rule-review-optimization.md)

> 用途:面试追问「这个项目还可以如何优化?添加哪些能力、功能、工具?」的回答素材。
> 状态:方向一、方向六、方向九、方向十已落地实施;其余方向为话术与规划。
> 关联:详细设计见本总文档第一部分,流程见第二部分;测试 `tests/test_rule_review_evaluation.py`、`tests/test_rule_review_observability.py`、`tests/test_rule_review_llm_judge_metrics.py`、`tests/test_rule_review_corrective.py`。

---

## 回答框架(开场 30 秒)

「我盘点过这个项目的优化空间,按**面试官视角**排了个优先级:先补可复现的指标(评估闭环)、再补可观测的底座(延迟分位数/审计字段)、然后是体验层(真流式/多轮)、最后是规模化(Milvus/PG 接线)和前沿探索(GraphRAG)。其中评估闭环、可观测性、RAGAS 风格的 LLM-as-judge 评测和 Judge 触发的自纠错回环我已经落地了,后面是规划中的二期。」

- 已落地:✅ 方向一(评估体系闭环)、方向六(可观测性)、方向九(RAGAS 风格 LLM-judge 评测)、方向十(Corrective-RAG 自纠错回环)
- 规划中:方向二~五、七、八

---

## 方向一:评估体系闭环 ✅(已实施)

**定位**:让「Top5 召回 65%→88%」从简历口号变成可复现的评测报告。

**现状缺口**:`data/evaluation/` 目录原本不存在;`evaluation.py` 的 EvalRunner 拿不到检索文本(`retrieved_texts` 恒为空),幻觉检测对空检索集把所有 evidence 全部误判为幻觉。

**怎么做(已落地)**:
1. `data/evaluation/test_cases.json` 补 14 条结构化测试集(价格上限/日内120%/跨省规则/多文档/澄清/未命中,难度分层);
2. pipeline 检索阶段日志透传 `retrieved_chunks`(文本截断 200 字符);
3. 修复幻觉检测空检索 bug:检索集为空时跳过检测并标记 `hallucination_check_skipped`,不再误报;
4. 新增检索层指标 `recall@k` 与 `MRR`(相关性代理:chunk 文本含期望关键词),进入 EvalReport。

**预期指标**:评测集 0→14 条;幻觉误报归零;产出决策准确率 / recall@k / MRR / 幻觉率四类基线。

**解析分层后的 98% 双评测口径**(见 §4.3):评测集扩至 18 条,其中 tc-015~018 为表格检索类用例
(tag=「表格检索」,关键词取表格单元格值,如「760」「四川主网」);解析精度评测
(`python -m src.rule_review.parsing_eval`,表格单元格识别准确率,手动入库 100%/MinerU ≥98%)与
检索召回评测(`avg_recall_at_k ≥ 0.98`)双断言落地,阈值常量为 `TABLE_CELL_ACCURACY_TARGET` /
`RETRIEVAL_RECALL_TARGET`。

**面试话术**(约 30 秒):

> 我的评估体系分三层:离线评测集、指标定义、在线回归。指标上我不只看决策准确率,还加了 recall@k 和 MRR 衡量检索层——这层指标能直接量化「65%→88%」是怎么来的。幻觉率用证据文本对检索文本的最长公共子串近似匹配检测。这里我修过一个真实 bug:评估器拿不到检索结果,空集合把所有 evidence 误判成幻觉,把检索结果透传进去后误报归零。每次迭代跑全量评测、报告 JSON 归档,保证改检索不伤决策、改 Prompt 不伤召回。

**追问预案**:
| 追问 | 回答要点 |
|---|---|
| 评测集多少条、谁标的? | 14 条结构化用例,覆盖 7 类场景,字段含标准答案与期望关键词;标注口径来自规则文档与专家复核记录。规模小是事实——定位是方案对比的 A/B 信号,最终验证靠线上数据 |
| 相关性为什么用关键词代理? | chunk_id 是随机生成无法预标注,用「期望关键词命中」做相关性代理;内容哈希 chunk 后可升级为 expected_chunk_ids 精确标注 |
| 幻觉检测的 LCS 阈值为什么是 0.3? | 证据文本对检索文本的最长公共子串占比低于 0.3 视为无依据;阈值按抽样人工核对校准,宁过勿漏(漏幻觉比误报危害大) |
| recall@k 和 MRR 的区别? | recall@k 看是否召回(0/1),MRR 看排第几(倒数排名),两者一起反映「召回全」和「排得前」 |

---

## 方向二:Judge 接线落地(LLM-as-Judge)

**定位**:把已实现但从未运行的 Judge 校验接进运行路径,是「LLM-as-Judge 评测标准」高频考点最硬的素材。

**现状缺口**:`pipeline.py` 构造器 `judge=None` 默认不挂;两处 `verify_with_fallback` 未传 `tool_logs`(Judge 看不到工具执行过程);`pipeline.py:625` 引用未导入的 `ToolCallLog`,工具审计写入被 except 静默吞掉。

**怎么做**:`is_judge_configured()` 门控(配置校验模型才挂载);`get_default_pipeline()` 惰性注入;两处调用补 `tool_logs`;修复 `ToolCallLog` import。降级语义不变:任何异常 → judge_skipped,绝不阻塞主流程。

**面试话术**:

> Judge 是 LLM-as-Judge 思路:大模型当裁判,校验小模型输出的三件事——幻觉、逻辑自洽、规则遗漏。我把它接进 pipeline 阶段 7,降级做得很重:超时、API 错误、格式异常一律跳过校验,绝不阻塞主流程。tool 调用日志也传给 Judge,让它知道结果是怎么算出来的,不是空判断。配置了校验模型才挂载,线上可 shadow 观察,效果稳定再全量。

**追问预案**:「Judge 用什么模型?」→ 独立配置 `JUDGE_MODEL`,与被审模型可不同(强模型审弱模型);「怎么保证不拖慢?」→ 60s 超时兜底 + shadow 模式先行。

---

## 方向三:基础设施全面接线(Milvus / PostgreSQL / Redis / vLLM / 沙箱)

**定位**:把 Phase 3/4 写完但零引用的组件接进运行路径,讲「本地零依赖可跑 vs 生产一键切换」的工程故事。

**现状缺口**:`milvus_store.py`、`pg_store.py`、`cache.py`、`local_model.py`、`SparseRetriever`、`ToolSandbox` 除 `__init__.py` 导出外零引用,运行路径仍是 FAISS + JSON 文件 + 内存单例。

**怎么做**:环境变量开关(`RULE_REVIEW_STORAGE=memory|milvus`、`RULE_REVIEW_CACHE_ENABLE`)按配置选实现;文档变更时按命名空间全量失效缓存保证一致性;PG 审计表落地(替换 JSON 文件)。

**面试话术**:

> Milvus、PG、Redis、vLLM 我都实现好了并且带降级,但默认走 FAISS+JSON+内存,保证本地零依赖能跑。接线是环境变量一键切换:embedding 索引换 Milvus、审计落 PG、检索和 LLM 结果走 Redis,文档删除时按命名空间全量失效缓存。我清楚每一层切换的失效边界——这是生产系统最怕讲不清的地方。

**追问预案**:「为什么不全量切换?」→ 分布式组件有部署成本与运维风险,小规模 FAISS 足够,预留开关让规模驱动选择;「缓存失效怎么保证一致性?」→ 文档变更 → 全量失效该文档命名空间缓存(当前 cache.py 的 `invalidate_document` 已实现)。

---

## 方向四:真流式 SSE + 澄清多轮闭环

**定位**:把「伪流式」(阶段标签 + 一次性 JSON)升级为 token 流,让澄清从「一问终止」变成多轮闭环。

**现状缺口**:`pipeline.py` 生成阶段调非流式 `generator.generate`;`generate_stream` 从未被调用;`needs_clarification` 即终止,`RuleReviewRequest.sessionId` 从未读取。

**怎么做**:`execute_stream` 生成阶段改消费 `generate_stream`,逐 token yield 同时累积全文,结束后再走 JSON 解析与 Judge 校验,解析失败自动回退非流式;澄清响应带 sessionId,前端追问后带上下文重跑。

**面试话术**:

> 现在的 SSE 是阶段标签加一次性 JSON,我改成真 token 流,首包时间降到 200ms 内。澄清也不是一问就终止,而是带 sessionId 多轮追问、把追问结果回填 query 重跑。关键设计是流式与后置校验的配合:token 边流边攒,结束后才做 JSON 解析和 Judge 校验,解析失败自动回退非流式——体验和正确性不互斥。

**追问预案**:「首 token 延迟怎么降?」→ 检索与 LLM 首包解耦,阶段标签先发,生成阶段逐 token 推流。

---

## 方向五:多轮对话记忆落地(查询链路)

**定位**:把查询链路的空壳 session 管理接上,正面回答「会话持久化」考点。

**现状缺口**:`session_manager.py:80 get_or_create_context` 全库零调用;每个请求新建 WorkflowRouter 和空 InMemorySaver,跨请求上下文为零;`conversation_id` 只用于日志追踪。

**怎么做**:按 sessionId 复用 checkpointer;前几轮问答摘要注入改写阶段;token 预算与过期回收;查询链路先补最小单元测试再接线(该链路当前 0 测试)。

**面试话术**:

> 记忆落地我分三层:存储、注入、回收。按 sessionId 复用 checkpointer,前几轮对话摘要注入当前 query 的改写阶段,超预算做压缩、过期即回收防止串 session。核心结论是记忆必须按 session 隔离,不能全局共享——这也是多智能体并发场景最常踩的坑。

---

## 方向六:可观测性 ✅(已实施)

**定位**:给「月调用 1 万+、好评率 91%」一个可审计的观测底座,让每个指标在日志里有出处。

**现状缺口**:`base_workflow.py:1653` 残留调试 print 把意图识别结果直打 stdout(数据泄露风险);`AuditRecord` 的 `model`/`tok_input`/`tok_output` 与 `retrieval.sparse_k` 从未填充;无延迟分位数统计;无结构化日志。

**怎么做(已落地)**:
1. 清理残留调试输出:删除 `print(post_processing, intent_result)`,两处 `traceback.print_exc()` 改为 `logger.exception`;
2. 新增 `src/rule_review/observability.py`:`LatencyStats` 按阶段记录耗时,计算 P50/P95/均值(线程安全,可 JSON 持久化);
3. pipeline 接线:检索/生成/Judge/总耗时四段写入统计;审计补齐 `model`、`tok_input/tok_output`(generator 从 `usage_metadata` 容错透传)、`sparse_k`(未接线如实记录 0)、各阶段 `latency_ms`;
4. 新增 `GET /v1/rule-review/observability/latency` 端点;
5. `LOG_FORMAT=json` 结构化日志开关(`JsonFormatter`,extra 字段 stage/latency_ms/query_id 并入 JSON),默认 text 保持现状。

**面试话术**:

> 月调用上万没观测等于盲跑。我做了三件事:全链路 query_id 串日志、分阶段延迟分位数、审计字段完整落库——模型名、token 数、检索各通道召回数、每阶段耗时。它回答三个问题:哪个阶段最慢、幻觉率趋势、某一单用户体验差的根因。好评率 91% 这个数字,我坚持要求它在日志里有出处。

**追问预案**:
| 追问 | 回答要点 |
|---|---|
| 分位数为什么用 P95 不用平均值? | 平均值被长尾污染,客服场景用户感知的是尾延迟,监控 P95 才能暴露劣化 |
| 顺带修复的 bug? | 非流式路径 `judge_audit` 变量未定义(有 Judge 时审计写入被静默吞掉),补了初始化;`_chunks_to_dict_list` 的装饰器错位也一并修正 |
| token 数怎么拿到的? | 从模型响应的 `usage_metadata` 容错读取,不同模型暴露位置不一,拿不到保持 0 不报错 |

---

## 方向七:检索增强(LLM 多 query 生成、GraphRAG、时间维度过滤)

**定位**:召回率突破点,对接「查询重写」「GraphRAG」两个高频考点。注意演进口径:
**88% 是纯混合检索(查询端优化前)的召回,解析分层落地后表格类召回达 98%**(见 §4.3 解析分层),
方向一/方向五的查询端优化继续把 98% 推向更高。

**现状缺口**:设计文档 §12.3 规划的「LLM 多 query 生成」未实现;检索只有 RRF 混合两路;无时间维度裁剪。

**怎么做**:改写阶段生成 2-3 个 query 变体并行检索后合并去重;时间解析前置过滤(chunk 元数据带时间范围);GraphRAG 抽取条款引用关系做一跳扩展;每个模块用方向一的 recall@k 离线验证增量。

**面试话术**:

> 混合检索到 88% 之后,我先在数据端做了解析分层:相对不重要的文档走本地 MinerU 解析,重要的政策问答表格手动入库,表格单元格识别准确率到了 100%,检索召回整体到 98%——解析精度是召回的地基。然后在查询端和结构端继续做:LLM 多 query 生成并行检索合并、时间维度先裁剪再检索、GraphRAG 把条款引用关系建成图做一跳扩展。每个模块上线前必须用离线 recall@k 验证增量,没有指标提升就不上线——这是我做检索迭代的原则。

**追问预案**:「GraphRAG 和向量检索的区别?」→ 向量检索对实体关联、多跳问题天然弱(「引用 A 的条款 X 是否也适用于场景 B」),GraphRAG 显式建模引用关系,一跳扩展可命中;代价是建图与维护成本,适合规则文档这种结构稳定、引用密集的语料。

---

## 方向八:工具系统增强 + 代码质量债

**定位**:面试前的「地基工程」,让所有指标可复现、可解释。

**现状缺口**:`tool_executor.py` TOOL_MAP 硬编码 5 个工具,LLM 的 tool_calls 无 JSON Schema 校验直接 `func(**args)`;聚合决策逻辑三份重复且语义有细微差异;查询链路 0 测试;残留死代码(`_can_share_api_call` 恒 False 等)。

**怎么做**:tool 参数 JSON Schema 校验(非法参数进函数前拦截);注册表配置驱动(加工具不改代码);三份决策逻辑合并为单一纯函数 + golden 测试锁行为;清死代码;查询链路补冒烟测试。

**面试话术**:

> 我把工具调用做了 JSON Schema 校验,非法参数在进入函数前就被拦截;注册表改成配置驱动,加工具不改代码。同时清掉了几百行死代码和一处把内部意图打 stdout 的调试 print——质量债不清理,前面所有指标都不可复现,这是我给项目上的第一课。

---

## 方向九:RAGAS 风格自动化评测指标(LLM-as-Judge)✅(已实施)

**定位**:面试「RAG 怎么自动化评测?」「LLM-as-judge 怎么落地?」「RAGAS 了解吗?」三件套的高频考点,把评估体系从「规则式指标」升级为「RAGAS 四指标」——检索质量和生成质量都有了 LLM 视角的量化口径。

**现状缺口**:方向一的评估只有规则式指标(decision_accuracy / keyword_recall / recall@k / MRR / LCS 幻觉检测),没有生成质量(faithfulness、答案相关性)与检索质量(context precision/recall)的语义级度量——关键词命中 ≠ 事实正确,需要 LLM 当裁判。

**怎么做(已落地)**:
1. 新增 `src/rule_review/llm_judge_metrics.py`,自研 RAGAS 四指标(**不引入 ragas 库**,可讲原理):
   - **Faithfulness(忠实度)**:从回答抽取 claims,逐条验证是否被检索 context 支持,分数 = 受支持 claims / 总 claims;claim 抽取与验证合并为**一次** LLM 调用(在 prompt 中要求同时输出 claims 与逐条 supported);
   - **Answer Relevancy(答案相关性)**:LLM judge 按 rubric 直接打分 0-1(官方做法是「由回答反生成若干问题 + 嵌入相似度」,需加载 bge-m3 约 2GB,当前为简化版,升级路径是复用 `DocumentStore.embedding_fn`);
   - **Context Precision(上下文精度)**:逐 chunk 判定与 question 的相关性,按 RAGAS 口径 `Σ(P@k × rel_k) / Σ rel_k` 加权,衡量「相关 chunk 是否排得靠前」;
   - **Context Recall(上下文召回)**:参考答案逐句判定是否被 context 支持,衡量「检索全不全」;
2. pipeline 检索 stage 新增 `retrieved_chunks_full`(完整 chunk 文本透传,原 200 字符截断保留),保证 LLM 判分不因截断失真;
3. `evaluation.py` 集成:`TestCase` 加可选 `reference_answer` 字段(缺省时 precision/recall 降级 None,旧测试集零破坏)、`EvalMetrics/EvalReport` 加四指标与聚合、`EvalRunner` 支持 `judge_model` 注入与 `ragas_enabled` 开关、异常全降级不阻断主流程;
4. 新增 CLI:`python -m src.rule_review.evaluation --ragas --top-k 10`,报告落盘 `data/evaluation/reports/`,摘要打印含降级计数;
5. 测试集 5 条代表性用例补 `reference_answer`,澄清/未命中用例故意不填演示降级路径。

**预期指标**:每用例 ≤5 次 LLM 调用(四指标中 claim 抽取+验证合并),14 条全量 ≈ 56-70 次;阈值建议(需按人工抽样校准口径):faithfulness ≥ 0.8 / answer_relevancy ≥ 0.7 / context_precision ≥ 0.7 / context_recall ≥ 0.8。

**面试话术**(约 30 秒):

> 我做了一套 RAGAS 风格的自动化评测,四个指标覆盖两个层面:检索层看 context precision 和 context recall——相关 chunk 排得靠不靠前、该召回的全不全;生成层看 faithfulness 和 answer relevancy——回答有没有忠实于检索到的规则原文、有没有跑题。实现上我没引 ragas 库,自己写的 LLM-as-judge:claim 抽取和验证合并成一次调用控制成本,每用例不超过 5 次调用。评测模型我特意走 httpx 直连而不是 langchain 的 ChatQwen——后者会拉进 torch,和 faiss 的 OpenMP 运行库在 macOS 上冲突,进程直接 abort,这个坑我踩过。所有指标都做了降级:缺参考答案、检索为空、模型调用失败,该指标就置 None 并标注,绝不阻断评估主流程。

**详细版面试话术**(约 2-3 分钟,面试官问「评测是怎么做的/评测数据集怎么做的/评测了哪些指标」时用;30 秒版是它的压缩骨架):

> **① 整体机制(开场)**:我们这个规则审查系统的评测,走的是离线评估闭环。我先说机制:我在 `evaluation.py` 里写了一个 `EvalRunner`,它对测试集里每条用例跑一遍完整的九阶段 pipeline——从问题改写一直到生成、Judge 校验,拿到的都是真实输出——然后从结果里自动提取 decision、evidence、reason 和检索阶段的 chunks,逐条计算指标,最后聚合成一份报告,按难度和标签分组统计,JSON 落盘到 `data/evaluation/reports/`。整个评估一键触发:`python -m src.rule_review.evaluation`,不依赖线上环境。
>
> **② 数据集怎么做的**:测试集是我手工构造的,14 条用例放在 `data/evaluation/test_cases.json`。每条用例是一个字典,包含问题、期望决策、期望关键词、期望引用的规则文档、难度分级和标签。构造时我有意覆盖三类决策——符合、不符合、无法判断,其中「无法判断」对应澄清和检索未命中这两条关键分支路径;还覆盖了多文档拆分场景(涉及现货、中长期、监管办法多份规则)、边界数值(价格上限是 760 元/MWh,我各做了一条 800 元超限和一条 500 元未超限的用例),以及实体变体——冀北、山西、河北南网,同一条规则换不同主体去测泛化能力。
>
> **③ 指标(两层)**:指标我分了两个层面。第一层是规则式指标,纯代码计算、零 LLM 成本:决策准确率,就是预测的符合/不符合/无法判断和标准答案做去标点空格后的精确匹配;关键词召回和证据来源召回,检查回答和引用里有没有覆盖期望关键词和期望规则文档;检索层有 recall@k 和 MRR,用「chunk 文本是否包含期望关键词」做相关性代理,衡量检索排名的质量;还有幻觉检测,用 LCS 最长公共子串做近似匹配,evidence 里的文本在检索结果里匹配率低于 0.3 就判为幻觉,这个能抓出模型编造的内容。第二层是 RAGAS 风格的 LLM-as-judge 指标,覆盖生成质量和检索质量的语义维度:Faithfulness 忠实度,把回答拆成 claims,逐条验证是否被检索到的规则原文支持;Answer Relevancy 看回答有没有切题;Context Precision 按 RAGAS 口径的 Σ(P@k×rel_k)/Σrel_k,衡量相关 chunk 排得靠不靠前;Context Recall 把参考答案逐句拆开,看句子有没有被检索结果支撑,衡量检索全不全。
>
> **④ 工程细节与权衡**(主动讲,体现深度):RAGAS 四指标是我自研的,没有引入 ragas 库——库是黑盒,自研可以完全掌控 prompt 和降级语义。成本控制上,我把 claim 抽取和逐条验证合并成一次 LLM 调用,每条用例最多 5 次调用,14 条全量约 60 次。评测模型我特意走 httpx 直连的 ProxyChatModel,不用 langchain 的 ChatQwen——后者会拉进 torch,在 macOS 上和 faiss 的 OpenMP 运行库冲突,进程直接 abort,这个坑我实际踩过;而且评测模型和生成模型独立配置,temperature 固定 0,保证可复现。所有指标都做了降级:缺参考答案、检索为空、模型调用失败,该指标就置 None 并标注原因,绝不阻断评估主流程——因为这是批量场景,所以 LLM 失败不重试,和线上 Judge 的「重试一次」策略有意区分,避免成本线性放大。
>
> **⑤ 不足与演进**(主动暴露短板):这套体系也有明确的薄弱点——测试集只有 14 条,覆盖面有限,后续可以做基于规则文档的自动化生成和人工审核;Answer Relevancy 我目前用的是简化版(直接打分),官方做法是反生成问题加嵌入相似度,需要加载 bge-m3,这是我的升级路径;另外 `expected_chunk_ids` 字段已经预留了,等 chunk 内容哈希稳定之后,检索指标可以从关键词代理升级成精确的 chunk 命中标注。我的整体原则是:每个优化必须可测量,离线指标或线上观测至少有一个能证明它,没有增量验证就不上线。

**追问预案**:
| 追问 | 回答要点 |
|---|---|
| 为什么不用现成的 ragas 库? | 库是黑盒,面试讲不透原理;自研可完全掌控 prompt 与降级语义;换指标口径只需改 prompt,不升级依赖 |
| Answer Relevancy 为什么简化成直接打分? | 官方做法「反生成问题 + 嵌入相似度」需要加载 bge-m3(约 2GB),且反生成本身也是一次 LLM 调用;打分版与「自研 LLM-as-judge」主题一致,升级路径已写在模块 docstring |
| 为什么 claim 抽取和验证合并成一次调用? | 两次调用会让成本翻倍(14 条 × 2);合并后让模型同时输出 claims 和逐条 supported,一次搞定,代价是 prompt 稍长 |
| LLM judge 和规则式指标的分工? | 规则式(keyword_recall/LCS 幻觉)快、零成本、可回归;LLM 指标是语义级、慢但准;两者互补——规则指标做每日回归,LLM 指标做迭代验收 |
| 评测模型为什么不能和生成模型同一个? | 判分模型与被评模型同源会有系统性偏差;评测走独立 `JUDGE_MODEL` 配置(强模型审弱模型),temperature=0 保证可复现 |
| 70 次调用成本怎么办? | 离线批量、串行 5-10 分钟可接受;指标结果带 `judge_latency_ms` 归因;后续可做调用级缓存与并发 |
| 为什么评测模型走 httpx 不走 ChatQwen? | ChatQwen → langchain_qwq → torch,自带 libomp.dylib,与 faiss 的 OpenMP 运行库冲突,同进程先加载后 faiss.search 直接 abort(踩过的坑);ProxyChatModel 纯 httpx,无此问题 |

---

## 方向十:Corrective-RAG 自纠错回环(Judge 触发扩大检索)✅(已实施)

**定位**:面试「RAG 遇到幻觉/证据不足怎么办?」「知道 Corrective-RAG / Self-RAG 吗?」「为什么 Judge 校验后不重新检索?」的高频考点,把「Judge 只校验不纠错」升级为「Judge 驱动检索层自纠错」的闭环。

**现状缺口**:单向 9 阶段流水线中,Judge 检出幻觉(`hallucinated_evidence`)或遗漏(`missing_rules`)时只能在**已有证据范围内**删证据、改结论——而 `missing_rules` 本身就是"context 里没有的东西",Judge 永远无法在现状下自我修复。评测集里「证据缺失导致结论不稳」的用例只能靠检索层参数硬调。

**怎么做(已落地)**:
1. **触发条件**:`_should_run_corrective()`——开关 `RULE_REVIEW_CORRECTIVE_ENABLED`(默认 true,可关)+ 非 `judge_skipped` + 幻觉/遗漏任一非空;
2. **补充 query 构造**(`_build_corrective_query`):rewritten_query + `missing_rules[].rule` 文本(≤50 字符 ×3 条,互相包含跳过,总长 ≤200);无遗漏时改用幻觉证据的 **section 标题**做关键词——幻觉证据本身是错的**不重查**,仅取其条款定位信息补检;
3. **二次检索 + 合并**:`retrieve_with_fallback(corrective_query, top_k=min(top_k×2, 50))`,与首轮按 `chunk_id` 去重合并(复用 `_merge_retrieve_results`);
4. **带反馈重新生成**:`generate(..., judge_feedback={hallucinated_evidence, missing_rules})`——反馈段只描述问题不下定论,防 LLM 过度服从;`generator` 幂等可安全二次调用;
5. **二次校验**:`verify_with_fallback(judge, 第二轮输出, rewritten_query, 合并chunks)`——传原改写 query 而非补充 query,保证 Judge 针对用户原问题;第二轮结果即终判,**最多 1 轮**;
6. **终止矩阵**:次轮检索空/生成失败 → 降级输出首轮结果不掩盖;次轮 LLM 判 not_found → 如实输出;次轮带 tool_calls → 预算内不重跑 Tool 直接送 Judge;
7. **可观测与评测适配**:审计新增 `corrective` 详情 dict 与各审计模型 `rounds` 字段(旧记录 load 向后兼容);SSE 新增 `re_retrieval/re_generation/re_judge` 标签;`evaluation.py` 幻觉检测与 RAGAS 上下文改为取**末次(合并后)检索结果**,避免把二次检索证据误判为幻觉。

**预期指标**:触发回环的请求(幻觉/遗漏被二次检索修正)幻觉率下降、evidence 覆盖率上升;代价是 p50 延迟增加一次检索+生成+校验(约 1.5-3 秒);评测集新增 corrective 场景用例后可量化「修正率 = 第二轮 verified / 触发数」。

**面试话术**(约 30 秒):

> 我把 Judge 从「只校验不纠错」升级成了「校验驱动纠错」的闭环。触发条件很明确:Judge 检出幻觉证据或遗漏规则,且校验没有跳过。触发后我先用遗漏的规则文本构造补充 query 去二次检索——注意幻觉证据本身是错的,我不会重查它,只用它的章节标题做定位关键词——然后和首轮结果按 chunk 去重合并,再带着 Judge 的反馈让 LLM 重新生成一轮,最后再过一遍 Judge,第二轮结果就是终判,最多一轮,不会无限循环。整个回环有完整的终止矩阵:二次检索没结果就保留首轮结果,不掩盖;重新生成失败就降级;审计里记录了触发原因、补充 query、合并了多少 chunk、第二轮校验结论,SSE 上也透出"补充检索中"这类进度标签。评测侧我也做了适配:幻觉检测改用合并后的证据集,不然二次检索补回来的证据会被误判成幻觉。

**详细版面试话术**(约 2 分钟,面试官问「Judge 发现问题后系统会怎么处理?」时用):

> **① 为什么做这个**:原来的流水线里 Judge 是最"憋屈"的一环——它发现了 evidence 是编的、发现了有规则没被引用,但只能在自己手头的证据里删删改改,`missing_rules` 这个东西本质上是"context 里不存在的内容",Judge 永远补不回来。这就是典型的 RAG 单程流水线缺陷:检索错了,后面全错,而且没有反馈回路。Corrective-RAG 的思路就是把 Judge 的信号反向喂给检索层。
>
> **② 触发与执行**:触发条件是 Judge 检出幻觉或遗漏任一非空,并且校验没跳过。执行分五步:第一步构造补充 query,把遗漏的规则文本拼进原问题——最多三条、每条五十字、和原问题重复的跳过;第二步二次检索,top_k 放大两倍;第三步和首轮结果按 chunk_id 合并去重;第四步带 Judge 反馈重新生成,反馈里只描述"漏了什么、哪条证据没对上原文",不下结论,防止 LLM 过度服从;第五步再过一遍 Judge,针对用户原问题校验,第二轮结果就是终判。
>
> **③ 工程细节与权衡**:第一,幻觉证据不重查——它本身是错的,重查等于给它第二次机会,我只取它的章节标题做定位关键词,因为章节标题里通常带着条款号;第二,最多一轮,第二轮无论结果如何都直接输出,不会陷入循环,最坏情况延迟翻倍但可控,而且有开关可以整体关掉;第三,全链路有终止矩阵——二次检索空、生成失败、判 not_found、Judge 跳过,每个分支都有明确的输出策略,绝不因为回环失败而吞掉首轮结果;第四,可观测性跟上,审计里记录触发原因、补充 query、合并 chunk 数、第二轮校验结论,评测侧幻觉检测改用合并后的证据集。
>
> **④ 不足与演进**:目前的二次检索 query 是启发式拼接,没让 LLM 参与生成检索词;回环只做了一轮,极端情况可能需要多轮迭代(代价是延迟);评测集里还没有专门的 corrective 场景用例,「修正率」这个指标还没有量化基线——下一步是补用例、跑出修正率数字,再考虑用 LLM 生成补充检索词。

**追问预案**:
| 追问 | 回答要点 |
|---|---|
| 为什么最多只做一轮? | 每轮增加一次检索+生成+校验,延迟约 1.5-3 秒;一轮能修正绝大多数"漏检索"场景;多轮收益递减且延迟不可控,后续可按轮数上限做成配置 |
| missing_rules 里的 source 是文档名,怎么定位 chunk? | 不反查 chunk——直接用 rule 文本关键词进检索层,靠 BM25+向量去召回对应条款;source 只作为审计信息 |
| 幻觉证据为什么不重查? | 它是 LLM 编的,重查等于给它第二次机会;真正缺的是遗漏的规则,所以 missing_rules 优先,幻觉只取 section 标题做定位关键词 |
| 怎么防止 LLM 第二轮过度服从反馈? | 反馈只描述问题不下结论(不写"第5条是答案");第二轮 Judge 仍独立校验,修正后的结果要再过一次校验关 |
| 和 Self-RAG / Corrective-RAG 的关系? | 都是"检索-生成-自省-纠错"闭环思路;Self-RAG 在生成中带反思 token,Corrective-RAG 用 Judge 信号触发二次检索,本项目是后者,触发信号来自幻觉检测与遗漏检查 |
| 评测侧怎么防止二次检索证据被误判幻觉? | evaluation.py 的幻觉检测和 RAGAS 上下文改为取末次(合并后)检索 stage(`_latest_retrieval_chunks`),而不是第一个 retrieval stage |

---

## 附:可添加的新工具候选(面试问「你想加什么工具」时用)

| 工具 | 业务动机 | 实现要点 |
|---|---|---|
| 历史裁决案例检索 | 相似条款/案例一跳召回,支撑「依据参考」 | 案例库向量化,复用现有混合检索 |
| 规则版本对比 | 规则修订后申报是否仍合规 | diff 新旧规则文本,标注变更条款 |
| 报价单/电量表核验 | 表格数值计算 + 边界校验 | 扩展现有 `extract_table_data` + 算术校验 |
| 费率阶梯计算器 | 阶梯电价/容量电价复合计算 | 扩展现有 `unit_converter`,参数化费率表 |
| 违规处罚测算 | 违规后处罚金额预估 | 规则条款参数化 + `arithmetic_compare` 组合 |

**话术模板**:「我会加一个 X 工具——业务上是 Y 需求,实现上复用现有 ToolExecutor 的配置驱动注册,再加 Z 参数校验,不动框架只加能力。」

---

## 回答节奏建议

1. **开场**:先讲已落地的三个(评估闭环、可观测性、RAGAS 评测)——「这三个是我盘点后优先做的,因为它们让简历上的指标可复现,还把 RAG 评测的 LLM-as-judge 口径落地了」;
2. **中段**:讲 1-2 个规划中的(建议 Judge 接线 + 真流式/多轮,二选一深入);
3. **收尾**:补一句方法论——「我的原则是:每个优化必须可测量(离线指标或线上观测),没有增量验证不上线」。

