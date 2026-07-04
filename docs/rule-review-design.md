# 电力规则审查系统 — 详细设计文档 v2

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

响应：

```json
{
  "status": "success",
  "data": {"doc_id": "doc_abc", "file_name": "规则.pdf", "page_count": 45, "chunk_count": 120}
}
```

### 5.3 `GET /v1/rule-review/documents` — 文档列表

### 5.4 `DELETE /v1/rule-review/documents/{doc_id}` — 删除文档

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
```

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
