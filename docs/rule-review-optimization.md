# 规则审查子系统优化方向清单(面试素材 + 演进规划)

> 用途:面试追问「这个项目还可以如何优化?添加哪些能力、功能、工具?」的回答素材。
> 状态:方向一、方向六已落地实施;其余方向为话术与规划。
> 关联:设计文档 `rule-review-design.md`、工作流文档 `rule-review-workflow.md`、测试 `tests/test_rule_review_evaluation.py`、`tests/test_rule_review_observability.py`。

---

## 回答框架(开场 30 秒)

「我盘点过这个项目的优化空间,按**面试官视角**排了个优先级:先补可复现的指标(评估闭环)、再补可观测的底座(延迟分位数/审计字段)、然后是体验层(真流式/多轮)、最后是规模化(Milvus/PG 接线)和前沿探索(GraphRAG)。其中前两个我已经落地了,后面是规划中的二期。」

- 已落地:✅ 方向一(评估体系闭环)、方向六(可观测性)
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

**定位**:88% 召回率的天花板突破点,对接「查询重写」「GraphRAG」两个高频考点。

**现状缺口**:设计文档 §12.3 规划的「LLM 多 query 生成」未实现;检索只有 RRF 混合两路;无时间维度裁剪。

**怎么做**:改写阶段生成 2-3 个 query 变体并行检索后合并去重;时间解析前置过滤(chunk 元数据带时间范围);GraphRAG 抽取条款引用关系做一跳扩展;每个模块用方向一的 recall@k 离线验证增量。

**面试话术**:

> 混合检索到 88% 之后,我往查询端和结构端做:LLM 多 query 生成并行检索合并、时间维度先裁剪再检索、GraphRAG 把条款引用关系建成图做一跳扩展。每个模块上线前必须用离线 recall@k 验证增量,没有指标提升就不上线——这是我做检索迭代的原则。

**追问预案**:「GraphRAG 和向量检索的区别?」→ 向量检索对实体关联、多跳问题天然弱(「引用 A 的条款 X 是否也适用于场景 B」),GraphRAG 显式建模引用关系,一跳扩展可命中;代价是建图与维护成本,适合规则文档这种结构稳定、引用密集的语料。

---

## 方向八:工具系统增强 + 代码质量债

**定位**:面试前的「地基工程」,让所有指标可复现、可解释。

**现状缺口**:`tool_executor.py` TOOL_MAP 硬编码 5 个工具,LLM 的 tool_calls 无 JSON Schema 校验直接 `func(**args)`;聚合决策逻辑三份重复且语义有细微差异;查询链路 0 测试;残留死代码(`_can_share_api_call` 恒 False 等)。

**怎么做**:tool 参数 JSON Schema 校验(非法参数进函数前拦截);注册表配置驱动(加工具不改代码);三份决策逻辑合并为单一纯函数 + golden 测试锁行为;清死代码;查询链路补冒烟测试。

**面试话术**:

> 我把工具调用做了 JSON Schema 校验,非法参数在进入函数前就被拦截;注册表改成配置驱动,加工具不改代码。同时清掉了几百行死代码和一处把内部意图打 stdout 的调试 print——质量债不清理,前面所有指标都不可复现,这是我给项目上的第一课。

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

1. **开场**:先讲已落地的两个(评估闭环、可观测性)——「这两个是我盘点后优先做的,因为它们让简历上的指标可复现」;
2. **中段**:讲 1-2 个规划中的(建议 Judge 接线 + 真流式/多轮,二选一深入);
3. **收尾**:补一句方法论——「我的原则是:每个优化必须可测量(离线指标或线上观测),没有增量验证不上线」。
