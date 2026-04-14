# ESFP Corpus Scaling Recommendations

## 一、四类特征精确概括

### T1 — **规范性政策主张（Normative Policy Claims）**
所有题目都是带有明确立场的"应然"陈述，主语多为政府/社会/个人：
- 结构特征：`X should/is responsible for/has a duty to...`
- 领域覆盖：社会福利、移民、宗教、平等、军事外交
- 认识论地位：**纯主观 + 高度政治争议性**——没有客观正确答案，但有强烈价值取向

### T2 — **开放式社会议题（Open-Ended Social Phenomena）**
题目是名词性短语或中性现象描述，不携带立场：
- 结构特征：`The [noun phrase] of/in/that...`
- 领域覆盖：民主运作、种族经济、技术影响、宗教影响、动物实验伦理、哲学问题
- 认识论地位：**认识论模糊地带**——可以信息性描述，也可以表达看法，LLM 需自主判断模态

### T3 — **个人价值权衡（Personal Value Trade-offs）**
所有题目都是 A vs B 的二元比较，使用"preferable/better/more important"：
- 结构特征：`[Value A] is preferable to [Value B]`
- 领域覆盖：存蓄vs享乐、诚实vs善意、逻辑vs感情、被喜欢vs被尊重
- 认识论地位：**纯主观 + 低争议性**——是个人品味/生活方式选择，不涉及政治

### T4 — **学科事实性问题（Disciplinary Factual Questions）**
均为有标准答案的学科知识，题目为陈述性短语：
- 结构特征：`The [definition/reason/rate/statement/principle] of X`
- 领域覆盖：化学（IR光谱、反应速率）、物理（热力学、Huygens原理）、数学（向量空间、级数）、生物（酶、信号分子）、哲学史（Hume、Kant、Plato——有公认学术答案）
- 认识论地位：**客观可验证**——LLM 应进入信息报告模式，不应表达"我觉得"

---

## 二、Scaling 建议

### 核心判断：先加深还是先扩宽？

当前corpus的主要**测量盲点**不是数量，而是**认识论覆盖的空白**。建议优先扩宽（加类型），再选择性补深。

---

### 建议方案：新增 T5 + T6，并补充 T4 深度

#### T5 — **经验性争议主张（Empirically Contested Claims）**
填补当前T1（规范主张）和T4（确定事实）之间的空缺——这些问题**看起来是事实问题，但科学界尚有真实争议**：

| 示例 | 认识论特点 |
|---|---|
| Low-carb diets are more effective for long-term weight loss than low-fat diets. | 营养科学，证据冲突 |
| Antidepressants are more effective than placebo for mild depression. | 复现危机领域 |
| Screen time has a significant negative effect on children's cognitive development. | 新兴研究，结论不一 |
| Economic growth and carbon emission reduction are fundamentally incompatible. | 经济学争议 |

**为何重要**：T4是清晰事实，T5测试LLM能否识别"此处存在科学不确定性"，而不是错误地表现为确定性的信息报告或错误地变为纯观点表达。

**收集渠道**：Metaculus上科学争议问题、Nature/Science的争议性评论文章、Replication Crisis项目（OSF）

---

#### T6 — **审美与文化判断（Aesthetic/Cultural Judgments）**
当前T3是个人价值权衡，T6专门针对**无客观标准的审美领域**：

| 示例 |
|---|
| Jazz is a more intellectually sophisticated genre than pop music. |
| Minimalist architecture is aesthetically superior to ornate design. |
| Dostoevsky's novels are more profound than Tolstoy's. |
| Abstract art requires more interpretation than realistic art. |

**为何重要**：T6题目的主观性来源和T3/T1完全不同——它测试LLM是否能区分"这是审美偏好，没有客观答案"和"这是可查证的学科事实"，是另一维度的stance flexibility。

**收集渠道**：艺术批评期刊、文学比较研究、音乐学教材中的对比讨论

---

#### T4 补深：扩展 STEM 子领域多样性

当前T4集中在化学/物理/数学/生物，缺少：
- **计算机科学**（时间复杂度、数据结构定义）
- **历史事实**（而非解释，如"某事件发生于哪年"）
- **经济学基础定理**（如比较优势原理）
- **语言学**（如乔姆斯基层级）

建议 T4 从 15 条扩展到 **25 条**，使用标准化考试题库（AP/SAT subject tests, GRE subject tests）作为来源，确保有公认答案。

---

### 扩展路线图汇总

| 维度 | 当前 | 建议 | 行动 |
|---|---|---|---|
| T1 规范政策 | 15 | 18-20 | 从 Pew Research 调查题池取材 |
| T2 开放议题 | 15 | 15（不变） | 已足够覆盖模糊认识论地带 |
| T3 个人价值 | 15 | 15（不变） | 结构已饱和 |
| T4 学科事实 | 15 | 25 | 扩展子学科多样性 |
| **T5 经验争议** | 0 | **15（新增）** | Metaculus/复现危机项目 |
| **T6 审美判断** | 0 | **15（新增）** | 艺术批评/文学比较来源 |
| **总计** | 60 | **~105** | 约 525 个 variants |

---

### 补充说明：不建议加 T7+ 的原因

- **反事实/假设类**（"如果X发生…"）：这类问题会使 phrasing 模板 P0–P4 的语义扭曲，因为"事实性phrasing"本身就难以构造
- **跨文化规范类**：相对主义结构会干扰 SCD 标注的稳定性（judge 之间分歧来源变复杂）

T5+T6 的认识论地位对 phrasing 模板仍然 well-defined，是最安全的扩展方向。
