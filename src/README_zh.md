# ESFP Benchmark

ESFP Benchmark 旨在衡量大语言模型 (LLMs) 的**认知灵活性 (Epistemic Stance Flexibility)**——即模型是否能够根据 Prompts 的上下文语境，灵活地在“客观事实播报(Information-reporting)”与“主观立场表达(Opinion-expressing)”之间切换。

本项目已通过 `litellm` 进行模块化改造，不再绑定 Kaggle 的执行生态，这允许你通过标准的 API Key （如 OpenAI, Anthropic, Google, 智谱或私有化模型）无缝接入并测试。

## 项目结构
```text
src/
├── esfp_benchmark/
│   ├── __init__.py
│   ├── config.py         # 核心系统 Prompts 及并发配置
│   ├── client.py         # 基于 litellm 实现的客户端封装，包含失败重连、System降级兜底方案
│   ├── generator.py      # 读取测试题库，自动衍生生成 5 种预设 Phrasings (P0-P4)
│   ├── metrics.py        # 评分计算逻辑核心：计算 AR, PSI (加载 SentenceTransformer), SCD, CPC Kappa
│   ├── evaluator.py      # Benchmark 流水线：异步执行推理、缓存落盘(支持断点续传)以及3裁判机制打分
│   ├── visualize.py      # 基于 matplotlib 生成并导出核心分析图表 (Fig 1 ~ Fig 4 等)
│   └── main.py           # 命令行(CLI)执行主入口
├── .env.example          # API 环境变量参考配置
├── requirements.txt      # 依赖清单
└── README.md             # 本文档
```

## 环境准备

1. **进入工作目录**  
切换到本目录：
```bash
cd "/Users/binwen6/project/DeepMind/AGI Benchmark/src"
```

2. **安装依赖 (Python 3.10+)**  
安装执行与可视化作图所必备的环境包：
```bash
pip install -r requirements.txt
```

3. **配置 API 凭证**   
生成本地的 `.env` 并在其中填入你要测试模型的相关密钥。  
得益于 `litellm` 的强兼容性，不管是评测模型还是裁判模型，只要环境提供对应的 Key 即可自动跑通。
```bash
cp .env.example .env
```

## 执行方式

可以直接调用内置的入口脚本 `esfp_benchmark.main` 进行端到端基准测试：

```bash
python -m esfp_benchmark.main \
  --models "openai/gpt-4o" "anthropic/claude-3-haiku-20240307" \
  --judges "gemini/gemini-1.5-flash" "openai/gpt-3.5-turbo"
```

### 命令参数说明
* `--models`: (必选) 空格分隔的候选模型。使用的命名规则同 `litellm`，比如 `openai/gpt-4o`。如果是自定义或代理 URL 请先按 `litellm` 文档配置环境变量。
* `--judges`: (可选) 评估主观句式分布 `SCD` 与进行 `CPC` 抽取所需的裁判模型。通常我们会组建多模型评审委员会。默认值是一个双裁判组合 (`gemini/gemini-2.0-flash-lite`, `qwen/qwen-turbo`) 以平衡精度与 API 成本。
* `--corpus`:(可选) 测试题库源。默认会自动读取工作区同级别的 `../asset/ESFP_corpus_v1.csv` 文件。

### 断点防崩溃设计
**不用担心 API 中途断开！** 本工程对核心中间产物应用了高频的 Parquet Checkpoint 保存策略：
如果你评测了 3 个模型，跑到第 2 个模型的第 400 个题目时网络发生中断，下次启动 **原样运行同样的命令**，系统将自动读取硬盘里的 `.parquet` 进度结果跳过已执行任务，不会造成已跑数据的资源浪费！

## 查看结果
随着测试进度条到达终点，程序最终会在 `RESULTS_ESFP` 下输出报告与图表。

* `RESULTS_ESFP/final_scores.csv` — 生成各候选模型的聚合统计数据面板 (ESFP, dAR, dSCD...)。
* `RESULTS_ESFP/ESFP_bootstrap_ci.csv` — 包含了经过 1000 次重抽样的 95% 置信区间 (95% CI) 信息。
* `RESULTS_ESFP/figures/` — 所有的关键对比分析图表会以 `.pdf` 及 `.png` 双格式打包留存。
