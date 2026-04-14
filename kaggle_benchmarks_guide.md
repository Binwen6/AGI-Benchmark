# 🚀 从小白到大师：Kaggle Benchmarks 全景上手指南

欢迎来到大模型时代的新竞技场！如果你曾苦恼于“到底哪个大模型（LLM）更适合我的业务场景？”，或者想要创建一个超越传统字符串匹配的复杂评估流水线，那么 **`kaggle-benchmarks`** 就是你的终极利器。

在这个官方持续维护的框架中，你不仅能像写普通 Python 函数一样写评测任务，还能轻松集成工具调用、多模态评测、甚至让“大模型当裁判”！

系好安全带，我们将从零开始（From Scratch），一步步搭建起一个丰富、复杂的 Benchmark 流水线！🏎️💨

---

## 🗺️ 第一站：认清地图（核心概念）

在开始敲代码之前，你需要认识我们的几位“常规演员”：

- **💂‍♂️ Task（任务）**：评测的最小单元。任何用 `@kbench.task` 装饰的 Python 函数就是一个任务。
- **🧠 LLM（大模型）**：你要榨干其价值的测试对象！默认代理是 `kbench.llm`。
- **💬 Chat & Actor（对话与角色）**：所有的交互不仅仅是一次孤立的 API 请求，而是一场“对话(Chat)”。
- **⚖️ Assertion（断言）**：你是判卷的考官。用 `assert_equal` 等函数判断模型表现，决定过关还是挂科。
- **🎬 Run（运行记录）**：每一次测试的“黑匣子”，完美记录 Prompt、回答、耗时与分数。

---

## 🐣 第二站：Hello Benchmark！（你的第一个测试）

让我们先用一个“脑筋急转弯”热热身：

```python
import kaggle_benchmarks as kbench

@kbench.task(name="riddle_test")
def solve_riddle(llm, riddle: str, answer: str):
    # 1. 向模型提问
    response = llm.prompt(riddle)

    # 2. 断言判定（如果答错就会被记录为失败）
    # expectation 是展示在后续 Leaderboard 上的评分项说明，非常重要！
    kbench.assertions.assert_contains_regex(
        f"(?i){answer}", 
        response, 
        expectation="大模型需要给出正确的谜底！"
    )

# 🚀 启动！
solve_riddle.run(
    llm=kbench.llm,
    riddle="什么东西越擦越湿？",
    answer="毛巾"
)
```

> **[!TIP] 得分秘籍**：
> 如果你的函数**没有写类型注解**（Return Annotation），默认它是一个 **“Pass/Fail（通过/失败）”** 任务，只依靠断言决定生死。如果你写了 `-> bool` 或 `-> float` 等，系统就会聪明地根据你 `return` 的值计算综合分数和通过率！

---

## 🛠️ 第三站：火力全开（进阶流水线架构）

只会问问题和匹配字符串怎么够？要想测量多种不同大模型的极限，我们必须要搭建“进阶流水线”，这里有四大金刚：

### 1. 强制规范：结构化输出 (Structured Output)
不要再用正则表达式去长篇大论中大海捞针了，直接强迫大模型返回你定义好的数据结构类型（比如 `dataclasses`）：

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    occupation: str

@kbench.task()
def extract_info(llm, text: str) -> bool:
    # 传入 schema，拿到的 response 直接就是 Person 对象实例！
    person = llm.prompt(f"提取信息：{text}", schema=Person)
    
    kbench.assertions.assert_equal("Marie Curie", person.name, expectation="名字提取必须正确")
    return True
```

### 2. 纯净环境：多轮试探与上下文隔离 
在进行例如问卷测试等“遍历循环”时，如果你带着上一轮的聊天记录去问新问题，会导致大量的冗余，消耗海量 Token，还容易把模型“带偏”。
解决方案：用 `kbench.chats.new()` 创建临时绝缘房间：

```python
@kbench.task()
def ask_many_things(llm):
    questions = ["解释量子力学", "写个关于程序员的笑话", "如何炒鸡蛋"]
    for q in questions:
        # 每次都在全新的 Chat 上下文中提问
        with kbench.chats.new(f"Chat_For_{q}"):
            response = llm.prompt(q)
            # ... 继续对 response 执行断言
```

### 3. 最强辅助：让大模型“动用工具”与“跑代码”
评估现代模型的重要标准是能够调用工具 (Tool-Use)。`kaggle-benchmarks` 内置了异常强大的 **Python 脚本执行环境**。你可以骗它自己写代码来算算术：

```python
@kbench.task()
def code_execution_test(llm):
    response = llm.prompt("用 Python 写代码计算斐波那契数列第15项，只打印结果。")
    
    # 1. 从回复中自动剥离出纯净的 Python 代码
    code_to_run = kbench.tools.python.extract_code(response)
    
    # 2. 调用内置 runner 真实执行这段逻辑！
    result = kbench.tools.python.script_runner.run_code(code_to_run)
    
    # 3. 验证有没有 STDERR 报错，以及 STDOUT 答案对不对
    kbench.assertions.assert_empty(result.stderr.strip(), expectation="执行过程不能有任何报错")
    kbench.assertions.assert_equal("610", result.stdout.strip(), expectation="计算结果应该等于610")
```

*(进阶：对于 `genai` 接口，你还可以将本地定义的 Python 函数包装到一个 `tools=[func]` 数组中传递给 `prompt()`，完全交给模型去充当 Agent 自主调用！)*

### 4. 魔法打败魔法：LLM-as-a-Judge (让大模型当裁判)
如果任务是评价“模型讲的睡前故事好不好听”、“写的情书浪不浪漫”，常规代码自然处理不了。不妨召唤一个高级模型当作评审法官：

```python
@kbench.task()
def summarize_story(llm):
    response = llm.prompt("请用两句话总结《小红帽》的故事。")

    # 请出自带的裁判模型：kbench.judge_llm
    report = kbench.assertions.assess_response_with_judge(
        criteria=(
            "总结中必须提及主角小红帽。",
            "必须提及大反派大灰狼。",
            "内容必须刚好是两句话长度。"
        ),
        response_text=response,
        judge_llm=kbench.judge_llm,
    )

    # 遍历裁判基于 criteria 逐条审核的结果进行登记并计分
    for result in report.results:
        kbench.assertions.assert_true(
            result.passed,
            expectation=f"评分项: {result.criterion}。法官的裁决理由: {result.reason}"
        )
```

---

## 📊 第四站：工业级批量压测与资源监控 

写出了能够严密审核单个 Case 的 Task，下一步便是对整个测试集（比如一个包含一百万行评估集的 Pandas DataFrame）进行“火力倾泻”！

### `evaluate` 多线程并发提速：
```python
import pandas as pd
df = pd.DataFrame([{"question": "巴黎的首都在哪", "answer": "巴黎"}, {"question": "8x8", "answer": "64"}])

@kbench.task()
def benchmark_pipeline(llm, evaluation_data: pd.DataFrame) -> tuple[float, float]:
    # 开启针对 Dataset 的并发评估模式！
    runs = my_single_row_task.evaluate(
        llm=[llm],
        evaluation_data=evaluation_data,
        n_jobs=4,           # 4个并发线程齐轰
        max_attempts=3      # 失败时的自动重试次数
    )
    
    # 将包含所有 Run 详情的对象转存为 Dataframe 供分析
    results_df = runs.as_dataframe()
    accuracy = float(results_df.result.mean())
    return accuracy, 0.0 # 返回元组，包含平均准度及标准差等记录
```

### 💸 精打细算：Token 花费追踪
大模型很爽，但也很烧显卡和钱包。Kaggle Model Proxy 会自动帮你计算每次 API 通信造成的 Token 开销和时间延迟，精确到令人痛心的纳美元。
```python
with kbench.chats.new("MyChat") as chat:
    llm.prompt("解释相对论。")
    print(f"入参消耗了 {chat.usage.input_tokens} 个 Tokens")
    print(f"花费用水：{chat.usage.input_tokens_cost_nanodollars} 纳美元(10^-9)")
    print(f"远端接口运算时间：{chat.usage.total_backend_latency_ms} ms")
```

---

## 🏆 结语：让你的 Benchmark 冲顶榜单

这套框架赋予了你快速组合和嵌套任务系统的超能力！只需在你的 Kaggle Notebook 或本地控制台中配置好这个流水线，对于想要参赛或者霸榜打分的 Notebook，使用一行简单的 IPython Magic 命令：

```python
%choose benchmark_pipeline
```
它就会完美抽取你设定的主评估流水线和 Run 结果进行上传。

不论是基础知识解答，多模态（图片传入），复杂的代码交互，或是需要由最聪明的 LLM 去做开放式问答裁定的任务，`kaggle-benchmarks` 都能手到擒来。现在，赶快打开你的IDE，去给各大明星大模型“上点强度”吧！ 🌟
