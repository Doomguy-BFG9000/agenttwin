[English](README.md) · [中文](README.zh-CN.md)

# AgentTwin

**Agent 的 VCR。** 把一次 Agent 运行录成一条 tape，之后离线回放，变成能提交到 CI 的单测。

不是 trace 仪表盘，不是监控平台：没有服务、没有数据库、没有 dashboard，只有一条 JSONL 文件和一次普通的断言。

30 秒版本：

1. 录制时调用模型，把请求哈希和响应写进 tape；
2. 回放时不调用模型、不执行真实工具，逐条比对 tape；
3. 比对不上就是失败（不是"重跑一次看看"）；断言结果最终就是一个进程退出码。

下面每个代码块都在本仓库上实际跑过：1 到 4 按顺序跑，5 只用仓库自带的 fixture。块下面的输出就是那次运行的输出。

## 1. 录一次

`record_demo.py`

```python
"""Record one agent run to a JSONL tape.  Run: python record_demo.py"""
import json

from agenttwin import Twin


def fake_llm(request):                    # stands in for your model client
    return {"text": f"order {request['prompt']} is on its way"}


def lookup_order(order_id):               # stands in for a real, side-effecting tool
    return {"order_id": order_id, "status": "paid"}


def agent(llm, tools):                    # your agent: identical when replaying
    reply = llm({"prompt": "A-1"})
    order = tools["lookup_order"]("A-1")
    return {"answer": reply["text"], "status": order["status"]}


with Twin.record("checkout", out="checkout.jsonl",
                 llm=fake_llm, tools={"lookup_order": lookup_order}) as rec:
    rec.returned(agent(rec.llm, rec.tools()))     # leaving the block writes the tape

lines = open("checkout.jsonl", encoding="utf-8").read().splitlines()
header = json.loads(lines[0])
print("header:", header["schema_version"], header["trace_id"], "seed=" + str(header["seed"]))
print("events:", " -> ".join(json.loads(line)["kind"] for line in lines[1:]))
print("line 2:", lines[1][:96] + " ...")
```

输出：

```text
header: 0.1 checkout seed=0
events: llm_call -> tool_call -> tool_result -> terminal
line 2: {"actor":"llm","kind":"llm_call","payload":{"prompt_hash":"1d783fcd9d8b565fc5b19532a4538139c2ab6 ...
```

真正"录"的只有最后三行：`with Twin.record(...) as rec:` 里跑一次 agent，`rec.returned(...)` 存下返回值，离开 `with` 时 tape 落盘。

`checkout.jsonl` 是纯 JSONL：第一行是 header（`schema_version` / `trace_id` / `seed` / `env_fingerprint` / `redaction`），之后每行一个事件；`kind` 只有 8 种：`llm_call`、`tool_call`、`tool_result`、`state_patch`、`decision`、`human_input`、`error`、`terminal`。任何文本工具都能读它。

## 2. 离线回放 + 断言

在同一个目录里接着跑：

`replay_demo.py`

```python
"""Replay the tape offline and assert on it.  Run after record_demo.py, same dir."""
import sys

from agenttwin import Twin, asserts


def agent(llm, tools):                    # the same agent, character for character
    reply = llm({"prompt": "A-1"})
    order = tools["lookup_order"]("A-1")
    return {"answer": reply["text"], "status": order["status"]}


report = Twin.load("checkout.jsonl").replay(agent)   # note: no llm=, no tools=
report.assert_(
    asserts.max_repeats("lookup_order", n=1),
    asserts.never_calls("refund"),
    asserts.terminates_within(10),
)

for result in report.results:
    status = "SKIP" if result.passed is None else ("PASS" if result.passed else "FAIL")
    print(f"{status}  {result.name}")
page = report.html("report.html")         # one self-contained file, opens from file://
print(f"html: {page.name} ({page.stat().st_size} bytes)")
print(f"divergent: {report.divergent}")
print(f"exit code: {report.exit_code}")
sys.exit(report.exit_code)
```

输出：

```text
PASS  max_repeats(lookup_order, n=1)
PASS  never_calls(refund)
PASS  terminates_within(n=10)
html: report.html (56546 bytes)
divergent: False
exit code: 0
```

两件容易忽略的事：

- `Twin.load("checkout.jsonl").replay(agent)` 没有 `llm=`，也没有 `tools=`：回放用的模型和工具都来自 tape。`llm(...)` 返回记录时的响应，`tools["lookup_order"](...)` 返回记录时的结果，真实函数一次都不会被调用。
- `sys.exit(report.exit_code)` 把断言结果变成普通退出码，`pytest` 或任意 CI 直接消费。

同一份结果的三种形态：`report.markdown()`（人读的表格）、`report.json()`（机器读，顶层 key 固定为 `schema_version` / `trace_id` / `mode` / `divergent` / `exit_code` / `results` / `misses` / `drifts` / `diff` / `header`）、`report.html("report.html")`（单文件报告，无外链，`file://` 直接打开；上面的字节数会随 tape 和运行环境略有不同）。 回放器顶部还带中英文切换（选择记在 `localStorage` 里），tape 内容本身不做翻译。

## 3. 请求变了就是硬失败

```python
"""A changed request is a hard failure, not a flaky test.  Run after record_demo.py."""
from agenttwin import DriftError, Twin


def changed_agent(llm, tools):
    llm({"prompt": "A-2"})                # one value changed in the request
    tools["lookup_order"]("A-1")


try:
    Twin.load("checkout.jsonl").replay(changed_agent)
except DriftError as exc:                 # PromptDrift / ArgsDrift / FingerprintDrift
    detail = exc.message.replace(exc.expected, exc.expected[:8] + "...").replace(exc.actual, exc.actual[:8] + "...")
    print(f"{type(exc).__name__}: {detail}")
    raise SystemExit(2)
```

输出：

```text
PromptDrift: prompt_hash mismatch at seq 0: tape expects 1d783fcd..., controller sent 4250ccd5...
```

注意是"抛出"而不是"记一笔再继续"：回放在这一步就停了，`SystemExit(2)` 让它成为 CI 里一个红色的 job。

## 4. 断言也可以是数据（可选 `[yaml]`）

`asserts.yaml`：

```yaml
# The same three checks as replay_demo.py, kept as data.
- {spec: max_repeats, args: {tool: lookup_order, n: 1}}
- {spec: never_calls, args: {tool: refund}}
- {spec: terminates_within, args: {n: 10}}
```

`yaml_demo.py`：

```python
"""The same three checks as data.  Run after record_demo.py."""
from agenttwin import Twin, asserts


def agent(llm, tools):
    reply = llm({"prompt": "A-1"})
    order = tools["lookup_order"]("A-1")
    return {"answer": reply["text"], "status": order["status"]}


report = Twin.load("checkout.jsonl").replay(agent)
report.assert_(*asserts.load_yaml("asserts.yaml"))   # needs the [yaml] extra

for result in report.results:
    print(f"{'PASS' if result.passed else 'FAIL'}  {result.name}")
print("exit code:", report.exit_code)
```

输出：

```text
PASS  max_repeats(lookup_order, n=1)
PASS  never_calls(refund)
PASS  terminates_within(n=10)
exit code: 0
```

## 本 tape 不承诺什么

这一节放在第二屏，因为最常被问的问题是"为什么我改了 temperature，回放结果没变"。

- **冻结的是请求和记录时的响应，不是可以重放一次的推理过程。** 回放时模型不会被第二次调用，所以采样参数（temperature / top_p / seed）本身不参与、也不影响结果。反过来说，如果你的 agent 把 temperature 放进请求 dict，它就是请求的一部分，会改变 `prompt_hash` 并触发 `PromptDrift`——请求变了本来就该失败。
- **没有真实副作用。** 回放中的工具调用返回 tape 里的 `result`，真实函数不执行；想"顺便真的调一下"也不行，`Twin.load(..., allow_real=True)` 直接 `ValueError`。
- **不重新调用模型。** 没有网络请求，没有 token 费用；tape 里有几条 `llm_call` 就回放几次，多一条也不会凭空出现。
- **四类 Drift 都是硬失败。** `DriftError` 与它的三个子类 `PromptDrift` / `ArgsDrift` / `FingerprintDrift` 都会抛出并终止回放；`TapeMiss`（tape 里没有对应记录）不是 Drift，它不抛出，但同样把 `exit_code` 变成 2。
- **回放不是"再跑一遍 agent"。** 命中 tape 时不重新执行模型或工具，没命中时也不会退化成真实调用。

## 不做什么

| 不做 | 为什么 |
| --- | --- |
| 不重新调用 LLM | 回放的价值是确定性和零成本；要新结果就重录一条 tape |
| 不产生真实副作用 | 回放要能在 CI 里、在断网的机器上跑 |
| 不承诺零配置支持所有框架 | agent 由你写成 `agent(llm, tools)` 或带 `run(llm, tools)` 的对象，tape 不认识你的框架 |
| 不做 SaaS / dashboard / 数据库 | tape 是文件，报告也是文件，没有后端可挂 |
| 不用 LLM-as-judge 判定对错 | 判定必须是确定性的断言；让模型打分等于把不确定性请回来 |

## 安装

```bash
python -m pip install -e .
```

只有在读写 YAML 断言文件时才需要额外依赖（`pyyaml` 仅在 `load_yaml()` 内部惰性导入）：

```bash
python -m pip install -e ".[yaml]"
```

Python 3.10+，运行时零第三方依赖：`pyproject.toml` 里 `dependencies = []`，`import agenttwin` 不会拉起 `pyyaml`。

### 退出码

| code | 含义 |
| --- | --- |
| 0 | 全部 PASS（SKIP 不算失败） |
| 1 | 有断言 FAIL |
| 2 | Drift / TapeMiss / SchemaError / schema 不兼容 |
| 3 | 配置或安装错误（tape 缺失、断言文件语法错误、缺依赖） |

`--strict` 把 SKIP 变成 1，优先级 3 > 2 > 1 > 0。`Report.exit_code` 只会产生 0 / 1 / 2；3 来自 CLI 的用法与配置层（`agenttwin.cli` 的 `EXIT_USAGE = 3`：tape 不存在、`--agent` 写法不对、断言文件读不了）。

### 命令行

装好之后 `agenttwin` 和 `python -m agenttwin` 是同一个入口（`agenttwin.cli:main`），五个子命令：`record` / `replay` / `diff` / `doctor` / `gallery`。先不录 tape，看仓库自带的十个 fixture（五组 before / after，每组一个断言族）：

```bash
python -m agenttwin gallery list
python -m agenttwin gallery run deathloop
```

输出（第二条是报告的开头和断言表；中间的 Diff / Fingerprint / Redaction 三节省略，`...` 就是省略处）：

```text
deathloop  Checkout retry loop
repaired  Checkout that stops on the failed observation
cleanup  Cleanup that deletes build/ with rm -rf
cleanup_guarded  Cleanup that leaves the tree alone
write_first  Release note written before the search
search_first  Release note that cites what it found
partial_receipt  Receipt without the charged total
full_receipt  Receipt that carries the total it charged
double_shipping  Order that pays shipping twice
correct_shipping  Order priced once

# AgentTwin report: gallery-deathloop

verdict: FAIL (exit code 1)
mode: replay
divergent: no
strict: no
...
| name | status | detail | first_offense_seq |
| --- | --- | --- | --- |
| max_repeats(shell, n=3) | FAIL | 6 calls of shell exceeds the limit of 3; first excess at seq 12 | 12 |
| budget_within(calls<=8) | FAIL | calls 13 > 8 (first excess at seq 12) | 12 |
| never_calls(shell, args_contains='rm -rf') | PASS | no call of shell matches | - |
```

`gallery run deathloop` 的退出码是 1，`gallery run repaired` 是 0：同一个断言集，区别只是那个 agent 会不会在第 13 次重复调用同一个工具。另外四组是同一个形状：`cleanup` / `cleanup_guarded`（`never_calls`）、`write_first` / `search_first`（`call_order`）、`partial_receipt` / `full_receipt`（`output_schema`）、`double_shipping` / `correct_shipping`（`state_invariant`）。`python -m agenttwin <command> --help` 看每个命令的参数。

## 状态

0.1.0 是仓库里的版本号，不是发布号：还没有对外发布。

- 已完成并有测试：录制/回放核心（`Twin.record` / `Twin.load`）、drift 检测、9 个断言工厂 + 组合子 + YAML 加载、`TraceDiff`、`Report`（JSON / Markdown / 单文件 HTML）、`agenttwin` 的五个子命令、openai-compatible adapter（鸭子类型，不 import SDK）、十个内置 fixture（五组 before / after）。
- 未验证：`.github/workflows/ci.yml` 只在本地解析过 YAML，没有在 GitHub 上真正跑过；本机沙箱装不了包，所以示例都用 `PYTHONPATH=src` 跑，`agenttwin` 控制台脚本由 `pyproject.toml` 声明但没真正执行过（`python -m agenttwin` 跑过）。
- 还没做：PyPI 发布、覆盖率门槛、任何托管服务或 dashboard。

