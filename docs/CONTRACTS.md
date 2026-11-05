# AgentTwin v0.1 契约冻结稿

> 本文件是实现层唯一的接口事实源。任何一条改动都等于升 `schema_version`，禁止边写边改。
> 上游：《AgentTwin 技术设计方案 v0.1（设计冻结稿）》§0–§16。本文件与设计稿冲突时以本文件为准，冲突及理由见 §9。

## 1. 数据契约：JSONL tape（`schema_version = "0.1"`）

- 编码：UTF-8、无 BOM、LF 换行、文件末尾有换行。
- 第 1 行是 header，其后每行一个 event，不允许空行与注释。
- `seq` 从 0 开始、稠密、严格递增；加载时校验，违反抛 `SchemaError`。
- 未知的顶层键：header 与 event 都**拒绝**（抛 `SchemaError`），以此保证 schema 漂移一定被检出。

### 1.1 header

| 键 | 类型 | 必需 | 语义 |
| --- | --- | --- | --- |
| `schema_version` | str | 是 | 固定 `0.1`。header 的识别标志（第一个含此键的行）。 |
| `trace_id` | str | 是 | 人类可读 id，如 `checkout`。 |
| `seed` | int | 是 | 录制会话 seed；回放原样复制。 |
| `env_fingerprint` | obj | 是 | 见 §1.5。 |
| `redaction` | obj | 是 | `{"redacted": int, "policy": obj, "mode": "scrub" | "whitelist"}`。 |

### 1.2 event

| 键 | 类型 | 必需 | 语义 |
| --- | --- | --- | --- |
| `seq` | int | 是 | 唯一排序键、回放游标、diff 对齐、patch 定位的公共主键。 |
| `t` | float | 是 | 逻辑时钟，`t = round(seq * 0.1, 4)`；加载校验偏差 > 1e-6 即 `SchemaError`。 |
| `kind` | str | 是 | 见 §1.4，8 种且只有 8 种。 |
| `actor` | str | 是 | agent 名或工具名。 |
| `payload` | obj | 是 | 按 kind 定形状，见 §1.4。 |
| `patched` | bool | 否 | 只在为 true 时写出该键；反事实注入点标记。 |

### 1.3 规范化与 hash

唯一实现（`trace.canonical_json`）：`json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=repr)`。
`sha256(text.encode("utf-8")).hexdigest()`，输出 64 位小写 hex。所有 `*_hash` 均用此规则，hash **不含** `seq`/`t`/`patched`。

### 1.4 8 种 kind 与 payload 形状

| kind | 必需 payload 键 | 可选键 |
| --- | --- | --- |
| `llm_call` | `prompt_hash`, `response` | `prompt_preview`, `model`, `tokens`{`prompt`,`completion`} |
| `tool_call` | `tool`, `args_hash` | `args_preview` |
| `tool_result` | `tool`, `result` | `ok`(bool) |
| `state_patch` | `patch`（`{json_pointer: value}`，键以 `/` 开头） | — |
| `decision` | `note` | 任意附加键 |
| `human_input` | `input` | — |
| `error` | `type`, `message` | — |
| `terminal` | `state`(obj), `state_hash` | `outcome`(`ok` | `error`) |

工具参数规范化：`{"args": [...], "kwargs": {..}}` → `canonical_json` → `args_hash`；回放比对 args_hash，不符抛 `ArgsDrift`。
LLM 请求规范化：整个 request dict（调用方给的任意 dict）→ `canonical_json` → `prompt_hash`；回放只比 `prompt_hash`。
`state_hash` = `canonical_json(final_state)` 的 sha256；`Trace.state()` 由 `state_patch` 依 seq 顺序折叠，`terminal.payload.state` 为准。

### 1.5 env_fingerprint

`{"model": str|null, "prompt_hash": str|null, "python": "3.12.7", "agenttwin": "0.1.0", "platform": "win32", "deps": {name: version}}`。
回放比对：`model`、`prompt_hash`、`python`(major.minor)、`agenttwin`(major.minor) 不一致即 `FingerprintDrift`（硬失败，除非 `allow_fingerprint_drift=True`）；`deps`/`platform` 只进报告不判失败。

## 2. seq 语义与回放游标

- 游标 = tape.events 的下标，只能前进，不允许回退或跳读。
- 一次 `llm(req)` 消耗 1 个 `llm_call`；一次 `tools[t](*a, **kw)` 消耗 1 个 `tool_call` 与其**配对的下一个** `tool_result`（中间不允许插入其它事件）。
- 控制器请求的类型/actor 与 tape 当前位置不符 → `TapeMiss`（记录进报告，不抛、不中断），报告 `exit_code` 按 §6 归 2。
- **R-TERM**：回放产出的 Trace 最后一个事件必须原样复制 tape 的 `terminal`（全部字段含 `t`、`payload`、`patched`），否则逐位一致永远差最后一帧。
- **R-PATCH**：`patch(seq=..., response|result|args)=` 替换该事件的对应 payload 值，并在产出 Trace 的同 seq 事件上置 `patched=True`，`t` 不变。
- 回放产出的 Trace 是新 header（`trace_id`、`seed`、`env_fingerprint` 复制自 tape）+ 控制器重新产生的事件 + 复制的 terminal。

## 3. 公开 API（全部由 `agenttwin/__init__.py` 导出）

```python
class Twin:
    @classmethod
    def record(cls, trace_id: str, *, out: str | Path, seed: int = 0,
               llm: Callable[[dict], Any] | None = None,
               tools: Mapping[str, Callable[..., Any]] | None = None,
               model: str | None = None,
               redactor: Redactor | None = None) -> TapeRecorder: ...
    @classmethod
    def load(cls, path: str | Path, *, allow_fingerprint_drift: bool = False,
             allow_real: bool = False) -> ReplaySession: ...   # allow_real=True 直接 ValueError
    @classmethod
    def from_openai_compatible(cls, client, **kw) -> Twin: ...

class TapeRecorder:      # context manager
    llm: Callable[[dict], Any]        # 转发真实 llm 并录带
    def tools(self) -> Mapping[str, Callable[..., Any]];
    def __enter__/__exit__            # 退出时写 terminal 并 flush

class ReplaySession:
    def patch(self, *, seq: int, response: Any = _UNSET, result: Any = _UNSET,
              args: Any = _UNSET) -> ReplaySession;   # 链式，返回自身副本
    def replay(self, agent, *, policy: Mapping[str, Any] | None = None) -> Report;

class Report:
    trace: Trace; baseline: Trace; results: list[AssertionResult];
    misses: list[TapeMiss]; drifts: list[DriftError];
    def assert_(self, *specs) -> Report;   # 链式求值
    def warn(self, *specs) -> Report;
    def markdown(self) -> str;
    def json(self) -> dict;
    def html(self, path: str | Path) -> Path;
    exit_code: int;
```

Agent 调用约定：`agent` 是 `agent(llm, tools)` 形式的 callable，或带 `.run(llm, tools)` 的对象。若其签名还接受 `policy`（关键字或第 3 个位置参数），`Twin` 传入 `policy` dict；否则不传。

## 4. 四类 Drift / TapeMiss（`drift.py` 定义，`proxy.py` 使用）

```python
class DriftError(Exception):        # 基类，硬失败
    type: str; seq: int | None; expected: Any; actual: Any
    def to_dict(self) -> dict       # {"type","seq","expected","actual","message"}
class PromptDrift(DriftError); class ArgsDrift(DriftError); class FingerprintDrift(DriftError)
class TapeMiss(Exception):          # 非致命：结构化结果
    seq: int; expected_kind: str; actor: str
    def to_dict(self) -> dict       # {"type":"TapeMiss","seq","expected_kind","actor"}
```

PromptDrift/ArgsDrift/FingerprintDrift 抛出即终止 replay 并记入 `report.drifts`；TapeMiss 不抛出，记入 `report.misses` 后控制器继续（其后续观测一律拿不到 → 由断言的 SKIP/FAIL 体现）。

## 5. 断言契约（`asserts.py`）

```python
@dataclass(frozen=True)
class AssertionResult: name: str; passed: bool | None; detail: str; first_offense_seq: int | None = None
@dataclass(frozen=True)
class AssertionSpec:  name: str; fn: Callable[[Trace], AssertionResult]
    def __call__(self, trace: Trace) -> AssertionResult;
@dataclass(frozen=True)
class AssertionGroup: name: str; specs: tuple[AssertionSpec, ...]; mode: str  # all|any|warn
    def evaluate(self, trace: Trace) -> list[AssertionResult];
```

断言是 `Trace -> AssertionResult` 的纯函数：无 IO、无随机、无模型、不读文件。失败必须给 `first_offense_seq`。
7 个工厂（签名即契约）：`max_repeats(tool, n=3)`、`no_deadlock(limit=3)`、`budget_within(max_calls=None, max_seconds=None, max_tokens=None, price_table=None)`、`never_calls(tool, args_contains=None)`、`call_order(names)`、`call_after(a, b)`、`state_invariant(path, predicate)`、`output_schema(schema)`、`terminates_within(n)`。
（`call_order` 与 `call_after` 计入「顺序约束」族，`no_deadlock` 与 `max_repeats` 计入「重复」族 —— 对外仍称 7 种断言族。）
组合子：`assert_(*specs)`（全硬）、`warn(*specs)`（软，不进退出码）、`assert_any(*specs)`、`assert_all(*specs)`，返回 `AssertionGroup`。
`output_schema` 用内置的纯 Python JSON-Schema 子集（`type/required/properties/enum/minimum/maximum/items/pattern`），**不引入 pydantic**；不支持的 keyword 在构造时 `ValueError`。
YAML 化：`asserts.load_yaml(path) -> list[AssertionGroup]`，`- {spec: max_repeats, args: {tool: shell, n: 3}, warn: false}`；`yaml` 延迟导入，缺失时给安装提示。

## 6. 退出码

| code | 含义 |
| --- | --- |
| 0 | 全部 PASS（SKIP 不算失败） |
| 1 | 有断言 FAIL |
| 2 | 出现 Drift / TapeMiss / SchemaError / schema 不兼容 |
| 3 | 配置或安装错误（找不到 tape、断言文件语法错、缺依赖） |

`--strict` 把 SKIP 视为失败（→ 1）。优先级：3 > 2 > 1 > 0。

## 7. Diff 契约（`diff.py`，v0.1 只做 seq 对齐）

```python
TraceDiff(baseline: Trace, current: Trace)
    .first_divergence_seq: int | None
    .events: list[{"seq": int, "change": "same"|"added"|"removed"|"changed", "baseline": dict|None, "current": dict|None}]
    .state_diff: list[{"path": str, "before": Any, "after": Any}]   # JSON Pointer 级
    .assertion_matrix: dict[str, {"baseline": bool|None, "current": bool|None}]
```
`changed` 的判定：同 seq 上 `kind`/`actor`/`payload` 任一不同。`first_divergence_seq` = 第一个非 `same` 的 seq。

## 8. 报告与 HTML 回放器契约

- `Report.json()` 顶层键固定：`schema_version, trace_id, mode, exit_code, results, misses, drifts, diff, header`。
- `results` 项：`{"name","status":"PASS"|"FAIL"|"SKIP","detail","first_offense_seq"}`。
- **HTML 必须是单文件**：`src/agenttwin/html/replayer.html` 内含唯一的占位符 `__AGENTTWIN_PAYLOAD__`（位于 `<script type="application/json" id="agenttwin-payload">` 内）；`report.html(path)` = 用 `json.dumps(payload)` 替换该占位符后写出。
- payload 形状：`{"trace_id","mode","header","baseline":[event...],"current":[event...],"results":[...],"diff":{"first_divergence_seq","events":[...]},"labels":{"baseline":"recorded","current":"replay"}}`。
- replayer.html 内**不得出现**任何 `http://` / `https://` / `src=` 外链 / `@import`；必须能 `file://` 直接打开。
- HTML 最小功能集：时间轴 + 播放/暂停/步进/速度、kind/actor 过滤、`patched` 高亮、before/after 分栏同步、断言面板（点击失败项跳到 `first_offense_seq`）、state diff 面板。

## 9. 与设计稿的解读/偏离（含理由）

1. **`allow_real` 的落点**（设计稿 §5 自相矛盾：既说可显式开启，又说 Replay 模式下直接拒绝）。取严格解：`Twin.load(..., allow_real=True)` 抛 `ValueError`；真实工具只存在于 `Twin.record()`。理由：回放不得有副作用是硬承诺，留一个参数就是留一个事故。
2. **`Twin` 的模块位置**：设计稿 §10 没有 `twin.py`，故 `Twin`/`TapeRecorder`/`ReplaySession`/`Report` 的门面放 `env.py`，由 `__init__.py` 导出。
3. **`output_schema` 不依赖 pydantic**：改为内置纯 Python 子集校验器。理由：核心路径零第三方强依赖是 `doctor` 与离线回放器成立的前提，一处例外就会破。
4. **`t` 由 `seq` 唯一决定**（`t = round(seq*0.1, 4)`），不可由调用方指定。理由：逻辑时钟一旦可写就不再是逻辑时钟，且这样 `t` 的校验能在加载期拦下手改过的 tape。
5. **未知顶层键拒绝**（而非忽略）。理由：schema 漂移必须响，不能让未来的字段在旧版本里被静默丢弃。
6. **`TapeMiss` 不中断回放**。理由：§3.4 第 3 条已定；它是策略 diff 成立的前提，中断就退化成崩溃。
7. **`doctor` 容忍缺 terminal 的 tape**（判定为 `incomplete` 而非 `SchemaError`）。理由：子进程录制崩溃时留下的半截 tape 必须可诊断，而不是不可读。
