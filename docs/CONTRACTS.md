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

- `Report.json()` 顶层键固定：`schema_version, trace_id, mode, divergent, exit_code, results, misses, drifts, diff, header`。
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
8. **R-TERM 的精确边界（收紧 §2 的无条件表述）**：回放产出的 Trace 必须自身合法（save → load 通过），因此
   - 若控制器消费的事件数**等于** tape 中 terminal 的 `seq`（未分叉）：terminal **原样复制**（`seq` / `t` / `payload` / `patched` 全同），逐位一致成立；
   - 否则（分叉、提前结束、或出现 TapeMiss）：terminal 用 `Trace.append` 落到新 `seq`，**只复制 `payload`**，`Report.json()['divergent'] = true`。
   理由：`t = round(seq*0.1, 4)` 与「terminal 的 t 原样复制」只在两者事件数相同时同时成立；无条件复制会写出一条自己都 load 不进去的 tape，把「逐位一致」偷换成「产出一个非法文件」。
9. **未知 `payload` 键一律接受**，只校验 §1.4 的必需键。理由：`decision` 明确允许任意附加键，且可选键要求向前兼容。
10. **`patched: false` 读入后归一为空**：写盘只在 true 时出现该键。
11. **header 必须在第 1 行**；「第一个含 `schema_version` 的行」是识别标志，不是扫描指令。
12. **`redaction` 与 `env_fingerprint` 的内层结构不由 `trace.py` 校验**（对象性即可），语义归 `redact.py` / `fingerprint.py`。理由：否则给 policy 加一个字段就是 tape 破坏性变更。
13. **脱敏是整值替换、首个匹配即止**：`'ping bob@x.com now'` → `'<redacted:email>'`；一个字符串最多计 1 次。理由：局部掩码会留下上下文残片，而「绝不泄漏」是硬承诺。
14. **显式传入 `drop_keys` / `patterns` 是替换默认集**（`None` 才用默认集）。
15. **只有 `whitelist` 模式才丢非白名单键**；drop-key 命中优先于白名单，未知 kind 在 whitelist 模式下全丢。
16. **`capture()` 的 `python` 存 major.minor**（`3.12`），`compare` 双侧归一后再比。
17. **本机测试环境事实**：workspace-write 沙箱下 pytest 的 `tmp_path` 不可用（`PermissionError: [WinError 5]`，路径 `%TEMP%\pytest-of-*`）。测试一律使用 `tests/conftest.py` 提供的 `tape_dir` fixture（根 `build/pytest-tmp`）。**禁止使用 `tmp_path`**。
18. **`divergent` 字段**（见 §8）：仅 R-TERM 第 2 分支为 true；未分叉的回放必须为 false，否则「逐位一致」无法自证。
19. **TapeMiss 之后 proxy 返回 `None`**：记 miss、游标**不回退也不跳过**，后续每次调用继续记 miss；不抛异常。理由：§3.4 第 3 条要求它不中断回放，而「跳过去找下一个同类事件」会让 diff 在两个不同时间点上对齐。
20. **hash 基于脱敏前的请求/参数，preview 存脱敏后的值**（各截断 4096 字符）：`prompt_hash = sha256(canonical_json(request))`、`args_hash = sha256(canonical_json({"args": [...], "kwargs": {...}}))`，二者都在 Redactor 之前算；`prompt_preview` / `args_preview` 存脱敏后的可读副本。理由：脱敏后的值不稳定（策略一改 hash 全变），而 preview 绝不能带 secret。
21. **`Twin.load` 的 fingerprint 策略**：`agenttwin`(major.minor) 不符一律抛 `FingerprintDrift`；`model` 只有在调用方**显式给出** `Twin.load(path, model=...)` 且与 tape 不符时才抛；`python` / `platform` / `deps` / `prompt_hash`，以及「tape 记了 model 而 load 没给 model」的情况，只记入 `fingerprint_diff`，不抛。
    - 理由一：§13 要求 3.10–3.13 CI 矩阵复用同一批 fixture，python 版本漂移若致命，跨版本回放这条承诺就不成立。

25. **`warn(...)` 组的软性随 Report 链传递**：`Report._extend` 见到 `AssertionGroup.mode == "warn"` 就把该组的每一行标成 soft，无论它是从 `report.warn(...)` 还是 `report.assert_(asserts.warn(...))` 进来的。理由：§5 写的是「`warn` 软、不进退出码」，而 detail 里带 `[warn]` 的行把退出码顶成 1，等于同一份报告自相矛盾。
    - 边界：嵌套在 `assert_any` / `assert_all` 里面的 warn 组仍按 hard 处理（宁可误报，也不静默吞掉一个真实失败）；软性只认最外层 spec 的 mode。
26. **`strict` 属于 Report，不属于 `Twin.load` / `ReplaySession`**：`Twin.load(path, strict=True)` 是 TypeError（frozen surface 无此参数）；开启方式是 `Report(strict=True)` 或 CLI `--strict`。**W3 的 CLI 必须让 `--strict` 真的把 SKIP 顶成退出码 1，并附测试**——目前仓库里没有任何路径会置 `Report.strict=True`，这条是已验证的缺口，不是未知项。
27. **SKIP 的裁定**：`never_calls(tool)` 在一个从未出现的工具上是 **PASS**（不变式已满足），不是 SKIP；SKIP 只留给「这份 tape 无法求值」的情形——`call_order` 里少于两个名字出现、`budget_within()` 一个上限都没给、`state_invariant` / `output_schema` 遇到既无 terminal 也无 state_patch 的 tape。理由：把「没违规」报成 SKIP 会让 `--strict` 把安全的 tape 判红。
