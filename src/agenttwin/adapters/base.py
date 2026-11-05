"""The frozen Agent Protocol (CONTRACTS section 3) -- one call shape for Twin.

A controller is either ``agent(llm, tools)`` or an object exposing
``agent.run(llm, tools)``; when its signature also accepts ``policy`` -- by name,
as the third positional parameter, or through ``*args`` -- Twin passes the policy
through.  ``env.py`` keeps a private copy of that decision (its surface is frozen);
this module is the standalone version, so an adapter can call a controller without
importing the record/replay facade.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AgentCallable",
    "AgentProtocol",
    "AgentRunner",
    "is_agent",
    "invoke",
]

_POSITIONAL_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


@runtime_checkable
class AgentCallable(Protocol):
    """CONTRACTS section 3 -- the callable form of a controller: agent(llm, tools)."""

    def __call__(self, llm: Any, tools: Mapping[str, Any]) -> Any:
        ...


@runtime_checkable
class AgentRunner(Protocol):
    """CONTRACTS section 3 -- the object form of a controller: agent.run(llm, tools)."""

    def run(self, llm: Any, tools: Mapping[str, Any]) -> Any:
        ...


# CONTRACTS section 3 allows either form, so the protocol is their union: a single
# Protocol cannot express it, because every member of a Protocol is required.
AgentProtocol = AgentCallable | AgentRunner


def is_agent(agent: Any) -> bool:
    """True when invoke() can call agent: the callable form or the .run form (CONTRACTS section 3)."""
    return callable(getattr(agent, "run", None)) or callable(agent)


def invoke(agent: Any, llm: Any, tools: Mapping[str, Any], policy: Any = None) -> Any:
    """Call agent the way CONTRACTS section 3 defines it and return its result.

    ``run`` wins over ``__call__`` when an object offers both (env.py decides the
    same way).  A signature that accepts ``policy`` always receives it, including
    ``None``: a controller whose third parameter is required stays callable when
    no policy was supplied.
    """
    runner = getattr(agent, "run", None)
    fn = runner if callable(runner) else agent
    if not callable(fn):
        raise TypeError(
            "agent must be callable(llm, tools) or expose run(llm, tools); "
            f"got {type(agent).__name__}"
        )
    style = _policy_style(fn)
    if style == "keyword":
        return fn(llm, tools, policy=policy)
    if style == "positional":
        return fn(llm, tools, policy)
    return fn(llm, tools)


def _policy_style(fn: Any) -> str | None:
    """How fn wants the policy: "keyword", "positional", or None when it takes none."""
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        # Not introspectable (a C builtin): the documented (llm, tools) form is all we can promise.
        return None
    named = parameters.get("policy")
    if named is not None:
        if named.kind is inspect.Parameter.POSITIONAL_ONLY:
            return "positional"
        return "keyword"
    kinds = [parameter.kind for parameter in parameters.values()]
    if inspect.Parameter.VAR_POSITIONAL in kinds:
        return "positional"
    positional = sum(1 for kind in kinds if kind in _POSITIONAL_KINDS)
    return "positional" if positional >= 3 else None
