"""The adapter layer: record an agent you did not write (design doc 4.2 L1 + L2).

``wrap`` and ``install`` speak to any object shaped like the OpenAI SDK client,
and ``invoke`` calls a controller the way CONTRACTS section 3 defines.  No SDK is
imported anywhere here -- the adapter is duck-typed, so importing the package
stays free of optional dependencies.
"""

from __future__ import annotations

from .base import AgentCallable, AgentProtocol, AgentRunner, invoke, is_agent
from .openai_compatible import install, wrap

__all__ = [
    "AgentCallable",
    "AgentProtocol",
    "AgentRunner",
    "install",
    "invoke",
    "is_agent",
    "wrap",
]
