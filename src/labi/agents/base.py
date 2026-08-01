"""Base agent interface.

Deliberately synchronous: BaseProvider.generate()/generate_stream() (see
providers/adaptive_registry.py) make blocking litellm calls, not
async ones, so an async process() here would just be a wrapper around
sync work with no actual concurrency gained -- it would only add a
mismatch agents would have to work around.

_generate() is the one call every subclass uses to reach a provider. It
deliberately does NOT call provider.generate() directly -- it goes
through registry.get_all() + providers/generation.py's stream_generate(),
the exact same path agent.py's main loop and try_web_search_answer()
already use. That means every agent automatically gets: the adaptive
scoring (success rate / latency / cost-weighted), quota dampening,
per-capability stats recording, cost tracking, and the fallback-to-next-
candidate behavior -- for free, and from one place, instead of each
agent reimplementing "call a model" slightly differently.
"""

from abc import ABC, abstractmethod
from typing import Optional

from labi.context.snapshot import ContextSnapshot
from labi.context.update import TaskUpdate
from labi.context.prompt_builder import PromptBuilder
from labi.providers.adaptive_registry import AdaptiveProviderRegistry
from labi.providers.generation import stream_generate, estimate_tokens, ProviderCallError, _c


class BaseAgent(ABC):
    """All agents inherit from this."""

    name: str = "base_agent"
    capability: str = "answering"  # planning, coding, validation, answering, web_search...

    def __init__(self, registry: AdaptiveProviderRegistry, prompt_builder: PromptBuilder,
                 stats_store=None, cost_tracker=None):
        self.registry = registry
        self.prompt_builder = prompt_builder
        self.stats_store = stats_store
        self.cost_tracker = cost_tracker
        # Name of the provider that most recently succeeded via
        # _generate() -- callers outside the agent (e.g. main()'s
        # memory_db.save_task) can read this after process() returns to
        # know which provider actually produced the final result, since
        # TaskUpdate itself doesn't carry a provider field.
        self.last_provider: Optional[str] = None

    @abstractmethod
    def process(self, snapshot: ContextSnapshot) -> TaskUpdate:
        """Process the snapshot and return proposed changes."""
        raise NotImplementedError

    def can_handle(self, snapshot: ContextSnapshot) -> bool:
        """Override to restrict when this agent should run."""
        return True

    def _generate(self, prompt: str, system_prompt: Optional[str] = None,
                   max_tokens: int = 1024, label: str = "Thinking", render: str = "text") -> str:
        """Picks providers for self.capability via the same adaptive
        ranking every other call path uses, then tries each in order
        (mirroring try_web_search_answer's fallback loop in agent.py)
        until one succeeds. Raises RuntimeError if none do -- callers
        (process() implementations below) turn that into a TaskUpdate
        with status='failed' rather than letting it propagate raw.

        Filters out providers whose context window clearly can't fit this
        specific prompt (min_context) -- the same optimization the old
        CLI loop applied only to the planning call, generalized here so
        every agent gets it, since it's computed from the actual prompt
        each one builds."""
        prompt_tokens = estimate_tokens(prompt) + (estimate_tokens(system_prompt) if system_prompt else 0)
        candidates = self.registry.get_all(self.capability, stats_store=self.stats_store,
                                            min_context=prompt_tokens + 200)
        if not candidates:
            raise RuntimeError(f"No provider registered for capability '{self.capability}'")
        last_exc = None
        for provider in candidates:
            try:
                result = stream_generate(
                    provider, prompt, system_prompt=system_prompt, max_tokens=max_tokens,
                    label=label, render=render, stats_store=self.stats_store,
                    capability=self.capability, cost_tracker=self.cost_tracker,
                )
                self.last_provider = provider.name
                return result
            except ProviderCallError as e:
                print(_c(f"   [{provider.name}] failed ({e.reason}) -- trying next provider...", "yellow"))
                last_exc = e
                continue
            except Exception as e:
                last_exc = e
                continue
        raise RuntimeError(f"All providers for capability '{self.capability}' failed") from last_exc
