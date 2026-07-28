"""Duck-types BaseProvider without touching litellm/the network. Shared
across tests/agents/ and tests/workflows/ so there's one definition, not
a copy per test file (tests/integration/test_web_pipeline.py has its own
near-identical copy for the same reason -- kept separate there since
that suite predates this package and moving it would widen that PR's
diff for no behavioral change)."""


class FakeProvider:
    def __init__(self, name, capabilities, response_text, priority=10, should_fail=False):
        self.name = name
        self.capabilities = capabilities
        self.priority = priority
        self.capability_priority = {}
        self.context_window = None
        self.model = "gpt-3.5-turbo"  # a model litellm has offline cost data for
        self._response_text = response_text
        self._should_fail = should_fail

    def priority_for(self, capability):
        return self.capability_priority.get(capability, self.priority)

    def generate_stream(self, prompt, system_prompt=None, max_tokens=768, history=None):
        if self._should_fail:
            raise RuntimeError(f"{self.name} is down")
        yield self._response_text

    def generate(self, prompt, system_prompt=None, max_tokens=768, history=None):
        if self._should_fail:
            raise RuntimeError(f"{self.name} is down")
        return {
            "content": self._response_text, "model": self.model,
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "provider": self.name,
        }
