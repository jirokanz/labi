"""
Real provider intelligence -- adaptive per-capability ranking, live model
discovery (Cerebras, Groq, Gemini, OpenRouter), and quota reference data.

Moved out of agent.py, where this used to live directly inside the CLI
entrypoint file alongside memory/workspace/replay/session logic (the
"god object" problem flagged in review). agent.py now imports from here
instead of defining its own copies.

Named AdaptiveProviderRegistry (not ProviderRegistry) deliberately --
there's already a simpler ProviderRegistry in providers/registry.py that
intelligence/router.py depends on, with a different interface (key-based
get_provider_status/select_best_provider vs. capability-based
get_best/get_all here). Colliding the names would have been confusing
even though only one is imported in a given file; keeping them distinct
makes it unambiguous which "provider registry" a given piece of code
means. agent.py imports this class aliased as ProviderRegistry for
backward compatibility with existing call sites and tests.
"""

import litellm


class BaseProvider:
    def __init__(self, name, model, api_base, api_key, capabilities=None, priority=100,
                 capability_priority=None, context_window=None):
        self.name = name
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.capabilities = capabilities or ["text_generation"]
        self.priority = priority
        # Optional per-capability override, e.g. {"coding": 5} to rank this
        # provider higher for coding specifically without changing its
        # (possibly lower) rank for other capabilities it also serves.
        self.capability_priority = capability_priority or {}
        # Known context window in tokens, or None if genuinely unverified.
        # Left None rather than guessed for providers where the actually
        # served model varies (OpenRouter/Cerebras free catalogs both
        # change under us -- see pick_*_model) since a wrong guess here
        # would silently exclude a provider that could actually handle
        # the request, or worse, include one that can't.
        self.context_window = context_window

    def priority_for(self, capability):
        return self.capability_priority.get(capability, self.priority)

    def _messages(self, prompt, system_prompt, history):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt, system_prompt=None, max_tokens=768, history=None):
        messages = self._messages(prompt, system_prompt, history)
        try:
            response = litellm.completion(
                model=self.model, messages=messages, api_base=self.api_base,
                api_key=self.api_key, max_tokens=max_tokens, temperature=0.3,
            )
        except Exception as e:
            raise Exception(f"Provider {self.name} failed: {e}")
        usage = response.get("usage", {})
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "provider": self.name,
        }

    def generate_stream(self, prompt, system_prompt=None, max_tokens=768, history=None):
        """Yields text chunks as they arrive, for a live 'vibe coding' feel.
        Falls back to a single chunk if the provider/model can't stream."""
        messages = self._messages(prompt, system_prompt, history)
        try:
            stream = litellm.completion(
                model=self.model, messages=messages, api_base=self.api_base,
                api_key=self.api_key, max_tokens=max_tokens, temperature=0.3,
                stream=True,
            )
            full = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    full.append(delta)
                    yield delta
            self._last_full = "".join(full)
        except Exception as e:
            raise Exception(f"Provider {self.name} failed: {e}")


class AdaptiveProviderRegistry:
    MIN_SAMPLES = 4          # below this, trust the static priority (cold start)
    LATENCY_PENALTY_PER_SEC = 5  # score points lost per second of avg latency
    # Score points lost per $ of avg cost per call, keyed by capability --
    # a flat weight across all capabilities would treat a $0.01 penalty on
    # "coding" (where a wrong answer costs an autofix retry loop, so quality
    # should dominate) the same as on "answering" (where a merely-good-enough
    # free answer is fine). Lower weight = cost matters less = quality-
    # sensitive capability; higher weight = cost matters more.
    COST_WEIGHTS = {
        "coding": 300,
        "planning": 500,
        "answering": 1000,
        "validation": 800,
    }
    DEFAULT_COST_WEIGHT = 500  # for capabilities not listed above (e.g. web_search)
    # All providers currently in build_registry() are free-tier, so this
    # stays a no-op today (avg_cost_usd == 0) -- it only bites once a paid
    # provider (e.g. the commented-out DeepSeek) is registered alongside
    # free ones, so cost is weighed instead of ignored.
    QUOTA_DAMPEN_FLOOR = 0.05  # never fully zero out a provider on quota alone --
                                # our tracked usage can undercount real usage
                                # (same key used outside this app), and a daily
                                # cap resets on its own, so "at the cap" isn't a
                                # hard guarantee of failure -- keep it as a
                                # last resort rather than excluding it outright.

    def __init__(self):
        self.providers = []

    def register(self, provider):
        self.providers.append(provider)

    def _quota_factor(self, provider, stats_store):
        """Multiplicative dampening (QUOTA_DAMPEN_FLOOR..1.0) based on how
        close this provider is to its known daily request cap. Providers
        with no published daily limit (OpenRouter, Cerebras, Mistral --
        see KNOWN_QUOTAS) are never dampened; there's no reliable number
        to judge them against, and guessing one would be exactly the kind
        of unverified-limit mistake we've already been burned by."""
        if stats_store is None:
            return 1.0
        usage = stats_store.get_daily_usage(provider.name)
        status = compute_quota_status(provider.name, usage)
        if status is None:
            return 1.0
        pct = min(status["pct_used"], 100.0) / 100.0
        return max(self.QUOTA_DAMPEN_FLOOR, 1.0 - pct)

    def _score(self, provider, capability, stats_store):
        """Higher is better. Blends measured success rate + latency + cost with
        the static priority as a prior, so a provider with few/no data points
        still ranks the same as the old hardcoded-priority behavior. Quota
        headroom then dampens whichever base score, applied uniformly so a
        provider nearing its daily cap gets deprioritized regardless of how
        good its priority/success/latency/cost numbers look in isolation."""
        static_score = 1000 - provider.priority_for(capability)  # invert: lower priority number = higher score
        if stats_store is None:
            base = static_score
        else:
            stats = stats_store.get_provider_stats(provider.name, capability)
            if not stats or stats["calls"] < self.MIN_SAMPLES:
                base = static_score
            else:
                success_rate = stats["successes"] / stats["calls"]
                avg_latency_s = stats["total_latency_ms"] / stats["calls"] / 1000
                avg_cost_usd = stats["total_cost_usd"] / stats["calls"]
                cost_weight = self.COST_WEIGHTS.get(capability, self.DEFAULT_COST_WEIGHT)
                # success rate dominates (0-100 range); latency and cost are
                # both tie-breaking penalties on top of it.
                base = (success_rate * 100) \
                    - (avg_latency_s * self.LATENCY_PENALTY_PER_SEC) \
                    - (avg_cost_usd * cost_weight)
        return base * self._quota_factor(provider, stats_store)

    def _fits(self, provider, min_context):
        if min_context is None or provider.context_window is None:
            return True  # unknown window -- don't exclude, we have no basis to
        return provider.context_window >= min_context

    def get_best(self, capability, stats_store=None, min_context=None):
        candidates = [p for p in self.providers
                      if capability in p.capabilities and self._fits(p, min_context)]
        if not candidates:
            return None
        candidates.sort(key=lambda p: self._score(p, capability, stats_store), reverse=True)
        return candidates[0]

    def get_all(self, capability, stats_store=None, min_context=None):
        candidates = [p for p in self.providers
                      if capability in p.capabilities and self._fits(p, min_context)]
        candidates.sort(key=lambda p: self._score(p, capability, stats_store), reverse=True)
        return candidates


def _fetch_json(url, headers=None, timeout=5):
    import urllib.request
    import json as _json
    # Without a browser/curl-like User-Agent, urllib sends the default
    # "Python-urllib/3.x", which some providers' bot-protection/WAF
    # blocks or rate-limits more aggressively than others -- this was
    # missing here (though present in the original check_cerebras_v3.py
    # script) and is the likely cause of Groq/Cerebras discovery failing
    # in the field while OpenRouter/Gemini/Mistral succeeded.
    merged_headers = {"User-Agent": "curl/8.0", **(headers or {})}
    req = urllib.request.Request(url, headers=merged_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _json.loads(resp.read())


# Populated with the last real exception per provider whenever discovery
# fails, so callers (see agent.py's register_discovered) can print an
# actual reason instead of just "could not discover a working model" --
# that message alone doesn't distinguish an auth failure, a network
# error, and an empty catalog, which made this hard to diagnose in the
# field (see conversation: Groq/Cerebras failing silently on a Pi while
# OpenRouter/Gemini/Mistral worked).
LAST_DISCOVERY_ERROR = {}


def _rank_pick(model_ids, preferred, fallback_to_first=True):
    """Shared picker for every provider below: given a live list of model
    ids and an ordered list of preferred name-fragments (best first),
    return the first model id containing the highest-ranked fragment.
    Falls back to model_ids[0] -- whatever the API happens to list first
    -- only if nothing on the preference list matches, since that's still
    better than hardcoding a name that can silently vanish from a
    provider's catalog (this is what bit us with Cerebras dropping Llama,
    and what's about to bite Groq when llama-3.3-70b-versatile is
    retired 08/16/2026)."""
    lowered = {mid.lower(): mid for mid in model_ids}
    for pref in preferred:
        for low, orig in lowered.items():
            if pref in low:
                return orig
    if fallback_to_first and model_ids:
        return model_ids[0]
    return None


# Explicit denylist, checked before ranking: models confirmed
# deprecated/paid-only that should never be picked even as a last-resort
# fallback (i.e. even if every entry in the matching PREFERRED_* list is
# absent from the catalog and _rank_pick would otherwise fall back to
# "whatever's listed first"). The preference lists above already push
# these to the back of the ranking, but that's not a guarantee they're
# never selected -- a denylist is.
AVOID_MODELS = {
    "cerebras": set(),  # nothing confirmed deprecated as of this writing
    "groq": {
        "llama-3.3-70b-versatile",  # deprecated by Groq, shutdown 08/16/2026
        "llama-3.1-8b-instant",     # deprecated by Groq, same announcement
    },
    "gemini": {
        "gemini-2.5-flash",  # deprecated, shutdown 10/16/2026, some accounts already failing early
    },
    "openrouter": set(),  # the $0-pricing filter already excludes paid variants
}


def _filter_avoid(model_ids, provider_key):
    avoid = AVOID_MODELS.get(provider_key, set())
    return [m for m in model_ids if m not in avoid]
PREFERRED_CEREBRAS_MODELS = [
    "llama-4", "llama4", "qwen3-235b", "gpt-oss-120b", "qwen3-32b",
    "zai-glm", "llama-3.3", "llama",
]
PREFERRED_GROQ_MODELS = [
    "gpt-oss-120b", "qwen3.6-27b", "gpt-oss-20b", "llama-3.3", "llama",
]
PREFERRED_GEMINI_MODELS = [
    "gemini-3.5-flash", "gemini-3.1-flash", "gemini-3-flash", "flash",
]
PREFERRED_OPENROUTER_MODELS = [
    # Confirmed live 07/26/2026: OpenRouter's free catalog has moved
    # away from Llama entirely (same shift Cerebras made earlier) --
    # llama-3.3-70b-instruct:free no longer exists there, which meant
    # this list fell through to picking whatever was listed first
    # (an obscure small model) rather than a deliberate choice. Ranked
    # by recognizable lab + larger parameter count, since we don't have
    # hard quality benchmarks across these to rank on: OpenAI's
    # gpt-oss family, then Nvidia's larger Nemotron variants, then
    # Google's Gemma, with "llama" kept last in case Llama free models
    # return later.
    "gpt-oss-120b", "gpt-oss-20b",
    "nemotron-3-ultra", "nemotron-3-super",
    "gemma-4-31b", "gemma-4",
    "llama-3.3-70b-instruct", "llama-3.1-70b-instruct", "llama",
]


def pick_cerebras_model(api_key, fetch_fn=None):
    """Cerebras's free-tier model catalog is confirmed to change without
    notice (one documented case: ~12 models down to 2 within months, same
    NotFoundError we hit live) -- hardcoding a model name is fragile by
    design. Query the live /v1/models endpoint and pick a workable one
    instead of guessing. Returns None (caller should skip registering
    Cerebras this session) if discovery fails or the catalog is empty."""
    fetch_fn = fetch_fn or (lambda k: _fetch_json(
        "https://api.cerebras.ai/v1/models", headers={"Authorization": f"Bearer {k}"}))
    try:
        data = fetch_fn(api_key)
        model_ids = _filter_avoid([m["id"] for m in data.get("data", [])], "cerebras")
    except Exception as e:
        LAST_DISCOVERY_ERROR["cerebras"] = f"{type(e).__name__}: {e}"
        return None
    if not model_ids:
        LAST_DISCOVERY_ERROR["cerebras"] = "catalog returned zero models"
        return None
    LAST_DISCOVERY_ERROR.pop("cerebras", None)
    return _rank_pick(model_ids, PREFERRED_CEREBRAS_MODELS)


def pick_groq_model(api_key, fetch_fn=None):
    """llama-3.3-70b-versatile (the model this used to be hardcoded to)
    is deprecated by Groq with a shutdown date of 08/16/2026. Discover
    live instead of re-hardcoding its replacement, so the next rename
    doesn't require another code change."""
    fetch_fn = fetch_fn or (lambda k: _fetch_json(
        "https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {k}"}))
    try:
        data = fetch_fn(api_key)
        model_ids = _filter_avoid([m["id"] for m in data.get("data", [])], "groq")
    except Exception as e:
        LAST_DISCOVERY_ERROR["groq"] = f"{type(e).__name__}: {e}"
        return None
    if not model_ids:
        LAST_DISCOVERY_ERROR["groq"] = "catalog returned zero models"
        return None
    LAST_DISCOVERY_ERROR.pop("groq", None)
    return _rank_pick(model_ids, PREFERRED_GROQ_MODELS)


def pick_gemini_model(api_key, fetch_fn=None):
    """gemini-2.5-flash (the model this used to be hardcoded to) is
    deprecated with an official shutdown of 10/16/2026, and some accounts
    report it already returning not-found ahead of that date."""
    fetch_fn = fetch_fn or (lambda k: _fetch_json(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={k}"))
    try:
        data = fetch_fn(api_key)
        model_ids = _filter_avoid([
            m["name"].split("/", 1)[-1] for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ], "gemini")
    except Exception as e:
        LAST_DISCOVERY_ERROR["gemini"] = f"{type(e).__name__}: {e}"
        return None
    if not model_ids:
        LAST_DISCOVERY_ERROR["gemini"] = "catalog returned zero generateContent-capable models"
        return None
    LAST_DISCOVERY_ERROR.pop("gemini", None)
    return _rank_pick(model_ids, PREFERRED_GEMINI_MODELS)


def pick_openrouter_model(api_key, fetch_fn=None):
    """OpenRouter lists free and paid variants of the same underlying
    model under different ids (e.g. an id with a ':free' suffix vs. the
    bare id, which is billed). The old hardcoded id
    (meta-llama/llama-3.1-70b-instruct, no ':free' suffix) was actually
    the *paid* route -- it would have quietly spent OpenRouter credits
    rather than failing. Restrict candidates to ids OpenRouter itself
    prices at $0 before ranking, so this stays a free-tier pick no
    matter what OpenRouter renames things to."""
    fetch_fn = fetch_fn or (lambda k: _fetch_json(
        "https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {k}"}))
    try:
        data = fetch_fn(api_key)
        free_ids = []
        for m in data.get("data", []):
            pricing = m.get("pricing", {})
            try:
                is_free = float(pricing.get("prompt", 1) or 0) == 0 and float(pricing.get("completion", 1) or 0) == 0
            except (TypeError, ValueError):
                is_free = False
            if is_free:
                free_ids.append(m["id"])
    except Exception as e:
        LAST_DISCOVERY_ERROR["openrouter"] = f"{type(e).__name__}: {e}"
        return None
    free_ids = _filter_avoid(free_ids, "openrouter")
    if not free_ids:
        LAST_DISCOVERY_ERROR["openrouter"] = "no $0-priced models found in catalog"
        return None
    LAST_DISCOVERY_ERROR.pop("openrouter", None)
    return _rank_pick(free_ids, PREFERRED_OPENROUTER_MODELS)


# Best-effort published free-tier limits, as verified at the time these
# providers were added to this file. These are NOT queried live -- we've
# already been burned repeatedly by providers changing things without
# notice (model catalogs, retirements), so treat these numbers as "last
# known good", not gospel. Re-verify against the provider's own docs if
# a number here looks suspiciously stale or a quota-based prediction
# doesn't match what you're actually seeing.
KNOWN_QUOTAS = {
    "groq": {"requests_per_day": 1000, "requests_per_minute": 30},
    "gemini": {"requests_per_day": 1500, "requests_per_minute": 15},
    "mistral": {"requests_per_day": None, "requests_per_minute": 1},  # "prototyping only" tier, RPM-limited
    "cerebras": {"requests_per_day": None, "requests_per_minute": None},  # catalog + limits both known to shift
    "openrouter": {"requests_per_day": None, "requests_per_minute": None},  # varies per free model, not fixed
}


def compute_quota_status(provider_name, usage, quotas=KNOWN_QUOTAS):
    """Pure function (no I/O) so it's directly testable: given today's
    usage {"requests": N, "tokens": M} and the known daily request limit
    for a provider, return remaining/used/pct, or None if no published
    daily limit is tracked for that provider."""
    quota = quotas.get(provider_name)
    if not quota or not quota.get("requests_per_day"):
        return None
    limit = quota["requests_per_day"]
    used = usage.get("requests", 0)
    remaining = max(limit - used, 0)
    pct_used = round((used / limit) * 100, 1) if limit else 0.0
    return {"limit": limit, "used": used, "remaining": remaining, "pct_used": pct_used}
