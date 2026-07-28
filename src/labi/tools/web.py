"""
Web observation tools -- search() and fetch() give Labi its missing
"observe the external world" step in the plan / code / validate loop.

Kept dependency-free (stdlib urllib only, matching adaptive_registry.py's
own _fetch_json helper) and safety-conscious: fetch() only allows http(s)
and refuses to resolve to loopback/private/link-local/reserved addresses,
since a URL reaching this function can originate from an LLM-produced
plan rather than a trusted human -- blindly fetching "whatever URL the
model said" is exactly the kind of implicit trust that has already bitten
this project once (see adaptive_registry.py's model-name denylists), just
in a more dangerous place (SSRF) instead of a merely-annoying one.
"""

import ipaddress
import json as _json
import os
import re
import socket
import urllib.parse
import urllib.request
from html.parser import HTMLParser

DEFAULT_TIMEOUT = 10
MAX_FETCH_BYTES = 500_000
USER_AGENT = "labi-agent/0.1 (+https://github.com/jirokanz/labi)"

# Populated with the last real reason a search()/fetch() call declined or
# failed, so callers can surface something more useful than "web search
# didn't work" -- same pattern as LAST_DISCOVERY_ERROR in
# adaptive_registry.py.
LAST_ERROR = {}


class _TextExtractor(HTMLParser):
    """Minimal, dependency-free HTML-to-text extractor -- drops
    <script>/<style>/<head> content and all tags, keeps everything else.
    Not a full readability algorithm; the only consumer is
    extract_content(), which feeds the result straight into an LLM
    prompt, so "readable enough to summarize" is the bar, not "clean
    article text"."""

    _SKIP_TAGS = {"script", "style", "noscript", "svg", "head"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skipping = 0
        self.chunks = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skipping += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skipping > 0:
            self._skipping -= 1

    def handle_data(self, data):
        if not self._skipping and data.strip():
            self.chunks.append(data.strip())


def extract_content(html, max_chars=8000):
    """Strip an HTML page down to plain, whitespace-collapsed text,
    truncated to max_chars. Truncation matters here for the same reason
    it matters in stream_generate's max_tokens elsewhere in this repo:
    this text goes straight into a prompt, so an untruncated page could
    silently blow the token/cost budget of whatever provider consumes
    it."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass  # malformed HTML -- keep whatever text was parsed before the error
    text = re.sub(r"\s+", " ", " ".join(parser.chunks)).strip()
    return text[:max_chars]


def _is_safe_host(hostname):
    """Refuse anything that resolves to a loopback/private/link-local/
    reserved/multicast address, so a plan that says e.g. 'fetch
    http://169.254.169.254/latest/meta-data/' (a cloud metadata endpoint)
    or 'http://localhost:6379/' can't turn this tool into an SSRF
    vector. Resolves the hostname itself rather than pattern-matching the
    string, since 'localhost', '127.1', decimal/hex IP encodings, and
    DNS rebinding are all ways to smuggle a private address past a
    string check but not past an actual resolve."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def fetch(url, timeout=DEFAULT_TIMEOUT, max_bytes=MAX_FETCH_BYTES, fetch_fn=None):
    """Fetch a URL's response body as text, with scheme + SSRF guards.
    Returns None (with LAST_ERROR['fetch'] set) rather than raising --
    matching the 'degrade, don't crash' pattern the pick_*_model
    functions use elsewhere in this repo -- since one bad URL in an
    autonomous loop shouldn't take the whole task down.

    fetch_fn is injectable for testing (same shape as pick_*_model's
    fetch_fn): given the url, return raw bytes.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        LAST_ERROR["fetch"] = f"unsupported scheme: {parsed.scheme!r}"
        return None
    if not parsed.hostname or not _is_safe_host(parsed.hostname):
        LAST_ERROR["fetch"] = f"refused unsafe/unresolvable host: {parsed.hostname!r}"
        return None

    def _default_fetch(u):
        req = urllib.request.Request(u, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(max_bytes + 1)

    fetch_fn = fetch_fn or _default_fetch
    try:
        raw = fetch_fn(url)
    except Exception as e:
        LAST_ERROR["fetch"] = f"{type(e).__name__}: {e}"
        return None
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    LAST_ERROR.pop("fetch", None)
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return raw.decode("latin-1", errors="replace")


def search(query, api_key=None, max_results=5, fetch_json_fn=None):
    """Live web search via the Brave Search API (free tier at time of
    writing -- see https://brave.com/search/api/). A JSON API is used
    instead of scraping a search engine's result HTML directly: HTML
    scraping breaks silently on any markup change, which is precisely
    the "hardcoded assumption about someone else's system" failure mode
    this codebase has already been burned by repeatedly (see
    adaptive_registry.py's model-catalog discovery history) -- a stable
    JSON contract avoids repeating that mistake here.

    Returns a list of {"title", "url", "snippet"} dicts (possibly
    empty), or None (with LAST_ERROR['search'] set) if no key is
    configured or the request fails. Callers should treat None as "web
    search unavailable this session" -- the same degrade-gracefully
    contract as a missing provider API key elsewhere in this repo -- and
    fall back to answering from the model's own knowledge.
    """
    api_key = api_key or os.environ.get("BRAVE_API_KEY")
    if not api_key:
        LAST_ERROR["search"] = "BRAVE_API_KEY not set"
        return None

    def _default_fetch_json(q):
        url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(
            {"q": q, "count": max_results}
        )
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
                "User-Agent": USER_AGENT,
            },
        )
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
            return _json.loads(resp.read())

    fetch_json_fn = fetch_json_fn or _default_fetch_json
    try:
        data = fetch_json_fn(query)
    except Exception as e:
        LAST_ERROR["search"] = f"{type(e).__name__}: {e}"
        return None

    results = []
    for item in data.get("web", {}).get("results", [])[:max_results]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
        })
    LAST_ERROR.pop("search", None)
    return results
