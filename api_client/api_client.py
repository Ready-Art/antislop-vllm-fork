import json
import logging
import threading
from contextlib import ExitStack, contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base_client import BaseApiClient
from core.models import ApiChunkResult

logger = logging.getLogger(__name__)


class _SessionPool:
    """
    One shared ``requests.Session`` for a ``base_url``, plus a count of how many
    requests are currently using it.

    Reference counting is what makes ``reset_connection_pool()`` safe: a session
    with ``active > 0`` is *retired* rather than closed, so its users keep a
    working pool and the last one to finish closes it.  This also keeps every
    client on the *same* session, so the configured pool size still bounds total
    upstream connections.
    """

    __slots__ = ("session", "pool_size", "active", "retired")

    def __init__(self, session: requests.Session, pool_size: int) -> None:
        self.session = session
        self.pool_size = pool_size
        self.active = 0
        self.retired = False


# Sessions are shared per base_url so that N ApiClient instances (one per batch
# worker, or per request in server mode) do not each open their own pool.
_SESSIONS_LOCK = threading.RLock()
_SHARED_POOLS: Dict[str, _SessionPool] = {}


@contextmanager
def _lease_session(base_url: str, pool_size: int) -> Iterator[requests.Session]:
    """
    Lease the shared session for *base_url* for the duration of one request.

    The session cannot be closed by a concurrent ``reset_connection_pool()`` while
    the lease is held: the reset only marks it retired, and the close is deferred
    to whichever request releases it last.  This is what makes a mid-stream reset
    safe.
    """
    pool = None
    try:
        with _SESSIONS_LOCK:
            pool = _SHARED_POOLS.get(base_url)
            if pool is None:
                pool = _SessionPool(ApiClient._build_session(pool_size), pool_size)
                _SHARED_POOLS[base_url] = pool
            pool.active += 1
            session = pool.session
        yield session
    finally:
        if pool is not None:
            _release_pool(pool)


def _release_pool(pool: _SessionPool) -> None:
    """Drop one reference; close the session if it was retired and now idle."""
    with _SESSIONS_LOCK:
        pool.active -= 1
        should_close = pool.retired and pool.active <= 0
    if should_close:
        try:
            pool.session.close()
            logger.info("Closed retired API session (last user finished).")
        except Exception as e:
            logger.warning(f"Error closing retired session: {e}")


class ApiClient(BaseApiClient):
    """
    OpenAI-compatible completions client that keeps a per-instance connection
    pool.  Pool size = <worker threads> + 8, passed in by the caller.
    """

    OPENAI_V1 = "https://api.openai.com/v1"

    # ------------------------------------------------------------------ #
    #  Init                                                              #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        pool_size: int,
    ):
        self._cancel_stream = False      # sampler sets this to abort

        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.completion_endpoint = f"{self.base_url}/completions"

        # -------- adopt the shared pool -------------
        # No session is opened here, and no reference to one is retained: every
        # request leases the *current* session for its base_url (see
        # _lease_session).  Holding a session (or pool) on the instance would go
        # stale as soon as another client reset the pool.
        with _SESSIONS_LOCK:
            pool = _SHARED_POOLS.get(self.base_url)
            if pool is None:
                pool = _SessionPool(self._build_session(pool_size), pool_size)
                _SHARED_POOLS[self.base_url] = pool
            self.pool_size = pool.pool_size

        logger.info(
            f"ApiClient ready (endpoint={self.completion_endpoint}, "
            f"shared_pool={self.pool_size})"
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_session(pool_size: int) -> requests.Session:
        """
        Create a requests.Session backed by a urllib3 connection pool large
        enough for high-thread workloads.
        """
        s = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            max_retries=Retry(
                total=3,
                backoff_factor=0.3,
                status_forcelist=(502, 503, 504),
                allowed_methods=frozenset(["POST"]),
            ),
        )
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        return s

    def cancel_current_stream(self):
        self._cancel_stream = True

    def reset_connection_pool(self):
        """
        Replace the shared connection pool for this ``base_url`` with a fresh one.

        Useful when the server appears stuck or unresponsive and stale keep-alive
        connections need to be discarded.

        This is safe to call while other requests are in flight.  The old session
        is **retired, not closed**: any request still using it keeps a working
        pool, and the session is closed by whichever request releases it last.
        New requests immediately get the replacement session.

        Previously this closed the shared session and dropped it from the cache
        outright, while every other ApiClient kept its own reference to that now
        closed object.  ``Session.close()`` does not break an in-flight response,
        so nothing crashed -- but the clients silently diverged onto separate
        pools, and the configured ``pool_maxsize`` cap stopped applying to the
        whole process.  Each reset could therefore multiply the number of upstream
        connections, which is precisely what the pool sizing is meant to bound.
        """
        pool_size = max(1, int(getattr(self, "pool_size", 10)))

        with _SESSIONS_LOCK:
            old = _SHARED_POOLS.get(self.base_url)
            new_pool = _SessionPool(self._build_session(pool_size), pool_size)
            _SHARED_POOLS[self.base_url] = new_pool

            old_idle = old is None or old.active <= 0
            if old is not None and not old_idle:
                old.retired = True

        if old is None:
            logger.info(f"Connection pool reset (pool_size={pool_size}).")
            return

        if old_idle:
            try:
                old.session.close()
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            logger.info(
                f"Connection pool reset (pool_size={pool_size}, old session closed)."
            )
        else:
            logger.info(
                f"Connection pool reset (pool_size={pool_size}); old session "
                f"retired with {old.active} request(s) still in flight and will "
                f"close when they finish."
            )

    # ------------------------------------------------------------------ #
    #  NEW: streaming generator                                          #
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    #  NEW: streaming generator                                          #
    # ------------------------------------------------------------------ #
    def generate_stream(
        self,
        prompt_text: str,
        max_tokens: int,
        top_logprobs: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        min_p: Optional[float],
        timeout: int,
        stop_sequences: Optional[List[str]],
        **kwargs,
    ):
        """
        Stream /v1/completions and yield one ApiChunkResult **per token**.

        * Supports vLLM, OpenAI, and OpenRouter.
        * Keeps the same provider-specific payload tweaks used in
        generate_chunk().
        * Early-abort when self._cancel_stream is set by the sampler.
        """

        # ---------- build request payload ----------------------------------
        payload: Dict[str, object] = {
            "model":       self.model_name,
            "prompt":      prompt_text,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "top_p":       top_p,
            "top_k":       top_k,
            "stream":      True,
            **kwargs,
        }
        if stop_sequences:
            payload["stop"] = stop_sequences

        # -- provider switches (mirror of generate_chunk) -------------------
        if self.base_url == self.OPENAI_V1:
            payload["logprobs"] = top_logprobs                 # OpenAI wants int
        elif self.base_url == "https://openrouter.ai/api/v1":
            payload.update({
                "min_p":        min_p,
                "logprobs":     True,
                "top_logprobs": top_logprobs,
                "provider": {
                    "order": ["Fireworks"],
                    "allow_fallbacks": False,
                },
            })
        else:                                                  # vLLM family
            payload.update({
                "min_p":        min_p,
                "logprobs":     top_logprobs,                  # ← vLLM wants int
                "top_logprobs": top_logprobs,
            })

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        logger.debug(f"STREAM POST {self.completion_endpoint}")
        logger.debug(f"Payload: {json.dumps(payload)[:500]}")

        # ---------- open HTTP stream ---------------------------------------
        last_finish_reason = None
        try:
            # Lease a session for the whole life of this stream, and open the
            # response through it.  Both are released when this generator is
            # exhausted or closed, so a concurrent reset_connection_pool() retires
            # the session instead of closing it and cannot tear this stream down
            # mid-flight.
            with ExitStack() as stack:
                session = stack.enter_context(
                    _lease_session(self.base_url, self.pool_size)
                )
                resp = stack.enter_context(session.post(
                    self.completion_endpoint,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=timeout,
                ))
                resp.raise_for_status()

                for raw in resp.iter_lines(decode_unicode=True):

                    # ---- sampler asked to abort? ------------------------------
                    if self._cancel_stream:
                        resp.close()
                        self._cancel_stream = False
                        break

                    if not raw or raw == "\n":
                        continue

                    if raw.startswith("data: "):
                        raw = raw[6:]

                    if raw.strip() == "[DONE]":
                        # signal natural end-of-stream
                        yield ApiChunkResult(
                            generated_text="",
                            finish_reason="stream_end",
                        )
                        break

                    # ---------- parse SSE JSON ---------------------------------
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning("Malformed SSE line: %s", raw[:160])
                        continue

                    # Pass-through OpenAI-style streamed error packets
                    if isinstance(obj, dict) and obj.get("error"):
                        err = obj["error"]
                        msg = (err.get("message") if isinstance(err, dict) else str(err)) or "Upstream stream error"
                        raise requests.HTTPError(msg, response=resp)

                    if not obj.get("choices"):
                        continue
                    choice = obj["choices"][0]

                    # Track finish_reason from the last chunk
                    last_finish_reason = choice.get("finish_reason")

                    # -- incremental text fragment ------------------------------
                    generated_text = choice.get("text", "")
                    # (some providers use .delta.content / .message.content – those
                    # branches are left in generate_chunk; for vLLM we only need .text)

                    # -- extract logprobs --------------------------------------
                    token_strings: List[str] = []
                    rel_logprobs: Dict[int, List[Tuple[str, float]]] = {}

                    lp_obj = choice.get("logprobs")
                    if lp_obj and isinstance(lp_obj, dict):
                        token_strings = lp_obj.get("tokens", [])
                        # vLLM: top_logprobs is list[dict] aligned with tokens
                        for idx, alt_dict in enumerate(lp_obj.get("top_logprobs", [])):
                            if isinstance(alt_dict, dict):
                                rel_logprobs[idx] = list(alt_dict.items())

                    yield ApiChunkResult(
                        generated_text=generated_text,
                        token_strings=token_strings,
                        logprobs=rel_logprobs,
                        finish_reason=last_finish_reason,
                    )

        except requests.exceptions.Timeout:
            logger.warning("Stream timed out. Yielding final chunk with finish_reason='timeout'.")
            # Yield a final chunk to signal the end, even if incomplete
            yield ApiChunkResult(
                generated_text="",
                token_strings=[],
                logprobs={},
                finish_reason="timeout",
            )
            raise
        except Exception as e:
            logger.warning(f"Stream ended unexpectedly: {e}")
            # If we have a finish_reason from earlier, the stream likely completed
            # Otherwise, signal an abnormal end
            if not last_finish_reason:
                yield ApiChunkResult(
                    generated_text="",
                    token_strings=[],
                    logprobs={},
                    finish_reason="error",
                )
            raise



    # ------------------------------------------------------------------ #
    #  Core call                                                         #
    # ------------------------------------------------------------------ #
    def generate_chunk(
        self,
        prompt_text: str,
        max_tokens: int,
        top_logprobs: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        min_p: Optional[float],
        timeout: int,
        stop_sequences: Optional[List[str]],
        **kwargs,
    ) -> ApiChunkResult:

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, object] = {
            "model":       self.model_name,
            "prompt":      prompt_text,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "top_p":       top_p,
            "top_k":       top_k,
            "stream":      False,
            **kwargs,
        }
        if stop_sequences:
            payload["stop"] = stop_sequences

        # API-flavour switches
        if self.base_url == self.OPENAI_V1:
            payload["logprobs"] = top_logprobs
        elif self.base_url == "https://openrouter.ai/api/v1":
            payload.update({"min_p": min_p, "logprobs": True, "top_logprobs": top_logprobs})
            # few providers support top logprobs or completions properly
            payload["provider"] = {
                "order": [
                    #"DeepInfra", # llama-3.1-8b, mistral-small-3, qwen-72b
                    #"Mistral" # mistral-small-3
                    #"Lambda", # llama-3.1-8b
                    #"NovitaAI",  # qwen-72b, llama-3.1-8b
                    #"Nebius AI Studio", # qwen-72b
                    #"Hyperbolic", # qwen-72b
                    #"inference.net", # llama-3.1-8b
                    #"Groq", # llama 3.1 8b
                    #"inference.net",
                    "Fireworks"
                ],
                "allow_fallbacks": False
            }
        else:  # vLLM-style
            payload.update({"min_p": min_p, "logprobs": top_logprobs})

        logger.debug(f"POST {self.completion_endpoint}")
        logger.debug(f"Payload: {json.dumps(payload)[:500]}")

        try:
            # Lease the session for the request and the body read only; the
            # response is fully buffered by resp.json(), so the lease can be
            # released before the (possibly slow) payload parsing below.  A
            # concurrent reset_connection_pool() retires the session rather than
            # closing it while this request is using it.
            with _lease_session(self.base_url, self.pool_size) as session:
                resp = session.post(
                    self.completion_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()
                data = resp.json()
        except requests.exceptions.Timeout:
            logger.error("API request timed out.")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            try:
                logger.error(f"Error detail: {resp.json()}")
            except Exception:
                pass
            raise

        logger.debug(f"Raw response: {json.dumps(data)[:2000]}")

        # Pass-through OpenAI-style error embedded in a 200 OK
        if "error" in data and data["error"]:
            err = data["error"]
            msg = (err.get("message") if isinstance(err, dict) else str(err)) or "Upstream error"
            # Attach original response so callers can propagate status/body
            raise requests.HTTPError(msg, response=resp)

        # Treat “no choices” as an upstream failure
        if not data.get("choices"):
            raise requests.HTTPError("Upstream returned no choices", response=resp)


        choice = data["choices"][0]

        # -------- generated text --------
        # supports both /completions (legacy) and /chat/completions payloads
        if "text" in choice:                              # legacy completions
            generated_text = choice["text"]
        elif "message" in choice and isinstance(choice["message"], dict):  # chat completions
            generated_text = choice["message"].get("content", "")
        else:
            generated_text = ""

        finish_reason = choice.get("finish_reason")


        # ------------ extract logprobs ------------
        token_strings: List[str] = []
        relative_logprobs: Dict[int, List[Tuple[str, float]]] = {}

        lp_obj = choice.get("logprobs")
        if lp_obj:
            if "tokens" in lp_obj:  # legacy format
                token_strings = lp_obj["tokens"]
                for idx, alt_dict in enumerate(lp_obj.get("top_logprobs", [])):
                    if isinstance(alt_dict, dict):
                        relative_logprobs[idx] = list(alt_dict.items())
            elif "content" in lp_obj and isinstance(lp_obj["content"], list):  # chat format
                for idx, item in enumerate(lp_obj["content"]):
                    tok = item.get("token")
                    if tok is None:
                        continue
                    token_strings.append(tok)
                    alt_list = item.get("top_logprobs")
                    if isinstance(alt_list, list):
                        relative_logprobs[idx] = [
                            (alt.get("token"), alt.get("logprob"))
                            for alt in alt_list
                            if alt.get("token") is not None
                        ]

        if not token_strings:
            logger.warning(
                "Logprobs present but no token strings parsed; "
                "back-tracking disabled for this chunk."
            )

        logger.info(
            f"Received {len(token_strings)} token strings "
            f"(finish_reason={finish_reason or 'None'})."
        )

        return ApiChunkResult(
            generated_text=generated_text,
            token_strings=token_strings,
            logprobs=relative_logprobs,
            finish_reason=finish_reason,
        )
