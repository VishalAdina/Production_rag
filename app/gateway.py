import os
import logfire
from typing import Any, Optional
from langchain_openai import ChatOpenAI
from openai import OpenAI
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL

from app.config import settings

# Active LLM configuration
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
BASE_MODEL = "qwen/qwen3.8-27b"

# Portkey Production Routing & Resilience Config (from 02_llm_gateway.ipynb Exp 10)
DEFAULT_PRODUCTION_CONFIG = {
    "strategy": {"mode": "fallback"},
    "request_timeout": 30000,  # 30s hard cap
    "retry": {
        "attempts": 2,
        "on_status_codes": [429, 500, 503],
    },
    "cache": {"mode": "simple"},  # simple exact match caching
}


def get_model_name() -> str:
    """Format model name for Portkey routing if slug exists, otherwise direct model name."""
    slug = (settings.PORTKEY_PRIMARY_SLUG or "").strip()
    if slug:
        return f"@{slug}/{BASE_MODEL}"
    return BASE_MODEL


def get_langchain_llm(feature: str = "default"):
    """
    Returns a LangChain ChatOpenAI LLM instance routed via Portkey Gateway
    (or falling back to direct Groq if Portkey slug is unconfigured).
    """
    logfire.info(f"Initializing LLM gateway for feature: {feature}")

    has_portkey = bool(settings.PORTKEY_API_KEY and (settings.PORTKEY_PRIMARY_SLUG or settings.PORTKEY_VIRTUAL_KEY or settings.PORTKEY_CONFIG_ID))

    if has_portkey:
        try:
            header_kwargs: dict[str, Any] = {
                "api_key": settings.PORTKEY_API_KEY,
                "metadata": {
                    "_user": "rag-pipeline",
                    "feature": feature,
                    "environment": "production",
                },
            }
            if settings.PORTKEY_CONFIG_ID:
                header_kwargs["config"] = settings.PORTKEY_CONFIG_ID
            else:
                header_kwargs["config"] = DEFAULT_PRODUCTION_CONFIG

            if settings.PORTKEY_VIRTUAL_KEY:
                header_kwargs["virtual_key"] = settings.PORTKEY_VIRTUAL_KEY

            headers = createHeaders(**header_kwargs)
            return ChatOpenAI(
                base_url=PORTKEY_GATEWAY_URL,
                api_key=settings.PORTKEY_API_KEY,
                model=get_model_name(),
                max_tokens=500,
                temperature=0.0,
                default_headers=headers,
            )
        except Exception as e:
            logfire.warning(f"Portkey LangChain LLM initialization fallback to Groq: {e}")

    # Fallback to direct Groq
    return ChatOpenAI(
        base_url=GROQ_BASE_URL,
        api_key=settings.GROQ_API_KEY,
        model=BASE_MODEL,
        max_tokens=500,
        temperature=0.0,
    )


class PortkeyGatewayClient:
    """
    Unified Portkey LLM Gateway client with automatic resilience:
    - Portkey caching, retries, and latency timeouts
    - Seamless fallback to direct Groq API if Portkey routing is not yet configured or fails
    - Exposes standard .chat.completions.create interface
    """

    def __init__(self):
        self._groq_client = OpenAI(
            base_url=GROQ_BASE_URL,
            api_key=settings.GROQ_API_KEY,
        )
        self._portkey = None
        self._init_portkey()

    def _init_portkey(self):
        if settings.PORTKEY_API_KEY:
            try:
                config = settings.PORTKEY_CONFIG_ID or DEFAULT_PRODUCTION_CONFIG
                self._portkey = Portkey(
                    api_key=settings.PORTKEY_API_KEY,
                    base_url=PORTKEY_GATEWAY_URL,
                    config=config,
                    virtual_key=settings.PORTKEY_VIRTUAL_KEY if settings.PORTKEY_VIRTUAL_KEY else None,
                )
                logfire.info("🌐 Portkey Gateway initialized with production resilience config.")
            except Exception as e:
                logfire.warning(f"Could not initialize Portkey client ({e}); will use direct Groq.")
                self._portkey = None

    class _ChatProxy:
        def __init__(self, parent: "PortkeyGatewayClient"):
            self._parent = parent

        class _CompletionsProxy:
            def __init__(self, parent: "PortkeyGatewayClient"):
                self._parent = parent

            def create(self, *args, **kwargs):
                # Ensure max_tokens is capped to protect Groq OTPM rate limits
                if "max_tokens" not in kwargs:
                    kwargs["max_tokens"] = 500

                # 1. Attempt Portkey Gateway if configured
                if self._parent._portkey is not None and (settings.PORTKEY_PRIMARY_SLUG or settings.PORTKEY_VIRTUAL_KEY or settings.PORTKEY_CONFIG_ID):
                    target_model = get_model_name()
                    kwargs["model"] = target_model
                    try:
                        with logfire.span("🌐 Portkey Gateway Call", model=target_model):
                            resp = self._parent._portkey.chat.completions.create(*args, **kwargs)
                            return resp
                    except Exception as e:
                        logfire.warning(f"Portkey gateway call encountered issue ({e}); falling back to direct Groq.")

                # 2. Fallback to direct Groq call
                kwargs["model"] = BASE_MODEL
                with logfire.span("⚡ Direct Groq Call (Fallback)", model=BASE_MODEL):
                    return self._parent._groq_client.chat.completions.create(*args, **kwargs)

        @property
        def completions(self):
            return self._CompletionsProxy(self._parent)

    @property
    def chat(self):
        return self._ChatProxy(self)


# Active singleton client used by responder node
portkey_client = PortkeyGatewayClient()


def extract_cache_status(response: Any) -> str:
    """
    Extracts x-portkey-cache-status header from response.
    Returns 'HIT' or 'MISS'.
    """
    try:
        # Check raw response headers
        headers = getattr(response, "_headers", None) or getattr(response, "headers", None)
        if headers and hasattr(headers, "get"):
            status = headers.get("x-portkey-cache-status") or headers.get("X-Portkey-Cache-Status")
            if status:
                return str(status).upper()

        # Check response metadata dictionary if present
        metadata = getattr(response, "metadata", None)
        if isinstance(metadata, dict):
            status = metadata.get("cache_status") or metadata.get("x-portkey-cache-status")
            if status:
                return str(status).upper()
    except Exception:
        pass

    return "MISS"
