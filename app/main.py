# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire

from app.config import settings

_logfire_base_url = getattr(settings, "LOGFIRE_BASE_URL", None)
_logfire_token = getattr(settings, "LOGFIRE_TOKEN", None)

if not _logfire_base_url and _logfire_token:
    if _logfire_token.startswith("pylf_v2_eu_"):
        _logfire_base_url = "https://logfire-eu.pydantic.dev"

if _logfire_token:
    logfire.configure(
        token=_logfire_token,
        console=False,
        advanced=logfire.AdvancedOptions(base_url=_logfire_base_url) if _logfire_base_url else None,
    )
else:
    logfire.configure(send_to_logfire=False, console=False)

import time
import uuid
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from app.agents.graph import build_graph
from app.guardrails import guard, initialize_rails
# from app.health import router as health_router
# from app.logging import set_request_id
# from app.services.health.connection_checker import check_all_connections, log_connection_summary

# Custom Prometheus metrics
RAG_REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total number of /query requests",
    ["status"],
)
RAG_REQUEST_DURATION = Histogram(
    "rag_request_duration_seconds",
    "Latency of /query requests in seconds",
)
GUARDRAILS_BLOCKS_TOTAL = Counter(
    "guardrails_blocks_total",
    "Number of requests blocked or allowed by guardrails",
    ["blocked"],
)

_security = HTTPBearer(auto_error=False)


def _init_rate_limiter():
    """Initialize rate limiting. Use Redis in production; fall back to in-memory storage locally."""
    from limits.storage import RedisStorage
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.extension import _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address

    redis_url = getattr(settings, "redis_url", None) or getattr(settings, "REDIS_URL", None)
    if redis_url:
        try:
            storage = RedisStorage(redis_url)
            if not storage.check() or not storage.storage.ping():
                raise ConnectionError("Redis did not respond to ping")
            app.state.limiter = Limiter(key_func=get_remote_address, storage_uri=redis_url)
            app.state.rate_limiter_storage = "redis"
            logfire.info("🚦 Rate limiting initialized via Redis.")
            app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
            return True
        except Exception as e:
            logfire.warning(f"⚠️ Redis unavailable ({e}); using in-memory rate limiting.")

    app.state.limiter = Limiter(key_func=get_remote_address)
    app.state.rate_limiter_storage = "memory"
    logfire.info("🚦 Rate limiting initialized via in-memory storage.")
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    return True


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(_security)):
    api_key = getattr(settings, "API_KEY", None) or getattr(settings, "RAG_API_KEY", None)
    if not api_key:
        return None

    if not credentials or credentials.credentials != api_key:
        logfire.warning("🔒 Unauthorized /query request: invalid or missing API key.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def _get_limiter_rule(times: int, seconds: int) -> str:
    if seconds % 60 == 0:
        return f"{times}/{seconds // 60}minute"
    if seconds % 3600 == 0:
        return f"{times}/{seconds // 3600}hour"
    return f"{times}/{seconds}second"


class _AppLimiter:
    def limit(self, rule_or_callable):
        def decorator(func):
            import functools

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                limiter = getattr(app.state, "limiter", None)
                if limiter is None:
                    return func(*args, **kwargs)

                rule = rule_or_callable() if callable(rule_or_callable) else rule_or_callable
                return limiter.limit(rule)(func)(*args, **kwargs)

            return wrapper

        return decorator


app_limiter = _AppLimiter()


def rate_limit(times: int = None, seconds: int = None):
    def _resolve_rule() -> str:
        t = times or getattr(settings, "RATE_LIMIT_PER_MINUTE", 100)
        s = seconds or 60
        return _get_limiter_rule(t, s)

    return app_limiter.limit(_resolve_rule)


app = FastAPI(title="Enterprise Agentic RAG API")
# app.include_router(health_router)


@app.middleware("http")
async def normalize_slashes(request: Request, call_next):
    if "//" in request.scope.get("path", ""):
        import re
        request.scope["path"] = re.sub(r"/+", "/", request.scope["path"])
    return await call_next(request)


Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.on_event("startup")
def startup_event():
    initialize_rails()
    app.state.rag_agent = build_graph()
    app.state.rate_limiter_enabled = _init_rate_limiter()

    # connection_results = check_all_connections()
    # all_healthy = log_connection_summary(connection_results)
    # strict = getattr(settings, "STRICT_STARTUP", False)
    # if strict and not all_healthy:
    #     failed = [name for name, r in connection_results.items() if not r.healthy]
    #     raise RuntimeError(f"STRICT_STARTUP enabled; failing services: {', '.join(failed)}")

    api_key = getattr(settings, "API_KEY", None) or getattr(settings, "RAG_API_KEY", None)
    if not api_key:
        logfire.warning("🔓 RAG_API_KEY is not set — /query is open to anyone. Set it in production.")


class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"


@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/graph")
def get_graph_image(_api_key: str = Depends(verify_api_key)):
    try:
        png_bytes = app.state.rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}


@app.post("/query")
@rate_limit()
def query(
    request: Request,
    body: QueryRequest,
    _api_key: str = Depends(verify_api_key),
):
    q = body.q
    thread_id = body.thread_id
    request_id = str(uuid.uuid4())
    # set_request_id(request_id)

    start = time.perf_counter()
    with logfire.span("🔍 /query", request_id=request_id, thread_id=thread_id):
        # Guardrails check
        rail_fired, rail_response = guard(q)
        if rail_fired:
            GUARDRAILS_BLOCKS_TOTAL.labels(blocked="true").inc()
            RAG_REQUESTS_TOTAL.labels(status="blocked").inc()
            RAG_REQUEST_DURATION.observe(time.perf_counter() - start)
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": [],
            }

        GUARDRAILS_BLOCKS_TOTAL.labels(blocked="false").inc()

        try:
            rag_agent = getattr(app.state, "rag_agent", None)
            if rag_agent is None:
                rag_agent = build_graph()
                app.state.rag_agent = rag_agent

            initial_state = {
                "messages": [{"role": "user", "content": q}],
                "current_query": q,
                "documents": [],
                "plan": ["Start"],
                "status": "Initializing Graph...",
            }
            config = {"configurable": {"thread_id": thread_id}}
            final_output = rag_agent.invoke(initial_state, config=config)

            RAG_REQUESTS_TOTAL.labels(status="success").inc()
            RAG_REQUEST_DURATION.observe(time.perf_counter() - start)
            logfire.info(
                "✅ RAG pipeline completed",
                request_id=request_id,
                thread_id=thread_id,
            )
            return {
                "question": q,
                "answer": final_output.get("final_answer"),
                "thought_process": final_output.get("plan"),
                "status": final_output.get("status"),
                "sources": final_output.get("documents", []),
            }
        except Exception as e:
            RAG_REQUESTS_TOTAL.labels(status="error").inc()
            RAG_REQUEST_DURATION.observe(time.perf_counter() - start)
            logfire.error(
                f"❌ RAG pipeline failed: {e}",
                request_id=request_id,
                thread_id=thread_id,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "request_id": request_id,
                    "status": "error",
                    "message": "Failed to process request. Please try again later.",
                    "detail": f"{type(e).__name__}: {str(e)}",
                },
            )
