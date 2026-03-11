"""Optional LLM tracing integration for LangSmith or Phoenix."""

from __future__ import annotations

import logging

from openai import OpenAI

from settings import AppSettings


LOGGER = logging.getLogger(__name__)
_PHOENIX_CONFIGURED = False


def configure_llm_tracing(settings: AppSettings) -> None:
    """Configure global tracing backends once per process."""
    if settings.llm_tracing_backend == "phoenix":
        _configure_phoenix(settings)


def instrument_openai_client(client: OpenAI, settings: AppSettings) -> OpenAI:
    """Wrap or return the OpenAI client based on tracing settings."""
    if settings.llm_tracing_backend == "langsmith" or settings.langsmith_tracing:
        return _wrap_langsmith(client)
    return client


def _wrap_langsmith(client: OpenAI) -> OpenAI:
    try:
        from langsmith.wrappers import wrap_openai
    except ImportError:
        LOGGER.warning(
            "LangSmith tracing requested but langsmith is not installed; continuing without tracing."
        )
        return client
    return wrap_openai(client)


def _configure_phoenix(settings: AppSettings) -> None:
    global _PHOENIX_CONFIGURED
    if _PHOENIX_CONFIGURED:
        return
    if not settings.phoenix_collector_endpoint:
        raise ValueError(
            "PHOENIX_COLLECTOR_ENDPOINT must be configured when LLM_TRACING_BACKEND=phoenix."
        )
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        LOGGER.warning(
            "Phoenix tracing requested but OpenInference/OpenTelemetry packages are not installed; continuing without tracing."
        )
        return

    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.phoenix_project_name})
    )
    exporter = OTLPSpanExporter(endpoint=settings.phoenix_collector_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    OpenAIInstrumentor().instrument()
    _PHOENIX_CONFIGURED = True
