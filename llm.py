"""Grounded answer generation via an OpenAI-compatible LLM provider.

Works with a local llama.cpp server, OpenAI, or Google Gemini (which exposes an
OpenAI-compatible endpoint). Provider, base URL, model and API key are resolved
from environment variables in `config.py`; the provider/model can also be
overridden per request (see `/api/ask`).

Environment variables (all optional):
    RAG_LLM_PROVIDER    local | openai | gemini          default: local
    RAG_LLM_MODEL       model name, or "auto"            default: provider preset
    RAG_LLM_API_KEY     explicit key for any provider    default: provider env var
    OPENAI_API_KEY      used by the openai provider
    GEMINI_API_KEY      used by the gemini provider (GOOGLE_API_KEY also works)
    RAG_LLM_BASE_URL    override the provider endpoint
"""
from config import USE_LLM, get_llm_settings

NOT_IN_CORPUS = "NOT_IN_CORPUS"

SYSTEM_PROMPT = (
    "You are a precise research assistant for Indian government gender budget "
    "documents. The sources are rows/tables extracted from budget PDFs, CSVs and "
    "spreadsheets; column headings are often missing, and several years of "
    "similar documents may be present.\n"
    "RULES:\n"
    "- Answer in ONE or TWO sentences, in English unless the question "
    "explicitly asks for another language, and only for the year and "
    "jurisdiction asked. Do not discuss other years.\n"
    "- Use the source whose file name matches the year asked in the question.\n"
    "- For a multi-column numeric row, the last number is the most recent "
    "year's outlay.\n"
    "- Give the figure with its unit (crore / lakh / thousand / rupees). If the "
    "document is marked (IN THOUSANDS), say so; when the question asks for a "
    "different unit, convert explicitly (1 crore = 10,000 thousand, 1 lakh = "
    "100 thousand) and show the original number too.\n"
    "- If the same scheme appears under both Part A and Part B, give both "
    "figures and say which part each is.\n"
    "- If a focus note is provided with the question, use it only to choose the "
    "right year, part, unit or scope. It is guidance, not evidence, and must "
    "never be quoted or cited.\n"
    "- Cite the source inline as [S1].\n"
    "- Reply with exactly " + NOT_IN_CORPUS + " ONLY if no source mentions the "
    "scheme or item at all."
)


def _client(settings):
    from openai import OpenAI

    return OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key or "not-needed",
        timeout=settings.timeout,
    )


def _resolve_model(client, settings):
    if settings.model and settings.model != "auto":
        return settings.model
    try:
        models = client.models.list().data
        if models:
            return models[0].id
    except Exception:  # noqa: BLE001 - fall back to a placeholder
        pass
    return "local-model"


def _session(provider=None, model=None):
    """Return (settings, client, model_name) for the chosen provider."""
    settings = get_llm_settings(provider, model)
    if settings.needs_api_key and not settings.api_key:
        env_hint = {
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY (or GOOGLE_API_KEY)",
        }.get(settings.provider, "RAG_LLM_API_KEY")
        raise RuntimeError(
            f"No API key configured for the {settings.provider} provider. "
            f"Set {env_hint} or RAG_LLM_API_KEY."
        )
    client = _client(settings)
    return settings, client, _resolve_model(client, settings)


def get_llm_info(provider=None, model=None):
    """Non-secret description of the resolved LLM configuration."""
    settings = get_llm_settings(provider, model)
    return {
        "provider": settings.provider,
        "model": settings.model,
        "base_url": settings.base_url,
        "api_key_set": bool(settings.api_key),
        "needs_api_key": settings.needs_api_key,
    }


def build_context(hits):
    lines = []
    for i, hit in enumerate(hits, start=1):
        meta = hit.get("metadata", {})
        loc = meta.get("file", "unknown")
        if meta.get("page") is not None:
            loc += f", page {meta['page']}"
        if meta.get("sheet") is not None:
            loc += f", sheet {meta['sheet']}"
        if meta.get("row") is not None:
            loc += f", row {meta['row']}"
        lines.append(f"[S{i}] ({loc})\n{hit['text']}")
    return "\n\n".join(lines)


def llm_ready(provider=None, model=None):
    if not USE_LLM:
        return False
    try:
        settings = get_llm_settings(provider, model)
        if settings.needs_api_key and not settings.api_key:
            return False
        # Keep the health check snappy: short timeout, no retries.
        client = _client(settings).with_options(
            timeout=min(settings.timeout, 10.0), max_retries=0
        )
        client.models.list()
        return True
    except Exception:  # noqa: BLE001 - health check must not crash
        return False


def generate(question, hits, provider=None, model=None, test_hint=None):
    """Return (answer_text, model_name). Raises on transport/config errors.

    `test_hint` is the optional dev-set `tests` note. It tells the model what
    the question is testing (unit conversion, year ambiguity, ...) and is
    deliberately kept out of the sources so it cannot be cited as evidence.
    """
    settings, client, model_name = _session(provider, model)
    focus = ""
    if test_hint:
        focus = (
            "\n\nEvaluation focus (guidance only — this is NOT a source and "
            "contains no facts to quote):\n"
            f"{test_hint}"
        )
    user = (
        f"Question: {question}"
        f"{focus}"
        f"\n\nSources:\n{build_context(hits)}\n\n"
        "Answer (or NOT_IN_CORPUS if the sources do not contain it):"
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
    text = (response.choices[0].message.content or "").strip()
    return text, model_name


def translate_to_english(text, provider=None, model=None):
    """Translate a (e.g. Hindi) question to English for retrieval."""
    settings, client, model_name = _session(provider, model)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Translate the user's question into English. Preserve scheme "
                    "names and numbers. Output only the translation, nothing else."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=160,
    )
    return (response.choices[0].message.content or "").strip()
