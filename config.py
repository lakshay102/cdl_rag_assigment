"""Shared configuration and helpers for the Gender Budget RAG system."""
import os
from dataclasses import dataclass


def _load_dotenv():
    """Load `.env` from the project root when python-dotenv is installed.

    Values already present in the real OS environment win over `.env`, which is
    python-dotenv's default behaviour.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional
        return
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


_load_dotenv()


CORPUS_DIR = os.environ.get("RAG_CORPUS_DIR", "corpus")
DB_DIR = os.environ.get("RAG_DB_DIR", "chroma_db")
COLLECTION_NAME = os.environ.get("RAG_COLLECTION", "gender_budget")

# Retrieval / answering knobs
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
# How many candidates to pull from the vector DB before lexical reranking.
RERANK_POOL = int(os.environ.get("RAG_RERANK_POOL", "60"))
# Cosine distance below which we consider the corpus to actually contain the
# answer. Above this we return answered=false instead of guessing.
MAX_DISTANCE = float(os.environ.get("RAG_MAX_DISTANCE", "0.65"))
# When the client asks for every chunk (k = 0, the "all" option in the UI) we
# rank the whole corpus but only hand a bounded set to the LLM and the page:
# an 8k-context model cannot take 1800+ chunks, and rendering them would freeze
# the browser. This is the size of that bounded set.
ALL_MAX_RESULTS = int(os.environ.get("RAG_ALL_MAX_RESULTS", "20"))

# PDF chunking.
#
# The text is tabular, not prose: a 800-character chunk is roughly 200 tokens
# for English prose but more for rows full of numbers. The bundled
# all-MiniLM-L6-v2 embedding model truncates at 256 tokens, so chunks that are
# much larger than this silently lose their tail. Measure before changing:
#     python ingest.py --dry-run
PDF_CHUNK_CHARS = int(os.environ.get("RAG_PDF_CHUNK_CHARS", "800"))
PDF_CHUNK_OVERLAP = int(os.environ.get("RAG_PDF_CHUNK_OVERLAP", "150"))
# Very short page tails are merged into the previous chunk instead of becoming
# a near-empty vector.
PDF_CHUNK_MIN_CHARS = int(os.environ.get("RAG_PDF_CHUNK_MIN_CHARS", "200"))

# Repeat the page's unit/column-header lines at the top of every chunk after the
# first one, so a row that starts mid-page still says e.g. "(IN THOUSANDS)".
PDF_HEADER_CONTEXT = os.environ.get(
    "RAG_PDF_HEADER_CONTEXT", "1"
).lower() not in ("0", "false", "no")
PDF_HEADER_CHARS = int(os.environ.get("RAG_PDF_HEADER_CHARS", "240"))

# The embedding model's effective limit. Only used to warn during --dry-run.
EMBED_MAX_TOKENS = int(os.environ.get("RAG_EMBED_MAX_TOKENS", "256"))

# --------------------------------------------------------------------------- #
# LLM provider settings
# --------------------------------------------------------------------------- #
# Every provider is spoken to through its OpenAI-compatible endpoint, so the
# same code works with a local llama.cpp server, OpenAI, and Google Gemini.
#
#   local  -> llama.cpp / LM Studio / Ollama, no real API key needed
#   openai -> https://api.openai.com/v1              (OPENAI_API_KEY)
#   gemini -> Google AI Studio OpenAI-compat endpoint (GEMINI_API_KEY)
#
# Pick the provider with RAG_LLM_PROVIDER. Any provider can be overridden with
# RAG_LLM_BASE_URL / RAG_LLM_MODEL / RAG_LLM_API_KEY.
USE_LLM = os.environ.get("RAG_USE_LLM", "1").lower() not in ("0", "false", "no")
LLM_PROVIDER = os.environ.get("RAG_LLM_PROVIDER", "local").strip().lower()

LLM_TEMPERATURE = float(os.environ.get("RAG_LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.environ.get("RAG_LLM_MAX_TOKENS", "512"))
LLM_TIMEOUT = float(os.environ.get("RAG_LLM_TIMEOUT", "120"))

# Empty string means "use the provider preset below".
LLM_BASE_URL_OVERRIDE = os.environ.get("RAG_LLM_BASE_URL", "").strip()
LLM_MODEL_OVERRIDE = os.environ.get("RAG_LLM_MODEL", "").strip()
LLM_API_KEY_OVERRIDE = os.environ.get("RAG_LLM_API_KEY", "").strip()

# Presets. "api_key_env" is checked in order when RAG_LLM_API_KEY is not set;
# "model" is the default model name ("auto" = ask the server what it has).
LLM_PROVIDERS = {
    "local": {
        "base_url": "http://localhost:5555/v1",
        "model": "auto",
        "api_key_env": ("RAG_LLM_API_KEY",),
        "default_api_key": "sk-local",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": ("OPENAI_API_KEY", "RAG_LLM_API_KEY"),
        "default_api_key": "",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.1-flash-lite",
        "api_key_env": ("GEMINI_API_KEY", "GOOGLE_API_KEY", "RAG_LLM_API_KEY"),
        "default_api_key": "",
    },
}

# Friendly aliases so `RAG_LLM_PROVIDER` is forgiving.
LLM_PROVIDER_ALIASES = {
    "google": "gemini",
    "google-gemini": "gemini",
    "google_ai": "gemini",
    "openai-compatible": "local",
    "openai_compatible": "local",
    "llama": "local",
    "llama.cpp": "local",
    "llamacpp": "local",
}


@dataclass(frozen=True)
class LLMSettings:
    """Fully resolved LLM connection details."""

    provider: str
    base_url: str
    model: str
    api_key: str
    temperature: float
    max_tokens: int
    timeout: float

    @property
    def needs_api_key(self):
        return self.provider in ("openai", "gemini")


def get_llm_settings(provider=None, model=None):
    """Resolve provider/base URL/model/key, allowing per-request overrides.

    Precedence for the model name:
        model argument > RAG_LLM_MODEL > provider default.
    Precedence for the API key:
        RAG_LLM_API_KEY > provider-specific env vars > provider default.
    """
    name = (provider or LLM_PROVIDER or "local").strip().lower()
    name = LLM_PROVIDER_ALIASES.get(name, name)
    spec = LLM_PROVIDERS.get(name)
    if spec is None:
        choices = ", ".join(sorted(LLM_PROVIDERS))
        raise ValueError(f"Unknown LLM provider {name!r}. Choose one of: {choices}")

    api_key = LLM_API_KEY_OVERRIDE
    if not api_key:
        for env_name in spec["api_key_env"]:
            value = os.environ.get(env_name, "").strip()
            if value:
                api_key = value
                break
    if not api_key:
        api_key = spec.get("default_api_key", "")

    chosen_model = (model or LLM_MODEL_OVERRIDE or spec["model"] or "auto").strip()

    return LLMSettings(
        provider=name,
        base_url=LLM_BASE_URL_OVERRIDE or spec["base_url"],
        model=chosen_model,
        api_key=api_key,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        timeout=LLM_TIMEOUT,
    )


def get_embedding_function():
    """Local, offline embedding function bundled with ChromaDB.

    Uses the all-MiniLM-L6-v2 ONNX model (downloaded once, on first use).
    """
    from chromadb.utils import embedding_functions

    return embedding_functions.DefaultEmbeddingFunction()
