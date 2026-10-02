import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Gemini
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

    # Qdrant
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")
    QDRANT_URL: str = (
        os.getenv("QDRANT_URL")
        or os.getenv("QDRANT_CLUSTER_ENDPOINT")
        or os.getenv("QUDRANT_CLUSTER_ENDPOINT")
        or "http://localhost:6333"
    )
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "enterprise_rag")

    # Lowercase & compatibility aliases
    @property
    def qdrant_api_key(self) -> str | None:
        return self.QDRANT_API_KEY

    @property
    def qdrant_url(self) -> str:
        return self.QDRANT_URL

    @property
    def drant_collection(self) -> str:
        return self.QDRANT_COLLECTION

    # Jina AI
    JINA_API_KEY: str | None = os.getenv("JINA_API_KEY")

    # Portkey (Optional)
    PORTKEY_API_KEY: str | None = os.getenv("PORTKEY_API_KEY")
    PORTKEY_PRIMARY_SLUG: str = os.getenv("PORTKEY_PRIMARY_SLUG", "")
    PORTKEY_VIRTUAL_KEY: str | None = os.getenv("PORTKEY_VIRTUAL_KEY")
    PORTKEY_CONFIG_ID: str | None = os.getenv("PORTKEY_CONFIG_ID")
    PORTKEY_FALLBACK_SLUG: str | None = os.getenv("PORTKEY_FALLBACK_SLUG")

    # Groq
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY: str | None = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Logfire
    LOGFIRE_TOKEN: str | None = os.getenv("LOGFIRE_TOKEN")
    LOGFIRE_BASE_URL: str | None = os.getenv("LOGFIRE_BASE_URL")


settings = Settings()
