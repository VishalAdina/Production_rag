import logfire
from qdrant_client import QdrantClient

from app.config import settings
from app.services.retrieval.embedding import embed_query

qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


def search_enterprise_knowledge(query: str, limit: int = 15) -> list[dict]:
    """
    Search Qdrant Cloud collection for relevant chunks using vector similarity.
    """
    with logfire.span("Qdrant Search", query=query, limit=limit):
        try:
            query_vector = embed_query(query)

            # Query using qdrant_client
            results = qdrant_client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=query_vector,
                limit=limit,
            ).points

            docs = []
            for point in results:
                payload = point.payload or {}
                text = payload.get("text", "")
                if text:
                    docs.append({
                        "content": text,
                        "source": payload.get("source", "unknown"),
                        "score": getattr(point, "score", 0.0),
                    })

            logfire.info(f"Retrieved {len(docs)} matching documents from Qdrant.")
            return docs

        except Exception as e:
            logfire.error(f"Failed to query Qdrant: {e}")
            return []
