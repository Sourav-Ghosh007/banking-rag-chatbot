import openai
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"

def get_embeddings(text: str) -> list[float]:
    try:
        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise

def get_batch_embeddings(texts: list[str]) -> list[list[float]]:
    try:
        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
        raise