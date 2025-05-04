import sys
from typing import List, Union
from openai import AsyncOpenAI
from loguru import logger
import numpy as np
from ..core.config import settings
from .cache import cache

# Configure Loguru for JSON output to stdout
logger.remove()
logger.add(sys.stdout, serialize=True, enqueue=True)

# Configure OpenAI
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# --- Reusable Embedding Function ---
async def get_embeddings(texts: Union[str, List[str]], model: str = "text-embedding-3-small") -> Union[List[float], List[List[float]]]:
    """
    Generates embeddings for a single text or a list of texts.

    Args:
        texts: A single string or a list of strings to embed.
        model: The embedding model to use.

    Returns:
        A single embedding vector if input is a string, or a list of embedding vectors if input is a list of strings.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    try:
        # Check cache first
        cached_embeddings = await cache.get_embeddings(texts, model)
        if cached_embeddings is not None:
            logger.info("Retrieved embeddings from cache.")
            return cached_embeddings

        is_single_text = isinstance(texts, str)
        input_data = [texts] if is_single_text else texts

        if not input_data:  # Handle empty list input
            logger.warning("get_embeddings called with empty list.")
            return []

        logger.info(f"Requesting embeddings for {len(input_data)} text(s) using model {model}.")
        response = await client.embeddings.create(
            model=model,
            input=input_data
        )

        embeddings = [data.embedding for data in response.data]
        result = embeddings[0] if is_single_text else embeddings

        # Cache the embeddings
        await cache.set_embeddings(texts, model, result)
        logger.info(f"Successfully retrieved and cached {len(embeddings)} embedding(s).")

        return result
    except Exception as e:
        logger.exception(f"Error getting embeddings with model {model}.")
        raise  # Re-raise the exception to be handled upstream

async def generate_summary(text: str, max_words: int = 30) -> str:
    try:
        # Check cache first
        cached_summary = await cache.get_summary(text, max_words)
        if cached_summary is not None:
            logger.info("Retrieved summary from cache.")
            return cached_summary

        logger.info("Generating summary for text.", text_length=len(text))
        response = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": f"Summarize the following text in approximately {max_words} words. Identify the main topic or provide a concise title as part of the summary."},
                {"role": "user", "content": text}
            ],
            max_tokens=max_words + 20,
            temperature=0.5,
        )
        summary = response.choices[0].message.content.strip()
        
        # Cache the summary
        await cache.set_summary(text, max_words, summary)
        logger.info("Summary generated and cached successfully.", summary_length=len(summary))
        
        return summary
    except Exception as e:
        logger.exception("Error during summarization.")
        raise

async def find_most_similar(query: str, texts: List[str]) -> tuple[str, float]:
    try:
        logger.info("Finding most similar text.", query=query, num_texts=len(texts))

        if not texts:
            logger.warning("find_most_similar called with empty texts list.")
            return "", 0.0
        
        query_vector = await get_embeddings(query)
        texts_vectors = await get_embeddings(texts)

        if not texts_vectors:  # Check if embedding failed for texts
            logger.error("Failed to get embeddings for the provided texts.")
            return "", 0.0  # Or raise an error

        max_similarity = -1.0
        closest_text = ""
        closest_text_index = -1

        for idx, text_vector in enumerate(texts_vectors):
            similarity_score = cosine_similarity(query_vector, text_vector)
            if similarity_score > max_similarity:
                max_similarity = similarity_score
                closest_text_index = idx

        if closest_text_index != -1:
            closest_text = texts[closest_text_index]
        elif texts:  # Fallback if no similarity > -1 found but texts exist
            closest_text = texts[0]
            logger.warning("Could not determine closest text based on similarity > -1, defaulting to first text.")
        else:
            # This case should ideally not be reached due to the initial check
            logger.error("No texts available to determine the closest one.")
            return "", 0.0

        logger.info("Similarity calculation successful.", closest_text=closest_text, score=max_similarity)
        return closest_text, max_similarity
    except Exception as e:
        logger.exception("Error during similarity calculation.")
        raise

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculates cosine similarity between two vectors using NumPy."""
    # Convert lists to NumPy arrays
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)

    # Calculate norms
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    # Check for zero vectors
    if norm1 == 0 or norm2 == 0:
        logger.warning("Attempted cosine similarity with zero vector.")
        return 0.0

    # Calculate dot product
    dot_product = np.dot(v1, v2)

    # Calculate similarity
    similarity = dot_product / (norm1 * norm2)

    # Clamp value due to potential floating point inaccuracies
    return float(np.clip(similarity, -1.0, 1.0))