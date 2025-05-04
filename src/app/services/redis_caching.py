import json
import hashlib
from typing import Optional, Any, Union, List
import redis
from ..core.config import settings

class RedisLangCache:
    def __init__(self):
        self.redis = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )
        self.ttl = settings.REDIS_TTL

    def _generate_key(self, prefix: str, data: Union[str, dict]) -> str:
        """Generate a unique cache key based on the input data"""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return f"{prefix}:{hashlib.sha256(data_str.encode()).hexdigest()}"

    async def get_embeddings(self, texts: Union[str, List[str]], model: str) -> Optional[Union[List[float], List[List[float]]]]:
        """Get cached embeddings if they exist"""
        cache_key = self._generate_key("emb", {"texts": texts, "model": model})
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        return None

    async def set_embeddings(self, texts: Union[str, List[str]], model: str, embeddings: Union[List[float], List[List[float]]]):
        """Cache embeddings with TTL"""
        cache_key = self._generate_key("emb", {"texts": texts, "model": model})
        self.redis.setex(
            cache_key,
            self.ttl,
            json.dumps(embeddings)
        )

    async def get_summary(self, text: str, max_words: int) -> Optional[str]:
        """Get cached summary if it exists"""
        cache_key = self._generate_key("sum", {"text": text, "max_words": max_words})
        return self.redis.get(cache_key)

    async def set_summary(self, text: str, max_words: int, summary: str):
        """Cache summary with TTL"""
        cache_key = self._generate_key("sum", {"text": text, "max_words": max_words})
        self.redis.setex(
            cache_key,
            self.ttl,
            summary
        )

# Global cache instance
cache = RedisLangCache()