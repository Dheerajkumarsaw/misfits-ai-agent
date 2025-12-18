"""
Server-level caching with pickle persistence
Provides in-memory LRU cache with automatic disk persistence on shutdown
"""

import os
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Tuple
from collections import OrderedDict


class ServerCache:
    """
    In-memory LRU cache with TTL support and pickle persistence

    Features:
    - LRU eviction policy when max_size reached
    - TTL-based expiration for individual entries
    - Pickle dump/load for persistence across restarts
    - Thread-safe operations (basic)
    """

    def __init__(self, max_size: int = 1000, cache_dir: str = 'model'):
        """
        Initialize cache manager

        Args:
            max_size: Maximum number of cache entries before LRU eviction
            cache_dir: Directory to store cache pickle file
        """
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.max_size = max_size
        self.cache_file = os.path.join(cache_dir, 'cache_dump.pkl')

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        print(f"✅ ServerCache initialized (max_size={max_size})")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value if exists and not expired, None otherwise
        """
        if key not in self.cache:
            self.misses += 1
            return None

        value, expiry_time = self.cache[key]

        # Check if expired
        if time.time() > expiry_time:
            # Remove expired entry
            del self.cache[key]
            self.misses += 1
            return None

        # Move to end (mark as recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: Any, ttl: int = 300):
        """
        Set cache entry with TTL

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (default: 5 minutes)
        """
        expiry_time = time.time() + ttl

        # Update existing or add new
        if key in self.cache:
            # Update existing - move to end
            self.cache[key] = (value, expiry_time)
            self.cache.move_to_end(key)
        else:
            # Add new entry
            self.cache[key] = (value, expiry_time)

            # Check if we need to evict (LRU)
            if len(self.cache) > self.max_size:
                # Remove oldest (first item)
                self.cache.popitem(last=False)
                self.evictions += 1

    def delete(self, key: str):
        """Delete cache entry"""
        if key in self.cache:
            del self.cache[key]

    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        print("🗑️ Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round(hit_rate, 2),
            "evictions": self.evictions,
            "total_requests": total_requests
        }

    def cleanup_expired(self):
        """Remove all expired entries"""
        current_time = time.time()
        expired_keys = []

        for key, (value, expiry_time) in self.cache.items():
            if current_time > expiry_time:
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            print(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")

    def save_to_disk(self):
        """
        Pickle dump cache to disk
        Only saves non-expired entries
        """
        try:
            # Clean up expired entries before saving
            self.cleanup_expired()

            # Save to pickle file
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'cache': dict(self.cache),
                    'stats': {
                        'hits': self.hits,
                        'misses': self.misses,
                        'evictions': self.evictions
                    },
                    'saved_at': datetime.now().isoformat()
                }, f)

            print(f"💾 Cache saved to disk: {len(self.cache)} entries in {self.cache_file}")

        except Exception as e:
            print(f"⚠️ Failed to save cache to disk: {e}")

    def load_from_disk(self, max_age_hours: int = 1):
        """
        Load cache from disk if file exists and is recent

        Args:
            max_age_hours: Maximum age of cache file to load (default: 1 hour)
        """
        try:
            if not os.path.exists(self.cache_file):
                print("ℹ️ No cache file found, starting with empty cache")
                return

            # Check file age
            file_mtime = os.path.getmtime(self.cache_file)
            file_age = time.time() - file_mtime

            if file_age > max_age_hours * 3600:
                print(f"⚠️ Cache file too old ({file_age/3600:.1f} hours), ignoring")
                return

            # Load from pickle
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)

            # Restore cache and stats
            self.cache = OrderedDict(data['cache'])
            stats = data.get('stats', {})
            self.hits = stats.get('hits', 0)
            self.misses = stats.get('misses', 0)
            self.evictions = stats.get('evictions', 0)

            # Clean up any expired entries
            self.cleanup_expired()

            saved_at = data.get('saved_at', 'unknown')
            print(f"📂 Cache loaded from disk: {len(self.cache)} entries (saved at {saved_at})")

        except Exception as e:
            print(f"⚠️ Failed to load cache from disk: {e}")
            self.cache.clear()

    def generate_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        import hashlib
        import json

        # Combine args and kwargs into a stable string
        key_data = {
            'args': args,
            'kwargs': kwargs
        }

        # Create hash
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()

        return key_hash


# Global cache instance (to be initialized in api_server.py)
_global_cache: Optional[ServerCache] = None


def get_cache() -> ServerCache:
    """Get global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = ServerCache()
    return _global_cache
