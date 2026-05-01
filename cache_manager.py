"""
Cache Manager
Implements disk and memory caching for bill processing results
"""

import logging
import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional, Any
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of OCR results"""
    
    def __init__(self, config=None):
        self.config = config
        self.cache_dir = Path(getattr(config, 'CACHE_DIR', 'data/cache')) if config else Path('data/cache')
        self.ttl = getattr(config, 'CACHE_TTL', 86400) if config else 86400  # 24 hours
        self.cache_type = getattr(config, 'CACHE_TYPE', 'disk') if config else 'disk'
        self.memory_cache = {}  # In-memory cache
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def get_file_hash(self, file_path: str) -> str:
        """
        Generate unique hash for file based on content
        
        Why:
        - Identifies file uniquely
        - Detects if file content changed
        - Enables cache invalidation
        
        Process:
        1. Read file content
        2. Calculate SHA256 hash
        3. Return hex digest
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"Hash calculation failed: {e}")
            return ""
    
    def get_from_cache(self, file_path: str) -> Optional[Any]:
        """Get result from cache if available and not expired"""
        try:
            file_hash = self.get_file_hash(file_path)
            
            if not file_hash:
                return None
            
            # Check memory cache first (fastest)
            if file_hash in self.memory_cache:
                cached_data = self.memory_cache[file_hash]
                if not self._is_expired(cached_data['timestamp']):
                    self.logger.info(f"Cache hit (memory): {file_path}")
                    return cached_data['result']
                else:
                    del self.memory_cache[file_hash]
            
            # Check disk cache if enabled
            if self.cache_type == 'disk':
                cache_file = self.cache_dir / f"{file_hash}.pkl"
                if cache_file.exists():
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    if not self._is_expired(cached_data['timestamp']):
                        self.logger.info(f"Cache hit (disk): {file_path}")
                        # Also add to memory cache for next access
                        self.memory_cache[file_hash] = cached_data
                        return cached_data['result']
                    else:
                        cache_file.unlink()
            
            self.logger.info(f"Cache miss: {file_path}")
            return None
            
        except Exception as e:
            self.logger.error(f"Cache retrieval failed: {e}")
            return None
    
    def save_to_cache(self, file_path: str, result: Any) -> bool:
        """Save result to cache"""
        try:
            file_hash = self.get_file_hash(file_path)
            
            if not file_hash:
                return False
            
            cached_data = {
                'timestamp': datetime.now(),
                'result': result,
                'file_path': file_path
            }
            
            # Save to memory cache
            self.memory_cache[file_hash] = cached_data
            
            # Save to disk if enabled
            if self.cache_type == 'disk':
                cache_file = self.cache_dir / f"{file_hash}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(cached_data, f)
                
                self.logger.info(f"Saved to cache: {file_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Cache save failed: {e}")
            return False
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """Check if cache entry is expired"""
        age = (datetime.now() - timestamp).total_seconds()
        return age > self.ttl
    
    def clear_cache(self) -> bool:
        """Clear all cache entries"""
        try:
            # Clear memory cache
            self.memory_cache.clear()
            
            # Clear disk cache
            if self.cache_type == 'disk':
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
            
            self.logger.info("Cache cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Cache clear failed: {e}")
            return False
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        try:
            memory_size = len(self.memory_cache)
            disk_files = len(list(self.cache_dir.glob("*.pkl"))) if self.cache_type == 'disk' else 0
            
            return {
                'memory_entries': memory_size,
                'disk_entries': disk_files,
                'total_entries': memory_size + disk_files,
                'cache_dir': str(self.cache_dir),
                'cache_type': self.cache_type,
                'ttl_seconds': self.ttl
            }
            
        except Exception as e:
            self.logger.error(f"Cache stats retrieval failed: {e}")
            return {}
