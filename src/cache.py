"""
Data caching module for stock price prediction
Caches fetched data and computed features to avoid redundant computation
"""

import os
import pickle
import hashlib
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataCache:
    """
    Cache for stock data and features
    """

    def __init__(self, cache_dir: str = "cache", ttl_days: int = 1):
        """
        Initialize cache

        Args:
            cache_dir: Directory for cache files
            ttl_days: Time to live in days (default: 1 day)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_days = ttl_days

    def _generate_key(self, symbol: str, start_date: str, end_date: str, 
                     data_type: str = "raw") -> str:
        """
        Generate cache key from parameters

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            data_type: Type of data ('raw', 'features', etc.)

        Returns:
            Cache key (filename)
        """
        key_string = f"{symbol}_{start_date}_{end_date}_{data_type}"
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{key_hash}.pkl"

    def _is_expired(self, filepath: Path) -> bool:
        """
        Check if cache file is expired

        Args:
            filepath: Path to cache file

        Returns:
            True if expired, False otherwise
        """
        if not filepath.exists():
            return True

        file_time = datetime.fromtimestamp(filepath.stat().st_mtime)
        age = datetime.now() - file_time

        return age.days > self.ttl_days

    def get(self, symbol: str, start_date: str, end_date: str, 
           data_type: str = "raw") -> Optional[pd.DataFrame]:
        """
        Get data from cache

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            data_type: Type of data

        Returns:
            Cached DataFrame or None if not found/expired
        """
        cache_key = self._generate_key(symbol, start_date, end_date, data_type)
        cache_path = self.cache_dir / cache_key

        if not cache_path.exists():
            logger.debug(f"Cache miss: {cache_key}")
            return None

        if self._is_expired(cache_path):
            logger.debug(f"Cache expired: {cache_key}")
            cache_path.unlink()  # Delete expired cache
            return None

        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Cache hit: {cache_key}")
            return data
        except Exception as e:
            logger.warning(f"Error loading cache {cache_key}: {str(e)}")
            return None

    def set(self, data: pd.DataFrame, symbol: str, start_date: str, 
           end_date: str, data_type: str = "raw") -> None:
        """
        Store data in cache

        Args:
            data: DataFrame to cache
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            data_type: Type of data
        """
        cache_key = self._generate_key(symbol, start_date, end_date, data_type)
        cache_path = self.cache_dir / cache_key

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Cached data: {cache_key}")
        except Exception as e:
            logger.warning(f"Error caching data {cache_key}: {str(e)}")

    def clear(self, data_type: Optional[str] = None) -> None:
        """
        Clear cache

        Args:
            data_type: Type of data to clear (None = clear all)
        """
        if data_type:
            # Clear specific type
            pattern = f"*_{data_type}.pkl"
            for file in self.cache_dir.glob(pattern):
                file.unlink()
                logger.info(f"Cleared cache: {file.name}")
        else:
            # Clear all
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()
                logger.info(f"Cleared cache: {file.name}")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cache

        Returns:
            Dictionary with cache statistics
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        
        total_size = sum(f.stat().st_size for f in cache_files)
        expired_count = sum(1 for f in cache_files if self._is_expired(f))
        
        return {
            'total_files': len(cache_files),
            'expired_files': expired_count,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }


# Global cache instance
_cache_instance: Optional[DataCache] = None


def get_cache(cache_dir: str = "cache", ttl_days: int = 1) -> DataCache:
    """
    Get or create global cache instance

    Args:
        cache_dir: Directory for cache files
        ttl_days: Time to live in days

    Returns:
        DataCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = DataCache(cache_dir, ttl_days)
    return _cache_instance

