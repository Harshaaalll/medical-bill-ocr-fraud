"""
Application Configuration
Manages all settings for the application
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings
import yaml
import json

class Settings(BaseSettings):
    """Main application settings"""
    
    # API Configuration
    API_TITLE: str = "Medical Bill OCR Pro API"
    API_VERSION: str = "2.0.0"
    API_DESCRIPTION: str = "Professional OCR system for medical bills with fraud detection"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    
    # OCR Configuration
    OCR_USE_GPU: bool = True
    OCR_LANG: str = "en"
    OCR_USE_ANGLE_DETECTION: bool = True
    OCR_CONFIDENCE_THRESHOLD: float = 0.3
    
    # LLM Configuration
    LLM_MODEL: str = "qwen:7b"  # Qwen 3 model
    LLM_ENABLED: bool = True
    LLM_API_KEY: Optional[str] = None
    LLM_ENDPOINT: str = "http://localhost:11434"
    LLM_TIMEOUT: int = 30
    
    # Fraud Detection
    FRAUD_DETECTION_ENABLED: bool = True
    FRAUD_AMOUNT_THRESHOLD: float = 100000.0
    FRAUD_ANOMALY_PERCENTILE: float = 95.0
    FRAUD_DUPLICATE_DAYS: int = 30
    
    # Caching
    CACHE_ENABLED: bool = True
    CACHE_TYPE: str = "disk"  # disk, redis, or memory
    CACHE_DIR: str = "data/cache"
    CACHE_TTL: int = 86400  # 24 hours
    REDIS_URL: Optional[str] = None
    
    # File Processing
    MAX_FILE_SIZE_MB: int = 50
    SUPPORTED_FORMATS: list = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
    BATCH_SIZE: int = 10
    
    # Preprocessing
    ENABLE_DESKEW: bool = True
    ENABLE_BILATERAL_FILTER: bool = True
    ENABLE_SCALING: bool = True
    TARGET_DPI: int = 300
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_DIR: str = "logs"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_config() -> Settings:
    """Load configuration from environment and config files"""
    return Settings()


def load_llm_config(config_path: str = "config/llm_config.json") -> dict:
    """Load LLM-specific configuration"""
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {
        "model": "qwen:7b",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 2000
    }


def load_app_config(config_path: str = "config/app_config.yaml") -> dict:
    """Load application configuration from YAML"""
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


# Global configuration instance
CONFIG = load_config()
LLM_CONFIG = load_llm_config()
APP_CONFIG = load_app_config()
