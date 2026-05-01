"""
Utility functions for file handling and common operations
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")


def save_json(data: Dict, filepath: str) -> None:
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    ensure_dir(os.path.dirname(filepath))
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: str) -> Dict:
    """
    Load JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary from JSON
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded JSON from {filepath}")
    return data


def list_files(directory: str, extension: str = None) -> List[str]:
    """
    List all files in directory
    
    Args:
        directory: Directory to search
        extension: Filter by extension (e.g., '.png')
        
    Returns:
        List of file paths
    """
    files = []
    
    for file in os.listdir(directory):
        if extension:
            if file.endswith(extension):
                files.append(os.path.join(directory, file))
        else:
            files.append(os.path.join(directory, file))
    
    logger.info(f"Found {len(files)} files in {directory}")
    return sorted(files)


def parse_amount(text: str) -> float:
    """
    Extract numeric amount from text
    
    Examples:
        "₹1,000.00" → 1000.0
        "Rs. 500" → 500.0
        "250" → 250.0
    
    Args:
        text: Text containing amount
        
    Returns:
        Float value or None if not found
    """
    import re
    
    # Remove currency symbols and commas
    cleaned = re.sub(r'[₹$Rs\.,\s]', '', text)
    
    # Extract number
    match = re.search(r'\d+\.?\d*', cleaned)
    
    if match:
        return float(match.group())
    
    return None


def calculate_accuracy(predicted: List[Dict], ground_truth: List[Dict]) -> Dict:
    """
    Calculate extraction accuracy metrics
    
    Args:
        predicted: List of predicted items
        ground_truth: List of actual items
        
    Returns:
        Dictionary with accuracy metrics
    """
    # Simple item count comparison
    predicted_count = len(predicted)
    actual_count = len(ground_truth)
    
    # Amount comparison
    predicted_total = sum(item.get('item_amount', 0) for item in predicted)
    actual_total = sum(item.get('item_amount', 0) for item in ground_truth)
    
    amount_error = abs(predicted_total - actual_total) / actual_total if actual_total > 0 else 0
    
    return {
        'predicted_item_count': predicted_count,
        'actual_item_count': actual_count,
        'item_count_accuracy': min(predicted_count, actual_count) / max(predicted_count, actual_count) if max(predicted_count, actual_count) > 0 else 0,
        'predicted_total': predicted_total,
        'actual_total': actual_total,
        'amount_error_percentage': amount_error * 100
    }
