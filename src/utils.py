"""
LinguaBridge Local - Utility Functions
Shared utilities for logging, configuration, and file operations.

CONTEXT FOR AI ASSISTANTS:
==========================
This module provides shared infrastructure used across all project phases:

1. Configuration Management:
   - load_config(): Loads central config.yaml with all hyperparameters
   - Config structure: data, teacher, student, inference, deployment, hardware, paths
   
2. Logging Setup:
   - setup_logging(): Creates both file and console handlers
   - Logs go to logs/linguabridge.log
   - Used by all training and inference modules
   
3. File I/O:
   - save_pickle/load_pickle(): For vocabularies (word2idx, idx2word)
   - Vocabularies are critical - they map tokens to indices for model input
   
4. Model Utilities:
   - count_parameters(): Reports trainable params (important for comparing teacher/student)
   - format_number(): Human-readable numbers (7B, 500M)
   
5. Directory Management:
   - ensure_directories(): Creates data/, models/, logs/, cache/ if missing
   - Called at startup of each phase

IMPORTANT DEPENDENCIES:
- All other modules import from here
- If config.yaml structure changes, update load_config() usage
- Pickle files must use same Python version for compatibility

USAGE PATTERN:
from src.utils import load_config, setup_logging, ensure_directories
config = load_config()
logger = setup_logging(config)
ensure_directories(config)
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pickle


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured logger instance
    """
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logs directory if it doesn't exist
    log_file = log_config.get('file', 'logs/linguabridge.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler() if log_config.get('console_output', True) else logging.NullHandler()
        ]
    )
    
    logger = logging.getLogger('LinguaBridge')
    return logger


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Ensure all required directories exist.
    
    Args:
        config: Configuration dictionary
    """
    paths = config.get('paths', {})
    for key, path in paths.items():
        os.makedirs(path, exist_ok=True)
    
    # Additional specific directories
    data_config = config.get('data', {})
    for key in ['raw_data_dir', 'processed_data_dir']:
        if key in data_config:
            os.makedirs(data_config[key], exist_ok=True)
    
    teacher_config = config.get('teacher', {})
    if 'output_dir' in teacher_config:
        os.makedirs(teacher_config['output_dir'], exist_ok=True)
    
    student_config = config.get('student', {})
    if 'output_dir' in student_config:
        os.makedirs(student_config['output_dir'], exist_ok=True)


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def count_parameters(model) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PaddlePaddle or PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    try:
        # PaddlePaddle
        return sum(p.numel() for p in model.parameters() if not p.stop_gradient)
    except AttributeError:
        # PyTorch
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: int) -> str:
    """
    Format large numbers with suffixes (K, M, B).
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string
    """
    if num < 1000:
        return str(num)
    elif num < 1_000_000:
        return f"{num/1000:.1f}K"
    elif num < 1_000_000_000:
        return f"{num/1_000_000:.1f}M"
    else:
        return f"{num/1_000_000_000:.1f}B"


def get_device_info(config: Dict[str, Any]) -> str:
    """
    Get device information based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Device string ('cpu' or 'gpu:0')
    """
    hardware_config = config.get('hardware', {})
    device = hardware_config.get('device', 'cpu')
    
    if device == 'cpu':
        return 'cpu'
    else:
        # For PaddlePaddle, return gpu:0 format
        return 'gpu:0'


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        
    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        self.current += n
        if self.current % max(1, self.total // 20) == 0:  # Update every 5%
            percent = (self.current / self.total) * 100
            print(f"\r{self.description}: {self.current}/{self.total} ({percent:.1f}%)", end='')
    
    def finish(self) -> None:
        """Mark as complete."""
        print(f"\r{self.description}: {self.total}/{self.total} (100.0%)")
        print()


def validate_file_exists(filepath: str, description: str = "File") -> None:
    """
    Validate that a file exists, raise error if not.
    
    Args:
        filepath: Path to file
        description: Description for error message
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{description} not found: {filepath}")


def estimate_memory_usage(num_samples: int, avg_length: int, dtype_size: int = 4) -> str:
    """
    Estimate memory usage for dataset.
    
    Args:
        num_samples: Number of samples
        avg_length: Average sequence length
        dtype_size: Size of data type in bytes (4 for float32)
        
    Returns:
        Formatted memory estimate
    """
    bytes_estimate = num_samples * avg_length * dtype_size
    mb_estimate = bytes_estimate / (1024 * 1024)
    
    if mb_estimate < 1024:
        return f"{mb_estimate:.1f} MB"
    else:
        return f"{mb_estimate/1024:.1f} GB"
