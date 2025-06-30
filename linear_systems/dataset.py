#!/usr/bin/env python3
"""
Dataset loading and handling for linear systems evaluation.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def load_dataset(dataset_path: str, max_systems: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Load linear systems dataset from NPZ file.
    
    Args:
        dataset_path: Path to NPZ file containing coefficient matrices (A), 
                     solution vectors (x), and RHS vectors (b)
        max_systems: Limit to first N systems if specified
    
    Returns:
        Dictionary containing 'A', 'x', 'b' arrays
    """
    logger = logging.getLogger(__name__)
    
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    logger.info(f"Loading dataset from {dataset_path}")
    
    try:
        data = np.load(dataset_path)
        
        # Validate expected keys exist
        required_keys = ['A', 'x', 'b']
        missing_keys = [key for key in required_keys if key not in data.files]
        if missing_keys:
            raise ValueError(f"Dataset missing required keys: {missing_keys}")
        
        A = data['A']  # Coefficient matrices (N, 2, 2)
        x = data['x']  # Solution vectors (N, 2)  
        b = data['b']  # RHS vectors (N, 2)
        
        # Validate shapes
        n_systems = A.shape[0]
        if x.shape != (n_systems, 2) or b.shape != (n_systems, 2):
            raise ValueError(f"Inconsistent dataset shapes: A={A.shape}, x={x.shape}, b={b.shape}")
        
        # Limit systems if requested
        if max_systems is not None and max_systems < n_systems:
            logger.info(f"Limiting evaluation to first {max_systems} systems")
            A = A[:max_systems]
            x = x[:max_systems]
            b = b[:max_systems]
            n_systems = max_systems
        
        logger.info(f"Loaded {n_systems} linear systems")
        
        return {
            'A': A,
            'x': x,
            'b': b
        }
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise