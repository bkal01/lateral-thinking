#!/usr/bin/env python3
"""
Utility functions for evaluation tasks.
"""

import logging
import sys


def setup_logging(log_filename: str = 'evaluation.log'):
    """Configure logging for evaluation scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_filename)
        ]
    )