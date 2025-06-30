#!/usr/bin/env python3
"""
Utility functions for linear systems evaluation.
"""

import logging
import sys


def setup_logging():
    """Configure logging for the evaluation script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('linear_systems_evaluation.log')
        ]
    )