#!/usr/bin/env python3
"""
Linear Systems Evaluation Script

Evaluates LLMs on solving systems of linear equations using three methods:
1. Substitution Method
2. Elimination Method  
3. Gaussian Elimination

Usage:
    uv run python evaluate_linear_systems/main.py --model <model_name> --dataset <path> --output <results.csv>
"""

import logging
import sys
import torch

from dataset import load_dataset
from evaluation import evaluate_system, print_summary_statistics, save_results_to_csv
from models import load_model_and_tokenizer
from utils import setup_logging

import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on linear equation solving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python evaluate_linear_systems.py --model Qwen/Qwen3-0.6B --dataset data/linear_systems.npz
    uv run python evaluate_linear_systems.py --model microsoft/phi-3-mini --dataset data/linear_systems.npz --max_systems 100
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='HuggingFace model identifier (e.g., Qwen/Qwen3-0.6B)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to NPZ dataset file containing linear systems'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='linear_systems_results.csv',
        help='Output CSV file for detailed results (default: linear_systems_results.csv)'
    )
    
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-6,
        help='Numerical tolerance for solution matching (default: 1e-6)'
    )
    
    parser.add_argument(
        '--max_systems',
        type=int,
        default=None,
        help='Limit evaluation to first N systems (for testing)'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation script entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        logger.info("Starting Linear Systems Evaluation")
        logger.info(f"Model: {args.model}")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Output: {args.output}")
        
        # Load dataset
        dataset = load_dataset(args.dataset, args.max_systems)
        n_systems = dataset['A'].shape[0]
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model, device)
        
        logger.info(f"Setup complete. Ready to evaluate {n_systems} systems.")
        
        # Initialize results tracking
        detailed_results = []
        
        # Evaluation loop
        for i in range(n_systems):
            logger.info(f"\n--- System {i+1}/{n_systems} ---")
            
            # Get current system
            A_i = dataset['A'][i]
            b_i = dataset['b'][i]
            x_true = dataset['x'][i]
            
            # Evaluate system across all methods
            system_results = evaluate_system(
                model, tokenizer, A_i, b_i, x_true, args.tolerance, i
            )
            detailed_results.extend(system_results)
        
        # Print summary statistics
        print_summary_statistics(detailed_results)
        
        # Save detailed results to CSV
        save_results_to_csv(detailed_results, args.output)
        
        logger.info("="*60)
        logger.info("Evaluation completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()