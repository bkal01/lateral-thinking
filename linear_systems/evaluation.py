#!/usr/bin/env python3
"""
Evaluation logic and answer parsing for linear systems evaluation.
"""

import csv
import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from models import generate_response
from prompts import generate_prompts


def parse_latex_answer(response: str) -> Optional[Tuple[float, float]]:
    """
    Parse LaTeX formatted answer from LLM response.
    
    Args:
        response: LLM response text
    
    Returns:
        Tuple of (x, y) values if parsing successful, None otherwise
    """
    # Pattern to match \boxed{\begin{pmatrix} x \\ y \end{pmatrix}}
    # Allow for various whitespace and formatting variations
    patterns = [
        r'\\boxed\{\\begin\{pmatrix\}\s*(-?\d+(?:\.\d+)?)\s*\\\\\s*(-?\d+(?:\.\d+)?)\s*\\end\{pmatrix\}\}',
        r'\\boxed\{\\begin\{pmatrix\}\s*(-?\d+(?:\.\d+)?)\s*\\\s*(-?\d+(?:\.\d+)?)\s*\\end\{pmatrix\}\}',
        r'boxed\{\\begin\{pmatrix\}\s*(-?\d+(?:\.\d+)?)\s*\\\\\s*(-?\d+(?:\.\d+)?)\s*\\end\{pmatrix\}\}',
        r'\\begin\{pmatrix\}\s*(-?\d+(?:\.\d+)?)\s*\\\\\s*(-?\d+(?:\.\d+)?)\s*\\end\{pmatrix\}',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
        if matches:
            try:
                x_val, y_val = matches[0]
                return (float(x_val), float(y_val))
            except (ValueError, IndexError):
                continue
    
    # Fallback: look for simple coordinate pairs in parentheses
    fallback_pattern = r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)'
    matches = re.findall(fallback_pattern, response)
    if matches:
        try:
            x_val, y_val = matches[-1]  # Take the last match
            return (float(x_val), float(y_val))
        except (ValueError, IndexError):
            pass
    
    return None


def evaluate_system(model, tokenizer, A_i: np.ndarray, b_i: np.ndarray, 
                   x_true: np.ndarray, tolerance: float, system_id: int) -> List[Dict]:
    """
    Evaluate a single system of linear equations across all methods.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        A_i: 2x2 coefficient matrix
        b_i: 2-element RHS vector
        x_true: True solution vector
        tolerance: Numerical tolerance for solution matching
        system_id: System identifier
    
    Returns:
        List of result dictionaries for each method
    """
    logger = logging.getLogger(__name__)
    methods = ['substitution', 'elimination', 'gaussian']
    results = []
    
    logger.info(f"True solution: x={x_true[0]:.1f}, y={x_true[1]:.1f}")
    
    # Generate prompts for all three methods
    prompts = generate_prompts(A_i, b_i)
    
    # Test each method
    for method in methods:
        logger.info(f"\n{method.upper()} METHOD:")
        logger.info(f"Prompt: {prompts[method]}")
        
        # Generate response with thinking mode
        response_dict = generate_response(model, tokenizer, prompts[method])
        
        # Log thinking content and final content separately
        logger.info(f"Thinking content: {response_dict['thinking_content']}")
        logger.info(f"Final content: {response_dict['content']}")
        logger.info(f"Full response: {response_dict['full_response']}")
        
        # Parse answer from the final content
        parsed_answer = parse_latex_answer(response_dict['content'])
        if parsed_answer is None:
            # Fallback: try parsing from full response
            parsed_answer = parse_latex_answer(response_dict['full_response'])
        
        # Track results
        is_correct = False
        parse_success = parsed_answer is not None
        
        if parse_success:
            x_pred, y_pred = parsed_answer
            logger.info(f"Parsed answer: x={x_pred}, y={y_pred}")
            
            # Check accuracy
            x_correct = abs(x_pred - x_true[0]) < tolerance
            y_correct = abs(y_pred - x_true[1]) < tolerance
            is_correct = x_correct and y_correct
            
            logger.info(f"Correct: {is_correct}")
        else:
            x_pred, y_pred = None, None
            logger.info("Failed to parse answer")
        
        # Store detailed results
        results.append({
            'system_id': system_id,
            'method': method,
            'ground_truth_x': x_true[0],
            'ground_truth_y': x_true[1],
            'predicted_x': x_pred,
            'predicted_y': y_pred,
            'correct': is_correct,
            'parse_success': parse_success,
            'thinking_content': response_dict['thinking_content'],
            'final_content': response_dict['content'],
            'full_response': response_dict['full_response']
        })
        
        logger.info("-" * 50)
    
    return results


def save_results_to_csv(detailed_results: List[Dict], output_path: str):
    """
    Save detailed results to CSV file.
    
    Args:
        detailed_results: List of result dictionaries
        output_path: Path to output CSV file
    """
    logger = logging.getLogger(__name__)
    
    if detailed_results:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['system_id', 'method', 'ground_truth_x', 'ground_truth_y', 
                         'predicted_x', 'predicted_y', 'correct', 'parse_success',
                         'thinking_content', 'final_content', 'full_response']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_results)
        
        logger.info(f"Detailed results saved to: {output_path}")


def print_summary_statistics(detailed_results: List[Dict]):
    """
    Print summary statistics for the evaluation.
    
    Args:
        detailed_results: List of result dictionaries
    """
    logger = logging.getLogger(__name__)
    methods = ['substitution', 'elimination', 'gaussian']
    
    # Initialize accuracy tracking
    results = {method: {'correct': 0, 'total': 0, 'parse_success': 0} for method in methods}
    
    # Aggregate results
    for result in detailed_results:
        method = result['method']
        results[method]['total'] += 1
        if result['correct']:
            results[method]['correct'] += 1
        if result['parse_success']:
            results[method]['parse_success'] += 1
    
    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    
    for method in methods:
        total = results[method]['total']
        correct = results[method]['correct']
        parse_success = results[method]['parse_success']
        
        accuracy = (correct / total * 100) if total > 0 else 0
        parse_rate = (parse_success / total * 100) if total > 0 else 0
        
        logger.info(f"\n{method.upper()} METHOD:")
        logger.info(f"  Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        logger.info(f"  Parse Success: {parse_success}/{total} ({parse_rate:.1f}%)")
    
    # Overall statistics
    total_systems = sum(results[method]['total'] for method in methods) // len(methods)
    total_correct = sum(results[method]['correct'] for method in methods)
    total_attempts = sum(results[method]['total'] for method in methods)
    total_parsed = sum(results[method]['parse_success'] for method in methods)
    
    overall_accuracy = (total_correct / total_attempts * 100) if total_attempts > 0 else 0
    overall_parse_rate = (total_parsed / total_attempts * 100) if total_attempts > 0 else 0
    
    logger.info(f"\nOVERALL STATISTICS:")
    logger.info(f"  Systems evaluated: {total_systems}")
    logger.info(f"  Total attempts: {total_attempts}")
    logger.info(f"  Overall accuracy: {total_correct}/{total_attempts} ({overall_accuracy:.1f}%)")
    logger.info(f"  Overall parse rate: {total_parsed}/{total_attempts} ({overall_parse_rate:.1f}%)")
    
    # Find best and worst performing methods
    method_accuracies = [(method, results[method]['correct']/results[method]['total']*100 if results[method]['total'] > 0 else 0) for method in methods]
    best_method = max(method_accuracies, key=lambda x: x[1])
    worst_method = min(method_accuracies, key=lambda x: x[1])
    
    logger.info(f"\nMETHOD COMPARISON:")
    logger.info(f"  Best performing: {best_method[0]} ({best_method[1]:.1f}%)")
    logger.info(f"  Worst performing: {worst_method[0]} ({worst_method[1]:.1f}%)")