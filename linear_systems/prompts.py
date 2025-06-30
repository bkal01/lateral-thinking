#!/usr/bin/env python3
"""
Prompt generation and equation formatting for linear systems evaluation.
"""

import numpy as np
from typing import Dict


def format_system_equations(A: np.ndarray, b: np.ndarray) -> str:
    """
    Convert coefficient matrix and RHS vector to equation string format.
    
    Args:
        A: 2x2 coefficient matrix
        b: 2-element RHS vector
    
    Returns:
        String representation like "-4x + 9y = -62, 4x = -28"
    """
    def format_term(coef: float, var: str, is_first: bool = False) -> str:
        if coef == 0:
            return ""
        elif coef == 1:
            return f"{var}" if is_first else f" + {var}"
        elif coef == -1:
            return f"-{var}" if is_first else f" - {var}"
        elif coef > 0:
            return f"{int(coef)}{var}" if is_first else f" + {int(coef)}{var}"
        else:
            return f"{int(coef)}{var}" if is_first else f" - {int(abs(coef))}{var}"
    
    # First equation: a11*x + a12*y = b1
    eq1_terms = []
    if A[0, 0] != 0:
        eq1_terms.append(format_term(A[0, 0], "x", True))
    if A[0, 1] != 0:
        eq1_terms.append(format_term(A[0, 1], "y", len(eq1_terms) == 0))
    eq1 = "".join(eq1_terms) + f" = {int(b[0])}"
    
    # Second equation: a21*x + a22*y = b2
    eq2_terms = []
    if A[1, 0] != 0:
        eq2_terms.append(format_term(A[1, 0], "x", True))
    if A[1, 1] != 0:
        eq2_terms.append(format_term(A[1, 1], "y", len(eq2_terms) == 0))
    eq2 = "".join(eq2_terms) + f" = {int(b[1])}"
    
    return f"{eq1}, {eq2}"


def generate_prompts(A: np.ndarray, b: np.ndarray) -> Dict[str, str]:
    """
    Generate prompts for all three solution methods.
    
    Args:
        A: 2x2 coefficient matrix
        b: 2-element RHS vector
    
    Returns:
        Dictionary with keys 'substitution', 'elimination', 'gaussian'
    """
    equations = format_system_equations(A, b)
    
    base_instruction = "Solve the following system of linear equations using {method}: {equations}. Return your final answer in the exact format: \\boxed{{\\begin{{pmatrix}} x_value \\\\ y_value \\end{{pmatrix}}}}"
    
    prompts = {
        'substitution': base_instruction.format(
            method="the substitution method",
            equations=equations
        ),
        'elimination': base_instruction.format(
            method="the elimination method", 
            equations=equations
        ),
        'gaussian': base_instruction.format(
            method="Gaussian elimination",
            equations=equations
        )
    }
    
    return prompts