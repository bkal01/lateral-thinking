import argparse
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate solvable systems of linear equations and save them to disk."
    )
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--num_vars", type=int, required=True)
    parser.add_argument("--num_equations", type=int, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument(
        "--low", type=int, default=-10, help="Lower bound (inclusive) for random ints"
    )
    parser.add_argument(
        "--high", type=int, default=10, help="Upper bound (inclusive) for random ints"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)

    num_samples, num_vars, num_equations = args.num_samples, args.num_vars, args.num_equations

    As = np.random.randint(args.low, args.high + 1, size=(num_samples, num_equations, num_vars))
    xs = np.random.randint(args.low, args.high + 1, size=(num_samples, num_vars))
    bs = np.einsum("sij,sj->si", As, xs)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    fname = (
        f"linear_systems_{num_samples}_samples_{num_equations}eq_{num_vars}vars.npz"
    )
    path = os.path.join(args.save_dir, fname)
    np.savez_compressed(path, A=As, b=bs, x=xs)
    print(f"Saved {num_samples} systems to {path}")

    # Verify shapes after load
    with np.load(path) as data:
        A_loaded, b_loaded, x_loaded = data["A"], data["b"], data["x"]

    assert A_loaded.shape == (num_samples, num_equations, num_vars), "Loaded A shape mismatch"
    assert b_loaded.shape == (num_samples, num_equations), "Loaded b shape mismatch"
    assert x_loaded.shape == (num_samples, num_vars), "Loaded x shape mismatch"
    print("Shape verification passed.")


if __name__ == "__main__":
    main()
