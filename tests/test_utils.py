"""Unit test for bayesian_shrink."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import bayesian_shrink

def test_bayesian_shrink_small_vs_large():
    """Small sample high avg should rank below large sample slightly lower avg."""
    # Publisher A: 4 books, avg 9.30
    # Publisher B: 35 books, avg 8.99
    # C=8.21 (global mean), m=9 (median book count)
    score_a = bayesian_shrink(9.30, 4, 8.21, 9)
    score_b = bayesian_shrink(8.99, 35, 8.21, 9)
    assert score_b > score_a, (
        f"Large pub (35bks, 8.99) should outrank small pub (4bks, 9.30): "
        f"{score_b:.4f} <= {score_a:.4f}"
    )

def test_bayesian_shrink_converges_to_C():
    """As n approaches 0, score approaches C."""
    score = bayesian_shrink(9.50, 0.1, 8.21, 9)
    assert abs(score - 8.21) < 0.5

def test_bayesian_shrink_converges_to_avg():
    """As n approaches infinity, score approaches avg."""
    score = bayesian_shrink(9.00, 100000, 8.21, 9)
    assert abs(score - 9.00) < 0.01

def test_bayesian_shrink_array_input():
    """Should work with numpy arrays and scalars."""
    import numpy as np
    scores = bayesian_shrink(np.array([9.3, 8.99]), np.array([4, 35]), 8.21, 9)
    assert scores[1] > scores[0]
