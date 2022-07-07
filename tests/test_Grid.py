from exact_laws.exact_laws_calc import Grid
import pytest
import numpy as np
import matplotlib.pyplot as plt

class TestInit:
    def test_init(self):
        N = [10,10,10]
        L = [125.6,126.5,12.5]
        c = [12.56,12.65,1.25]
        grid = Grid(N,L,c)
        assert np.array_equal(grid.N,N), f"error on N attribute" 
        assert np.array_equal(grid.L,L), f"error on L attribute"
        assert np.array_equal(grid.c,c), f"error on c attribute"

class TestCheck: 
    """voir contenu du fichier output_process_[...].txt."""
    pass
