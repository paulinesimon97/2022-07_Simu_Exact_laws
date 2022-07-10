from ast import Not
import pytest
from exact_laws.preprocessing.process_on_standard_h5_file import *
from .. import not_implemented_warning as NIW

class TestVerifFileExistence:
    
    def test(self):
        NIW(verif_file_existence)

class TestCheckFile:
    
    def test(self):
        NIW(check_file)

class TestDataBinning:
    
    def test(self):
        NIW(data_binning)

class TestBinAnArray:
        
    def test_bin2_ones(self):
        tab = np.ones((10, 10, 10))
        bin = 2
        result = bin_an_array(tab, bin)
        expected_result = np.ones((5, 5, 5))
        assert np.array_equal(result, expected_result), f"error on the binning of an array"

    def test_bin2_gradx(self):
        x = np.arange(0, 10)
        tab = np.ones((10, 10, 10))
        for i in range(10):
            tab[i] = tab[i] * x[i]
        bin = 2
        result = bin_an_array(tab, bin)
        expected_result = np.ones((5, 5, 5))
        expected_x = np.array([0.5, 2.5, 4.5, 6.5, 8.5])
        for i in range(5):
            expected_result[i] = expected_result[i] * expected_x[i]
        assert np.array_equal(result, expected_result), f"error on the binning of an array"

class TestBinArraysInH5:
    
    def test(self):
        NIW(bin_arrays_in_h5)