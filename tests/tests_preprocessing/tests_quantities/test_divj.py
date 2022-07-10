from ... import not_implemented_warning as NIW
from exact_laws.preprocessing.quantities.divj import *
import os
import numpy as np
import h5py as h5
import pytest 
import warnings

class TestDivJ:
    def test_compressible(self):
        divj = DivJ()
        assert divj.name == 'divj', 'error in name'
        assert divj.incompressible is False, 'error in incompressible property'
    
    def test_incompressible(self):
        divj = DivJ(incompressible=True)
        assert divj.name == 'Idivj', 'error in name'
        assert divj.incompressible is True, 'error in incompressible property'

@pytest.fixture
def init_arg_fb():
    filename = "test_file.h5"
    npoint = 50
    x = np.arange(0, npoint) / npoint * 2 * np.pi
    sinx = np.sin(x)
    cosx = np.cos(x)
    dic_quant = {}
    dic_quant['rho'] = 4 * np.ones([npoint, npoint, npoint])
    dic_quant['bx'] = np.array(
        [sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['by'] = np.array(
        [2 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['bz'] = np.array(
        [3 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect = {}
    dic_expect['divj'] = np.array([1 / 4 * ((3 * cosx[i] * cosx[j] * sinx[k] - 2 * cosx[i] * sinx[j] * cosx[k]) + (
            sinx[i] * cosx[j] * cosx[k] - 3 * cosx[i] * cosx[j] * sinx[k]) + (
                                                    2 * cosx[i] * sinx[j] * cosx[k] - sinx[i] * cosx[j] * cosx[k]))
                                   for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['Idivj'] = np.array([1 * ((3 * cosx[i] * cosx[j] * sinx[k] - 2 * cosx[i] * sinx[j] * cosx[k]) + (
            sinx[i] * cosx[j] * cosx[k] - 3 * cosx[i] * cosx[j] * sinx[k]) + (
                                                    2 * cosx[i] * sinx[j] * cosx[k] - sinx[i] * cosx[j] * cosx[k]))
                                   for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_param = {'N': [npoint, npoint, npoint], 'c': [x[1], x[1], x[1]], 'precision': x[1] * x[1] * x[1] * x[1]}
    yield filename, dic_quant, dic_param, dic_expect
    os.remove(filename)
    
class TestCreateDataset:
    
    def test_compressible(self, init_arg_fb):
        with h5.File(init_arg_fb[0], 'w') as f:
            divj = DivJ()
            divj.create_datasets(f, init_arg_fb[1], init_arg_fb[2])
        with h5.File(init_arg_fb[0], 'r') as f:
            for quant in ['divj']:
                assert np.max(np.abs(init_arg_fb[3][quant] - np.array(f[quant]))) < init_arg_fb[2][
                    'precision'], f"error on {quant} recording" 
    
    def test_incompressible(self, init_arg_fb):
        with h5.File(init_arg_fb[0], 'w') as f:
            divj = DivJ(incompressible=True)
            divj.create_datasets(f, init_arg_fb[1], init_arg_fb[2])
        with h5.File(init_arg_fb[0], 'r') as f:
            for quant in ['Idivj']:
                assert np.max(np.abs(init_arg_fb[3][quant] - np.array(f[quant]))) < init_arg_fb[2][
                    'precision'], f"error on {quant} recording"  

class TestLoad:
    def test_compressible(self):
        divj_name, divj = load()
        assert divj_name == divj.name, 'error in name'
        assert divj.incompressible is False, 'error in incompressible property'
        
    def test_incompressible(self):
        divj_name, divj = load(incompressible=True)
        assert divj_name == divj.name, 'error in name'
        assert divj.incompressible is True, 'error in incompressible property'


