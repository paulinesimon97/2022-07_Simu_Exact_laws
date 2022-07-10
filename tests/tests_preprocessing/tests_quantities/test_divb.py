from ... import not_implemented_warning as NIW
from exact_laws.preprocessing.quantities.divb import *
import os
import numpy as np
import h5py as h5
import pytest 
import warnings

class TestDivB:
    def test_compressible(self):
        divb = DivB()
        assert divb.name == 'divb', 'error in name'
        assert divb.incompressible is False, 'error in incompressible property'
    
    def test_incompressible(self):
        divb = DivB(incompressible=True)
        assert divb.name == 'Idivb', 'error in name'
        assert divb.incompressible is True, 'error in incompressible property'

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
    dic_expect['divb'] = np.array(
        [1 / 2 * cosx[i] * sinx[j] * sinx[k] + sinx[i] * cosx[j] * sinx[k] + 3 / 2 * sinx[i] * sinx[j] * cosx[k] for i
         in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['Idivb'] = np.array(
        [1 * cosx[i] * sinx[j] * sinx[k] + 2 * sinx[i] * cosx[j] * sinx[k] + 3 * sinx[i] * sinx[j] * cosx[k] for i
         in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_param = {'N': [npoint, npoint, npoint], 'c': [x[1], x[1], x[1]], 'precision': x[1] * x[1] * x[1] * x[1]}
    yield filename, dic_quant, dic_param, dic_expect
    os.remove(filename)
    
class TestCreateDataset:
    
    def test_compressible(self, init_arg_fb):
        with h5.File(init_arg_fb[0], 'w') as f:
            divb = DivB()
            divb.create_datasets(f, init_arg_fb[1], init_arg_fb[2])
        with h5.File(init_arg_fb[0], 'r') as f:
            for quant in ['divb']:
                assert np.max(np.abs(init_arg_fb[3][quant] - np.array(f[quant]))) < init_arg_fb[2][
                    'precision'], f"error on {quant} recording" 
    
    def test_incompressible(self, init_arg_fb):
        with h5.File(init_arg_fb[0], 'w') as f:
            divb = DivB(incompressible=True)
            divb.create_datasets(f, init_arg_fb[1], init_arg_fb[2])
        with h5.File(init_arg_fb[0], 'r') as f:
            for quant in ['Idivb']:
                assert np.max(np.abs(init_arg_fb[3][quant] - np.array(f[quant]))) < init_arg_fb[2][
                    'precision'], f"error on {quant} recording" 

class TestLoad:
    def test_compressible(self):
        divb_name, divb = load()
        assert divb_name == divb.name, 'error in name'
        assert divb.incompressible is False, 'error in incompressible property'
        
    def test_incompressible(self):
        divb_name, divb = load(incompressible=True)
        assert divb_name == divb.name, 'error in name'
        assert divb.incompressible is True, 'error in incompressible property'
