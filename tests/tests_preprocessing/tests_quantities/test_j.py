from ... import not_implemented_warning as NIW
from exact_laws.preprocessing.quantities.j import *
import os
import numpy as np
import h5py as h5
import pytest 
import warnings

class TestJ:
    def test_compressible(self):
        j = J()
        assert j.name == 'j', 'error in name'
        assert j.incompressible is False, 'error in incompressible property'
    
    def test_incompressible(self):
        j = J(incompressible=True)
        assert j.name == 'Ij', 'error in name'
        assert j.incompressible is True, 'error in incompressible property'

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
    dic_expect['Ijx'] = np.array(
        [(3 * sinx[i] * cosx[j] * sinx[k] - 2 * sinx[i] * sinx[j] * cosx[k]) for i in range(npoint) for j in
         range(npoint) for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['Ijy'] = np.array(
        [(sinx[i] * sinx[j] * cosx[k] - 3 * cosx[i] * sinx[j] * sinx[k]) for i in range(npoint) for j in range(npoint)
         for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['Ijz'] = np.array(
        [(2 * cosx[i] * sinx[j] * sinx[k] - sinx[i] * cosx[j] * sinx[k]) for i in range(npoint) for j in range(npoint)
         for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['jx'] = np.array(
        [1 / 4 * (3 * sinx[i] * cosx[j] * sinx[k] - 2 * sinx[i] * sinx[j] * cosx[k]) for i in range(npoint) for j in
         range(npoint) for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['jy'] = np.array(
        [1 / 4 * (sinx[i] * sinx[j] * cosx[k] - 3 * cosx[i] * sinx[j] * sinx[k]) for i in range(npoint) for j in
         range(npoint) for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['jz'] = np.array(
        [1 / 4 * (2 * cosx[i] * sinx[j] * sinx[k] - sinx[i] * cosx[j] * sinx[k]) for i in range(npoint) for j in
         range(npoint) for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_param = {'N': [npoint, npoint, npoint], 'c': [x[1], x[1], x[1]], 'precision': x[1] * x[1] * x[1] * x[1]}
    yield filename, dic_quant, dic_param, dic_expect
    os.remove(filename)

class TestGetOriginalQuantity:
    
    def test(self, init_arg_fb):
        with h5.File(init_arg_fb[0], 'w') as f:
            pass
        J.get_original_quantity(init_arg_fb[1], init_arg_fb[2])
        for quant in ['jx', 'jy', 'jz']:
            assert np.max(np.abs(init_arg_fb[3]['I'+quant] - init_arg_fb[1][quant])) < init_arg_fb[2][
                'precision'], f"error on {quant} recording" 
        
class TestCreateDataset:
    
    def test_compressible(self, init_arg_fb):
        with h5.File(init_arg_fb[0], 'w') as f:
            j = J()
            j.create_datasets(f, init_arg_fb[1], init_arg_fb[2])
        with h5.File(init_arg_fb[0], 'r') as f:
            for quant in ['jx', 'jy', 'jz']:
                assert np.max(np.abs(init_arg_fb[3][quant] - np.array(f[quant]))) < init_arg_fb[2][
                    'precision'], f"error on {quant} recording" 
    
    def test_incompressible(self, init_arg_fb):
        with h5.File(init_arg_fb[0], 'w') as f:
            j = J(incompressible=True)
            j.create_datasets(f, init_arg_fb[1], init_arg_fb[2])
        with h5.File(init_arg_fb[0], 'r') as f:
            for quant in ['Ijx', 'Ijy', 'Ijz']:
                assert np.max(np.abs(init_arg_fb[3][quant] - np.array(f[quant]))) < init_arg_fb[2][
                    'precision'], f"error on {quant} recording"   

class TestLoad:
    def test_compressible(self):
        j_name, j = load()
        assert j_name == j.name, 'error in name'
        assert j.incompressible is False, 'error in incompressible property'
        
    def test_incompressible(self):
        j_name, j = load(incompressible=True)
        assert j_name == j.name, 'error in name'
        assert j.incompressible is True, 'error in incompressible property'
