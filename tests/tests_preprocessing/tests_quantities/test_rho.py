from ... import not_implemented_warning as NIW
from exact_laws.preprocessing.quantities.rho import *
import os
import numpy as np
import h5py as h5
import pytest 
import warnings

class TestRho:
    
    def test_compressible(self):
        rho = Rho()
        assert rho.name == 'rho', 'error in name'
        assert rho.incompressible is False, 'error in incompressible property'
    
    def test_incompressible(self):
        rho = Rho(incompressible=True)
        assert rho.name == 'Irho', 'error in name'
        assert rho.incompressible is True, 'error in incompressible property'
    

@pytest.fixture
def init_arg_frho():
    filename = "test_file.h5"
    npoint = 50
    x = np.arange(0, npoint) / npoint * 2 * np.pi
    sinx = np.sin(x)
    cosx = np.cos(x)
    dic_quant = {}
    dic_quant['rho'] = np.array(
        [sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect = {}
    dic_expect['rho'] = np.array(
        [sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['Irho'] = np.ones(
        (npoint, npoint, npoint))
    dic_param = {'N': [npoint, npoint, npoint], 'c': [x[1], x[1], x[1]], 'precision': x[1] * x[1] * x[1] * x[1]}
    yield filename, dic_quant, dic_param, dic_expect
    os.remove(filename)
    
class TestCreateDataset:
    
    def test_compressible(self, init_arg_frho):
        with h5.File(init_arg_frho[0], 'w') as f:
            rho = Rho()
            rho.create_datasets(f, init_arg_frho[1], init_arg_frho[2])
        with h5.File(init_arg_frho[0], 'r') as f:
            for quant in ['rho']:
                assert np.max(np.abs(init_arg_frho[3][quant] - np.array(f[quant]))) < init_arg_frho[2][
                    'precision'], f"error on {quant} recording"
    
    def test_incompressible(self,init_arg_frho):
        with h5.File(init_arg_frho[0], 'w') as f:
            rho = Rho(incompressible=True)
            rho.create_datasets(f, init_arg_frho[1], init_arg_frho[2])
        with h5.File(init_arg_frho[0], 'r') as f:
            for quant in ['Irho']:
                assert np.max(np.abs(init_arg_frho[3][quant] - np.array(f[quant]))) < init_arg_frho[2][
                    'precision'], f"error on {quant} recording"
    
    
class TestLoad:
    
    def test_compressible(self):
        rho_name, rho = load()
        assert rho_name == rho.name, 'error in name'
        assert rho.incompressible is False, 'error in incompressible property'
        
    def test_incompressible(self):
        rho_name, rho = load(incompressible=True)
        assert rho_name == rho.name, 'error in name'
        assert rho.incompressible is True, 'error in incompressible property'
    
 
