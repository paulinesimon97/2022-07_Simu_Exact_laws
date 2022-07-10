from ... import not_implemented_warning as NIW
from exact_laws.preprocessing.quantities.gradrho import *
import os
import numpy as np
import h5py as h5
import pytest 
import warnings

class TestGradRho:
      
    def test_compressible(self):
        gradrho = GradRho()
        assert gradrho.name == 'gradrho', 'error in name'
        assert gradrho.incompressible is False, 'error in incompressible property'
    
    def test_incompressible(self):
        gradrho = GradRho(incompressible=True)
        assert gradrho.name == 'Igradrho', 'error in name'
        assert gradrho.incompressible is True, 'error in incompressible property'

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
    dic_expect['dxrho'] = np.array(
        [cosx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['dyrho'] = np.array(
        [sinx[i] * cosx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['dzrho'] = np.array(
        [sinx[i] * sinx[j] * cosx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['Idxrho'] = np.zeros(
        (npoint, npoint, npoint))
    dic_expect['Idyrho'] = np.zeros(
        (npoint, npoint, npoint))
    dic_expect['Idzrho'] = np.zeros(
        (npoint, npoint, npoint))
    dic_param = {'N': [npoint, npoint, npoint], 'c': [x[1], x[1], x[1]], 'precision': x[1] * x[1] * x[1] * x[1]}
    yield filename, dic_quant, dic_param, dic_expect
    os.remove(filename)
    
class TestCreateDataset:
    
    def test_compressible(self, init_arg_frho):
        with h5.File(init_arg_frho[0], 'w') as f:
            gradrho = GradRho()
            gradrho.create_datasets(f, init_arg_frho[1], init_arg_frho[2])
        with h5.File(init_arg_frho[0], 'r') as f:
            for quant in ['dxrho','dyrho','dzrho']:
                assert np.max(np.abs(init_arg_frho[3][quant] - np.array(f[quant]))) < init_arg_frho[2][
                    'precision'], f"error on {quant} recording"
    
    def test_incompressible(self,init_arg_frho):
        with h5.File(init_arg_frho[0], 'w') as f:
            gradrho = GradRho(incompressible=True)
            gradrho.create_datasets(f, init_arg_frho[1], init_arg_frho[2])
        with h5.File(init_arg_frho[0], 'r') as f:
            for quant in ['Idxrho','Idyrho','Idzrho']:
                assert np.max(np.abs(init_arg_frho[3][quant] - np.array(f[quant]))) < init_arg_frho[2][
                    'precision'], f"error on {quant} recording"
        
class TestLoad:
    
    def test_compressible(self):
        gradrho_name, gradrho = load()
        assert gradrho_name == gradrho.name, 'error in name'
        assert gradrho.incompressible is False, 'error in incompressible property'
        
    def test_incompressible(self):
        gradrho_name, gradrho = load(incompressible=True)
        assert gradrho_name == gradrho.name, 'error in name'
        assert gradrho.incompressible is True, 'error in incompressible property'