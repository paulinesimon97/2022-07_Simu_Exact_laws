from ... import not_implemented_warning as NIW
from exact_laws.preprocessing.quantities.pm import *
import os
import numpy as np
import h5py as h5
import pytest 
import warnings

class TestPM:
    
    def test_compressible(self):
        pm = PM()
        assert pm.name == 'pm', 'error in name'
        assert pm.incompressible is False, 'error in incompressible property'
    
    def test_incompressible(self):
        pm = PM(incompressible=True)
        assert pm.name == 'Ipm', 'error in name'
        assert pm.incompressible is True, 'error in incompressible property'
    

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
    dic_expect['Ipm'] = np.array(
        [7 * (sinx[i] * sinx[j] * sinx[k]) * (sinx[i] * sinx[j] * sinx[k]) for i in range(npoint) for j in range(npoint)
         for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['pm'] = np.array(
        [7 / 4 * (sinx[i] * sinx[j] * sinx[k]) * (sinx[i] * sinx[j] * sinx[k]) for i in range(npoint) for j in
         range(npoint) for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_param = {'N': [npoint, npoint, npoint], 'c': [x[1], x[1], x[1]], 'precision': x[1] * x[1] * x[1] * x[1]}
    yield filename, dic_quant, dic_param, dic_expect
    os.remove(filename)
    
class TestCreateDataset:
    
    def test_compressible(self, init_arg_fb):
        with h5.File(init_arg_fb[0], 'w') as f:
            pm = PM()
            pm.create_datasets(f, init_arg_fb[1], init_arg_fb[2])
        with h5.File(init_arg_fb[0], 'r') as f:
            for quant in ['pm']:
                assert np.max(np.abs(init_arg_fb[3][quant] - np.array(f[quant]))) < init_arg_fb[2][
                    'precision'], f"error on {quant} recording" 
    
    def test_incompressible(self, init_arg_fb):
        with h5.File(init_arg_fb[0], 'w') as f:
            pm = PM(incompressible=True)
            pm.create_datasets(f, init_arg_fb[1], init_arg_fb[2])
        with h5.File(init_arg_fb[0], 'r') as f:
            for quant in ['Ipm']:
                assert np.max(np.abs(init_arg_fb[3][quant] - np.array(f[quant]))) < init_arg_fb[2][
                    'precision'], f"error on {quant} recording"  
    
      

class TestLoad:
    def test_compressible(self):
        pm_name, pm = load()
        assert pm_name == pm.name, 'error in name'
        assert pm.incompressible is False, 'error in incompressible property'
        
    def test_incompressible(self):
        pm_name, pm = load(incompressible=True)
        assert pm_name == pm.name, 'error in name'
        assert pm.incompressible is True, 'error in incompressible property'
 
