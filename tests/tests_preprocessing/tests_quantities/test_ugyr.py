from ... import not_implemented_warning as NIW
from exact_laws.preprocessing.quantities.ugyr import *
import os
import numpy as np
import h5py as h5
import pytest 
import warnings

class TestUGyr:
    
    def test_compressible(self):
        ugyr = UGyr()
        assert ugyr.name == 'ugyr', 'error in name'
        assert ugyr.incompressible is False, 'error in incompressible property'
    
    def test_incompressible(self):
        ugyr = UGyr(incompressible=True)
        assert ugyr.name == 'Iugyr', 'error in name'
        assert ugyr.incompressible is True, 'error in incompressible property'
    
@pytest.fixture
def init_arg_fp():
    filename = "test_file.h5"
    npoint = 50
    x = np.arange(0, npoint) / npoint * 2 * np.pi
    sinx = np.sin(x)
    cosx = np.cos(x)
    dic_quant = {}
    dic_quant['rho'] = 2 * np.ones([npoint, npoint, npoint])
    dic_quant['ppar'] = np.array(
        [sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['pperp'] = np.array(
        [2 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect = {}
    dic_expect['ugyr'] = np.array(
        [5 / 4 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in
         range(npoint)]).reshape((npoint, npoint, npoint))
    dic_param = {'N': [npoint, npoint, npoint], 'c': [x[1], x[1], x[1]], 'precision': x[1] * x[1] * x[1] * x[1]}
    yield filename, dic_quant, dic_param, dic_expect
    os.remove(filename)
  
class TestCreateDataset:
    
    def test_compressible(self, init_arg_fp):
        with h5.File(init_arg_fp[0], 'w') as f:
            ugyr = UGyr()
            ugyr.create_datasets(f, init_arg_fp[1], init_arg_fp[2])
        with h5.File(init_arg_fp[0], 'r') as f:
            for quant in ['ugyr']:
                assert np.max(np.abs(init_arg_fp[3][quant] - np.array(f[quant]))) < init_arg_fp[2][
                    'precision'], f"error on {quant} recording"
    
    def test_incompressible(self,init_arg_fp):
        with h5.File(init_arg_fp[0], 'w') as f:
            ugyr = UGyr(incompressible=True)
            with pytest.raises(Exception) as exc :
                ugyr.create_datasets(f, init_arg_fp[1], init_arg_fp[2])
            assert exc.type == NotImplementedError, 'error on exception raises'
    
      

class TestLoad:
    
    def test_compressible(self):
        ugyr_name, ugyr = load()
        assert ugyr_name == ugyr.name, 'error in name'
        assert ugyr.incompressible is False, 'error in incompressible property'
        
    def test_incompressible(self):
        ugyr_name, ugyr = load(incompressible=True)
        assert ugyr_name == ugyr.name, 'error in name'
        assert ugyr.incompressible is True, 'error in incompressible property'
    
 
