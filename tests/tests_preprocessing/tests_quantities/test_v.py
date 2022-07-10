from ... import not_implemented_warning as NIW
from exact_laws.preprocessing.quantities.v import *
import os
import numpy as np
import h5py as h5
import pytest 

class TestV:
    def test_compressible(self):
        v = V()
        assert v.name == 'v', 'error in name'
        assert v.incompressible is False, 'error in incompressible property'
    
    def test_incompressible(self):
        v = V(incompressible=True)
        assert v.name == 'Iv', 'error in name'
        assert v.incompressible is True, 'error in incompressible property'

@pytest.fixture
def init_arg_fv():
    filename = "test_file.h5"
    npoint = 50
    x = np.arange(0, npoint) / npoint * 2 * np.pi
    sinx = np.sin(x)
    cosx = np.cos(x)
    dic_quant = {}
    dic_quant['vx'] = np.array(
        [sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['vy'] = np.array(
        [2 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['vz'] = np.array(
        [3 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect = dic_quant
    dic_param = {'N': [npoint, npoint, npoint], 'c': [x[1], x[1], x[1]], 'precision': x[1] * x[1] * x[1] * x[1]}
    yield filename, dic_quant, dic_param, dic_expect
    os.remove(filename)
        
class TestCreateDataset:
    def test_compressible(self, init_arg_fv):
        with h5.File(init_arg_fv[0], 'w') as f:
            v = V()
            v.create_datasets(f, init_arg_fv[1], init_arg_fv[2])
        with h5.File(init_arg_fv[0], 'r') as f:
            for quant in ['vx', 'vy', 'vz']:
                assert np.max(np.abs(init_arg_fv[3][quant] - np.array(f[quant]))) < init_arg_fv[2][
                    'precision'], f"error on {quant} recording"
    
    def test_incompressible(self,init_arg_fv):
        with h5.File(init_arg_fv[0], 'w') as f:
            v = V(incompressible=True)
            with pytest.raises(Exception) as exc :
                v.create_datasets(f, init_arg_fv[1], init_arg_fv[2])
            assert exc.type == NotImplementedError, 'error on exception raises'
        

class TestLoad:
    def test_compressible(self):
        v_name, v = load()
        assert v_name == v.name, 'error in name'
        assert v.incompressible is False, 'error in incompressible property'
        
    def test_incompressible(self):
        v_name, v = load(incompressible=True)
        assert v_name == v.name, 'error in name'
        assert v.incompressible is True, 'error in incompressible property'
    
 
