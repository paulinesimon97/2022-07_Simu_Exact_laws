from ... import not_implemented_warning as NIW
from exact_laws.preprocessing.quantities.gradv import *
import os
import numpy as np
import h5py as h5
import pytest 
import warnings

class TestGradV:
    def test_compressible(self):
        gradv = GradV()
        assert gradv.name == 'gradv', 'error in name'
        assert gradv.incompressible is False, 'error in incompressible property'
    
    def test_incompressible(self):
        gradv = GradV(incompressible=True)
        assert gradv.name == 'Igradv', 'error in name'
        assert gradv.incompressible is True, 'error in incompressible property'

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
    dic_expect = {}
    dic_expect['dxvx'] = np.array(
        [cosx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['dyvx'] = np.array(
        [sinx[i] * cosx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['dzvx'] = np.array(
        [sinx[i] * sinx[j] * cosx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['dxvy'] = np.array(
        [2 * cosx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['dyvy'] = np.array(
        [2 * sinx[i] * cosx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['dzvy'] = np.array(
        [2 * sinx[i] * sinx[j] * cosx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['dxvz'] = np.array(
        [3 * cosx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['dyvz'] = np.array(
        [3 * sinx[i] * cosx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['dzvz'] = np.array(
        [3 * sinx[i] * sinx[j] * cosx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_param = {'N': [npoint, npoint, npoint], 'c': [x[1], x[1], x[1]], 'precision': x[1] * x[1] * x[1] * x[1]}
    yield filename, dic_quant, dic_param, dic_expect
    os.remove(filename)
    
class TestCreateDataset:
    def test_compressible(self, init_arg_fv):
        with h5.File(init_arg_fv[0], 'w') as f:
            gradv = GradV()
            gradv.create_datasets(f, init_arg_fv[1], init_arg_fv[2])
        with h5.File(init_arg_fv[0], 'r') as f:
            for quant in ['dxvx', 'dyvx', 'dzvx', 'dxvy', 'dyvy', 'dzvy', 'dxvz', 'dyvz', 'dzvz']:
                assert np.max(np.abs(init_arg_fv[3][quant] - np.array(f[quant]))) < init_arg_fv[2][
                    'precision'], f"error on {quant} recording"
    
    def test_incompressible(self,init_arg_fv):
        with h5.File(init_arg_fv[0], 'w') as f:
            gradv = GradV(incompressible=True)
            with pytest.raises(Exception) as exc :
                gradv.create_datasets(f, init_arg_fv[1], init_arg_fv[2])
            warnings.warn('no input quantities Iv available')
        
class TestLoad:
    
    def test_compressible(self):
        gradv_name, gradv = load()
        assert gradv_name == gradv.name, 'error in name'
        assert gradv.incompressible is False, 'error in incompressible property'
        
    def test_incompressible(self):
        gradv_name, gradv = load(incompressible=True)
        assert gradv_name == gradv.name, 'error in name'
        assert gradv.incompressible is True, 'error in incompressible property'
