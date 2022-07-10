from ... import not_implemented_warning as NIW
from exact_laws.preprocessing.quantities.graduiso import *
import os
import numpy as np
import h5py as h5
import pytest 
import warnings

class TestGradUIso:
    def test_compressible(self):
        graduiso = GradUIso()
        assert graduiso.name == 'graduiso', 'error in name'
        assert graduiso.incompressible is False, 'error in incompressible property'
    
    def test_incompressible(self):
        graduiso = GradUIso(incompressible=True)
        assert graduiso.name == 'Igraduiso', 'error in name'
        assert graduiso.incompressible is True, 'error in incompressible property'

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
    dic_expect['ppar'] = np.array(
        [1 / 2 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in
         range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['pperp'] = np.array(
        [sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['dxuiso'] = np.array(
        [5 / 4 * cosx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in
         range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['dyuiso'] = np.array(
        [5 / 4 * sinx[i] * cosx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in
         range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['dzuiso'] = np.array(
        [5 / 4 * sinx[i] * sinx[j] * cosx[k] for i in range(npoint) for j in range(npoint) for k in
         range(npoint)]).reshape((npoint, npoint, npoint))
    dic_param = {'N': [npoint, npoint, npoint], 'c': [x[1], x[1], x[1]], 'precision': x[1] * x[1] * x[1] * x[1]}
    yield filename, dic_quant, dic_param, dic_expect
    os.remove(filename)
    
class TestCreateDataset:
    
    def test_compressible(self, init_arg_fp):
        with h5.File(init_arg_fp[0], 'w') as f:
            graduiso = GradUIso()
            graduiso.create_datasets(f, init_arg_fp[1], init_arg_fp[2])
        with h5.File(init_arg_fp[0], 'r') as f:
            for quant in ['dxuiso', 'dyuiso', 'dzuiso']:
                assert np.max(np.abs(init_arg_fp[3][quant] - np.array(f[quant]))) < init_arg_fp[2][
                    'precision'], f"error on {quant} recording"
    
    def test_incompressible(self,init_arg_fp):
        with h5.File(init_arg_fp[0], 'w') as f:
            graduiso = GradUIso(incompressible=True)
            with pytest.raises(Exception) as exc :
                graduiso.create_datasets(f, init_arg_fp[1], init_arg_fp[2])
            assert exc.type == NotImplementedError, 'error on exception raises'

class TestLoad:
    
    def test_compressible(self):
        graduiso_name, graduiso = load()
        assert graduiso_name == graduiso.name, 'error in name'
        assert graduiso.incompressible is False, 'error in incompressible property'
        
    def test_incompressible(self):
        graduiso_name, graduiso = load(incompressible=True)
        assert graduiso_name == graduiso.name, 'error in name'
        assert graduiso.incompressible is True, 'error in incompressible property'
    
