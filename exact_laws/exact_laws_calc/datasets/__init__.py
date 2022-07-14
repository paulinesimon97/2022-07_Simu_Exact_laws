import h5py as h5
import numpy as np
import logging
from .dataset import Dataset
from ..grids.grid import Grid
from ..grids import load_grid_from_dict


def load(quantities={},grid=Grid(),params={}):  
    return Dataset(params,quantities,grid)      
        
def read_standard_file(filename):
    """extract contents of the standard file called filename"""
    with h5.File(filename, "r") as f:
        quantities = {k:np.ascontiguousarray(f[k]) for k in f.keys() if not "param" in k}
        laws = f['param/laws'][()]
        laws = [law.decode() for law in laws]
        terms = f['param/terms'][()]
        terms = [term.decode() for term in terms]
        grid_params = {k:np.ascontiguousarray(f[f'param/{k}']) for k in f["param"].keys() if k in ['L','c','N']}
        params = {k:eval(str(f[f'param/{k}'][()])) for k in f["param"].keys() if not k in ['L','c','N','laws']}
    params["rho_mean"] = 1  # np.mean(np.sort(data['rho'].flatten()))
    return quantities, params, grid_params, laws, terms

def load_from_standard_file(filename):
    quantities, params, grid_params, laws, terms = read_standard_file
    dataset = load(quantities = quantities, params = params, grid = load_grid_from_dict(grid_params))
    return dataset, laws, terms
