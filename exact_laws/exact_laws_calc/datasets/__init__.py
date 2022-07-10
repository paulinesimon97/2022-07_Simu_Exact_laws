import h5py as h5
import numpy as np
import logging

from ..grids import Grid

class Dataset:
    """Classe des données"""

    def __init__(self, params={}, quantities={}, grid=Grid()):
        """Initialisation of the mother class data"""
        self.params = params
        self.quantities = quantities  # dictionary 
        self.grid = grid  # grid
    
    def check(self,name):
        message = f"Check Dataset object {name}:"
        message += f"\n\t Params:"
        for k in self.params.keys():
            message += f"\n\t\t - {k} = {self.params[k]}"
        message += f"\n\t Quantities:"
        for k in self.quantities.keys():
            #tab = self.quantities[k]
            message += f"\n\t\t - {k}"  # = {np.mean(np.sort(tab.copy().reshape(np.product(tab.shape))))}")
        logging.info(message)
        self.grid.check(name+'.grid')
            
        
def read_standard_file(filename):
    """extract contents of the standard file called filename"""
    with h5.File(filename, "r") as f:
        quantities = {k:np.ascontiguousarray(f[k]) for k in f.keys() if not "param" in k}
        laws = f['param/laws'][()]
        laws = [law.decode() for law in laws]
        grid = {k:np.ascontiguousarray(f[f'param/{k}']) for k in f["param"].keys() if k in ['L','c','N']}
        params = {k:eval(str(f[f'param/{k}'][()])) for k in f["param"].keys() if not k in ['L','c','N','laws']}
    params["rho_mean"] = 1  # np.mean(np.sort(data['rho'].flatten()))
    return quantities, params, laws, grid
