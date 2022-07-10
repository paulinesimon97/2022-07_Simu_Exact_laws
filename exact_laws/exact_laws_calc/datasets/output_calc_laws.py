import h5py as h5
import numpy as np
import logging

from . import Dataset
from ..grids.scale_logcyl import Grid_scale_logcyl

class OutputCalcLaws(Dataset):
    """Classe des données finales
    Contient : un dictionaire de paramètre param, un dictionaire avec les données datadic et des information sur la grille des données scale
    """

    def __init__(self, params, coeffs, param_grid, another_grid):
        Dataset.__init__(self,params=params)
        self.set_grid_from_another_grid(another_grid,param_grid)
        self.set_quantities_from_term_names(params['terms'])
        self.coeffs = coeffs
        self.state = 0  # int indiquant l'état de remplissage des cubes de données
    
    def set_grid_from_another_grid(self,another_grid,param_grid):
        if param_grid['coord'] == 'logcyl':
            N = (another_grid.N / 2).astype(int)
            L = another_grid.L / 2
            c = another_grid.c
            self.grid = Grid_scale_logcyl(N=N, L=L, c=c, 
                                     Nmax_scale=param_grid['Nmax_scale'],
                                     Nmax_list=param_grid['Nmax_list'], 
                                     kind=param_grid["kind"]
                                     )
        else: 
            raise NotImplementedError('')
        
        
    def set_quantities_from_term_names(self,term_names):
        scalar_shape = self.grid.N  # taille pour termes scalaires
        vector_shape = np.append(scalar_shape, 3)  # taille pour termes vectoriels
        div_shape = np.append(vector_shape, 2)  # taille pour termes finalement dérivés
        for i, t in enumerate(term_names):
            if t.startswith("flux"):
                self.quantities[t] = np.zeros(vector_shape, dtype=np.float64)
                self.quantities[f"div_{t}"] = np.zeros(scalar_shape, dtype=np.float64)
                self.quantities[f"term_div_{t}"] = np.zeros(div_shape, dtype=np.float64)
            else:
                self.quantities[t] = np.zeros(scalar_shape, dtype=np.float64)
    
    def check(self, name):
        message = f"Check OutputCalcLaws object {name}:"
        message += f"\n\t Params:"
        for k in self.params.keys():
            message += f"\n\t\t - {k} = {self.params[k]}"
        message += f"\n\t Quantities:"
        for k in self.quantities.keys():
            message += f"\n\t\t - shape {k} = {np.shape(self.quantities[k])}"
        message += f"\n\t Coeffs:"
        for k in self.coeffs.keys():
            if k.startswith('law_'):
                message += f"\n\t\t - {k}:"
                for t in self.coeffs[k].keys():
                    message += f"\n\t\t\t - {t}: {self.coeffs[k][t]}"
        logging.info(message)
        self.grid.check(name+'.grid')
                
    def record_to_h5file(self, filename):
        with h5.File(filename, "w") as file:
            file.create_group("params")
            for k in self.params.keys():
                file["params"].create_dataset(k, data=self.params[k])
            file.create_group("coeffs")
            for k in self.coeffs.keys():
                file["coeffs"].create_dataset(k, data=self.coeffs[k])
            file.create_group("quantities")
            for k in self.datadic.keys():
                file["quantities"].create_dataset(k, data=self.quantities[k])
            file.create_group("grid")
            for k in self.grid.grid.keys():
                if "listperp" in k:
                    n = len(max(self.grid.grid[k], key=len))
                    lst_2 = [x + [[np.nan, np.nan]] * (n - len(x)) for x in self.grid.grid[k]]
                    file["grid"].create_dataset(k, data=lst_2)
                elif "listnorm" in k:
                    n = len(max(self.grid.grid[k], key=len))
                    lst_2 = [x + [np.nan] * (n - len(x)) for x in self.grid.grid[k]]
                    file["grid"].create_dataset(k, data=lst_2)
                else:
                    file["grid"].create_dataset(k, data=self.grid.grid[k])