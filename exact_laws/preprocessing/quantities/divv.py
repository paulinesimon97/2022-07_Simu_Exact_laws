import numpy as np
import numexpr as ne

from ...mathematical_tools import derivation

class DivV:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'divv'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        inc = 'I' * self.incompressible
        divv = derivation.div(
            [dic_quant[f"{inc}vx"], dic_quant[f"{inc}vy"], dic_quant[f"{inc}vz"]], 
            dic_param["c"], 
            precision = 4, 
            period = True
        )
        ds_name = f"{self.name}"
        file.create_dataset(
            ds_name,
            data = divv,
            shape = dic_param["N"],
            dtype = np.float64,
        )    
        
def load(incompressible=False):
    divv = DivV(incompressible=incompressible)
    return divv.name, divv
