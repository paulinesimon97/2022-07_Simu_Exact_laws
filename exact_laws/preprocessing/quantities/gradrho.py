import numpy as np
import numexpr as ne

from ...math import derivation

class GradRho:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'gradrho'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        if self.incompressible:
            for axisd in ('x', 'y', 'z'):
                ds_name = f"Id{axisd}rho"
                file.create_dataset(
                    ds_name,
                    data = np.zeros(dic_param["N"]),
                    shape = dic_param["N"],
                    dtype = np.float64,
                ) 
        
        else:
            dxrho, dyrho, dzrho = derivation.grad(
                dic_quant[f"rho"], 
                dic_param["c"], 
                precision = 4, 
                period = True
            )
            for axisd in ('x', 'y', 'z'):
                ds_name = f"d{axisd}rho"
                file.create_dataset(
                    ds_name,
                    data = eval(f"d{axisd}rho"),
                    shape = dic_param["N"],
                    dtype = np.float64,
                )      
        
def load(incompressible=False):
    gradrho = GradRho(incompressible=incompressible)
    return gradrho.name, gradrho
