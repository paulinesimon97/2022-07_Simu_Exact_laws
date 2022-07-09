import numpy as np
import numexpr as ne

from ...math import derivation

class GradUIso:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'graduiso'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        if self.incompressible:
            raise NotImplementedError("")
        
        uiso = ne.evaluate("(ppar+pperp+pperp)/2/rho", local_dict=dic_quant)
        dxuiso, dyuiso, dzuiso = derivation.grad(
            uiso, 
            dic_param["c"], 
            precision = 4, 
            period = True
        )
        for axisd in ('x', 'y', 'z'):
            ds_name = f"d{axisd}uiso"
            file.create_dataset(
                ds_name,
                data = eval(f"d{axisd}uiso"),
                shape = dic_param["N"],
                dtype = np.float64,
            )      
        
def load(incompressible=False):
    graduiso = GradUIso(incompressible=incompressible)
    return graduiso.name, graduiso
