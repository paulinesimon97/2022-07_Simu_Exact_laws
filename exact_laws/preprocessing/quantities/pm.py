import numpy as np
import numexpr as ne


class PM:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'pm'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        if self.incompressible:
            ds_name = f"{self.name}"
            file.create_dataset(
                ds_name,
                data = ne.evaluate("bx*bx/2+by*by/2+bz*bz/2", local_dict=dic_quant),
                shape = dic_param["N"],
                dtype = np.float64,
            )
        else:
            ds_name = f"{self.name}"
            file.create_dataset(
                ds_name,
                data = ne.evaluate("(bx*bx+by*by+bz*bz)/2/rho", local_dict=dic_quant),
                shape = dic_param["N"],
                dtype = np.float64,
            )

def load(incompressible=False):
    pm = PM(incompressible=incompressible)
    return pm.name, pm
