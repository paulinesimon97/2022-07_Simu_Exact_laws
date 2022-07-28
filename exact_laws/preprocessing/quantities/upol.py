import numpy as np
import numexpr as ne


class UPol:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'upol'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        if self.incompressible:
            raise NotImplementedError("")
        
        ds_name = f"{self.name}"
        cst = (np.mean(ne.evaluate(f"(ppar+pperp+pperp)/3", local_dict=dic_quant))
                / np.mean(ne.evaluate(f"rho**(5/3)", local_dict=dic_quant)))
        rho = dic_quant['rho']
        file.create_dataset(
            ds_name,
            data = ne.evaluate("cst/(5/3-1)*rho**(5/3-1)"),
            shape = dic_param["N"],
            dtype = np.float64,
        )

def load(incompressible=False):
    upol = UPol(incompressible=incompressible)
    return upol.name, upol
