import numpy as np
import numexpr as ne

from ...mathematical_tools import derivation


class DivB:
    def __init__(self, incompressible=False):
        self.name = "I" * incompressible + "divb"
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        if self.incompressible:
            divb = derivation.div(
                [dic_quant[f"bx"], dic_quant[f"by"], dic_quant[f"bz"]], 
                dic_param["c"], 
                precision = 4, 
                period = True
            )
        else:
            divb = derivation.div(
                [
                    ne.evaluate("bx/sqrt(rho)", local_dict=dic_quant),
                    ne.evaluate("by/sqrt(rho)", local_dict=dic_quant),
                    ne.evaluate("bz/sqrt(rho)", local_dict=dic_quant),
                ],
                dic_param["c"],
                precision = 4,
                period = True,
            )
        ds_name = f"{self.name}"
        file.create_dataset(
            ds_name,
            data = divb,
            shape = dic_param["N"],
            dtype = np.float64,
        )

def load(incompressible=False):
    divb = DivB(incompressible=incompressible)
    return divb.name, divb
