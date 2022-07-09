import numpy as np
import numexpr as ne

from ...math import derivation
from .j import J

class DivJ:
    def __init__(self, incompressible=False):
        self.name = "I" * incompressible + "divj"
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        if not ("jx" in dic_quant.keys() or "jy" in dic_quant.keys() or "jz" in dic_quant.keys()):
            J.get_original_quantity(dic_quant, dic_param)
            
        if self.incompressible:
            divj = derivation.div(
                [dic_quant['jx'], dic_quant['jy'], dic_quant['jz']],
                dic_param["c"],
                precision = 4,
                period = True,
            )
            
        else:
            divj = derivation.div(
                [
                    ne.evaluate("jx/rho", local_dict=dic_quant),
                    ne.evaluate("jy/rho", local_dict=dic_quant),
                    ne.evaluate("jz/rho", local_dict=dic_quant),
                ],
                dic_param["c"],
                precision = 4,
                period = True,
            )
        ds_name = f"{self.name}"
        file.create_dataset(
            ds_name,
            data = divj,
            shape = dic_param["N"],
            dtype = np.float64,
        )

def load(incompressible=False):
    divj = DivJ(incompressible=incompressible)
    return divj.name, divj
