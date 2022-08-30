import numpy as np
import numexpr as ne

from ...mathematical_tools import derivation

def get_original_quantity(dic_quant, dic_param, delete=False):
    dic_quant['jx'], dic_quant['jy'], dic_quant['jz'] = derivation.rot(
        [dic_quant["bx"], dic_quant["by"], dic_quant["bz"]],
        dic_param["c"],
        precision = 4,
        period = True,
    )
    if delete: 
        del(dic_quant[f'bx'],dic_quant[f'by'],dic_quant[f'bz'])

class J:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'j'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        if not ("jx" in dic_quant.keys() or "jy" in dic_quant.keys() or "jz" in dic_quant.keys()):
            get_original_quantity(dic_quant, dic_param)
            
        if self.incompressible:
            for axis in ('x', 'y', 'z'):
                ds_name = f"{self.name}{axis}"
                file.create_dataset(
                    ds_name,
                    data = dic_quant[ds_name[1:]],
                    shape = dic_param["N"],
                    dtype = np.float64,
                )
        else:
            for axis in ('x', 'y', 'z'):
                ds_name = f"{self.name}{axis}"
                file.create_dataset(
                    ds_name,
                    data = ne.evaluate(f"{ds_name}/rho", local_dict=dic_quant),
                    shape = dic_param["N"],
                    dtype = np.float64,
                )


def load(incompressible=False):
    j = J(incompressible=incompressible)
    return j.name, j
