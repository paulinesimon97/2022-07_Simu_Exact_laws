import numpy as np
import numexpr as ne

from ...mathematical_tools import derivation

class W:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'w'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        inc = 'I' * self.incompressible
        wx, wy, wz = derivation.rot(
            [dic_quant[f"{inc}vx"], dic_quant[f"{inc}vy"], dic_quant[f"{inc}vz"]],
            dic_param["c"],
            precision = 4,
            period = True
        )
        for axis in ('x', 'y', 'z'):
            ds_name = f"{self.name}{axis}"
            file.create_dataset(
                ds_name,
                data = eval(f"w{axis}"),
                shape = dic_param["N"],
                dtype = np.float64,
            )      
        


def load(incompressible=False):
    w = W(incompressible=incompressible)
    return w.name, w
