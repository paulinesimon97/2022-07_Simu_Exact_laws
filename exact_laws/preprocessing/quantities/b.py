import numpy as np
import numexpr as ne


class B:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'b'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        if self.incompressible:
            for axis in ('x', 'y', 'z'):
                ds_name = f"{self.name}{axis}"
                file.create_dataset(
                    ds_name,
                    data=dic_quant[ds_name[1:]],
                    shape=dic_param["N"],
                    dtype=np.float64,
                )
        else:
            for axis in ('x', 'y', 'z'):
                ds_name = f"{self.name}{axis}"
                file.create_dataset(
                    ds_name,
                    data=ne.evaluate(f"{ds_name}/sqrt(rho)", local_dict=dic_quant),
                    shape=dic_param["N"],
                    dtype=np.float64,
                )


def load(incompressible=False):
    b = B(incompressible=incompressible)
    return b.name, b
