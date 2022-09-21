from typing import List

from .abstract_term import calc_flux_with_numba
from .flux_drpisov import FluxDrpisov, calc_in_point_with_sympy

class FluxDrpmv(FluxDrpisov):
    def __init__(self):
        FluxDrpisov.__init__(self)
    
    def calc(self, vector:List[int], cube_size:List[int], rho, pm, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, pm, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','pm', 'v']

def load():
    return FluxDrpmv()

def print_expr():
    return FluxDrpmv().print_expr()