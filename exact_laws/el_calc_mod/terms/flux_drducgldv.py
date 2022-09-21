from typing import List

from .abstract_term import calc_flux_with_numba
from .flux_drduisodv import FluxDrduisodv, calc_in_point_with_sympy

class FluxDrducgldv(FluxDrduisodv):
    def __init__(self):
        FluxDrduisodv.__init__(self)
    
    def calc(self, vector:List[int], cube_size:List[int], rho, ucgl, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, ucgl, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','ucgl', 'v']

def load():
    return FluxDrducgldv()

def print_expr():
    return FluxDrducgldv().print_expr()