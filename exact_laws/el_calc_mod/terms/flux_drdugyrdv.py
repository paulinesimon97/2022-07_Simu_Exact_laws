from typing import List

from .abstract_term import calc_flux_with_numba
from .flux_drduisodv import FluxDrduisodv, calc_in_point_with_sympy

class FluxDrdugyrdv(FluxDrduisodv):
    def __init__(self):
        FluxDrduisodv.__init__(self)
    
    def calc(self, vector:List[int], cube_size:List[int], rho, ugyr, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, ugyr, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','ugyr', 'v']

def load():
    return FluxDrdugyrdv()

def print_expr():
    return FluxDrdugyrdv().print_expr()