
from typing import List
import sympy as sp

from .abstract_term import calc_source_with_numba
from .diss_v import DissV, calc_in_point_with_sympy

class DissB(DissV):
    def __init__(self):
        DissV.__init__(self)

    def calc(self, vector: List[int], cube_size: List[int], rho, bx, by, bz, hdmx, hdmy, hdmz, **kwarg
        ) -> List[float]:
        return calc_source_with_numba(
            calc_in_point_with_sympy, *vector, *cube_size, rho, bx, by, bz, hdmx, hdmy, hdmz)

    def variables(self) -> List[str]:
        return ["hdm", "b", "rho"]


def load():
    return DissB()


def print_expr():
    return DissB().print_expr()
