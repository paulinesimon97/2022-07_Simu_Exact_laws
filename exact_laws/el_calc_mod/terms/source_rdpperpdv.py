from typing import List
import sympy as sp
from .abstract_term import calc_source_with_numba
from .source_rdpisodv import SourceRdpisodv, calc_in_point_with_sympy, calc_with_fourier


class SourceRdpperpdv(SourceRdpisodv):
    def __init__(self):
        SourceRdpisodv.__init__(self)

    def calc(self, vector: List[int], cube_size: List[int], rho, pperp, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, pperp, divv)

    def calc_fourier(self, rho, pperp, divv, **kwarg) -> List:
        return calc_with_fourier(rho, pperp, divv)

    def variables(self) -> List[str]:
        return ["rho", "pgyr", "divv"]


def load():
    return SourceRdpperpdv()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceRdpperpdv().expr
