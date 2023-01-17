from typing import List
from numba import njit
import sympy as sp

from .abstract_term import AbstractTerm, calc_source_with_numba

class CorRbb(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        self.fct = sp.lambdify(
            sp.symbols(
                ("rho'","rho",
                 "bx'", "by'", "bz'", "bx", "by", "bz",
                )),
            self.expr,
            "numpy",)

    def set_sympy_expr(self):
        bxP, byP, bzP = sp.symbols(("bx'", "by'", "bz'"))
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))
        rhoP, rhoNP = sp.symbols(("rho'", "rho"))
        
        psbb = bxP*bxNP + byP*byNP + bzP*bzNP

        self.expr = (rhoP+rhoNP)*psbb/2 /2  

    def calc(
        self, vector: List[int], cube_size: List[int], rho, bx, by, bz, **kwarg
    ) -> List[float]:
        return calc_source_with_numba(
            calc_in_point_with_sympy, *vector, *cube_size, rho, bx, by, bz)

    def variables(self) -> List[str]:
        return ["b", "rho"]
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr


def load():
    return CorRbb()


def print_expr():
    return CorRbb().print_expr()


@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             rho,
                             bx, by, bz, 
                             f=njit(CorRbb().fct)):
    
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    
    return f(rhoP, rhoNP,
        bxP, byP, bzP, bxNP, byNP, bzNP
    )

