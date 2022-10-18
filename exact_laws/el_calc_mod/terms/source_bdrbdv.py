from typing import List
from numba import njit
import sympy as sp
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceBdrbdv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("rho'", "rho",
                      "bx'", "by'", "bz'",
                      "bx", "by", "bz",
                      "divv'"
                )
        self.fct = sp.lambdify(
            sp.symbols(quantities),
            self.expr,
            "numpy",
        )
        
    def set_sympy_expr(self):
        rhoP, rhoNP = sp.symbols(("rho'", "rho"))
        bxP, byP, bzP = sp.symbols(("bx'", "by'", "bz'"))
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))
        divvP = sp.symbols(("divv'"))

        drbx = rhoP * bxP - rhoNP * bxNP
        drby = rhoP * byP - rhoNP * byNP
        drbz = rhoP * bzP - rhoNP * bzNP
        
        self.expr = (bxNP * drbx + byNP * drby + bzNP * drbz) * divvP

    def calc(self, vector: List[int], cube_size: List[int], rho, bx, by, bz, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, bx, by, bz, divv)

    def variables(self) -> List[str]:
        return ["rho", "b", "divv"]

    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr

def load():
    return SourceBdrbdv()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceBdrbdv().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, bx, by, bz, divv,f=njit(SourceBdrbdv().fct)):

    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    divvP, divvNP = divv[ip, jp, kp], divv[i, j, k]

    return (f(rhoP, rhoNP, bxP, byP, bzP, bxNP, byNP, bzNP, divvP) 
            + f(rhoNP, rhoP, bxNP, byNP, bzNP, bxP, byP, bzP, divvNP))
