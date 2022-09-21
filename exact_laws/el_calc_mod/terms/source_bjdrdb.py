from typing import List
from numba import njit
import sympy as sp
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceBjdrdb(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("rho'", "rho",
                      "jx'", "jy'", "jz'",
                      "bx", "by", "bz",
                      "divb'"
                )
        self.fct = sp.lambdify(
            sp.symbols(quantities),
            self.expr,
            "numpy",
        )
        
    def set_sympy_expr(self):
        rhoP, rhoNP = sp.symbols(("rho'", "rho"))
        jxP, jyP, jzP = sp.symbols(("jx'", "jy'", "jz'"))
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))
        divbP = sp.symbols(("divb'"))

        drho = rhoP - rhoNP
        psjb = jxP * bxNP + jyP * byNP + jzP * bzNP
        
        self.expr = drho * psjb * divbP
        
    def calc(self, vector: List[int], cube_size: List[int],
             rho, bx, by, bz, jx, jy, jz, divb, **kwarg) -> (float):
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size,
                                      rho, bx, by, bz, jx, jy, jz, divb)

    def variables(self) -> List[str]:
        return ["rho", "b", "j", "divb"]
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr


def load():
    return SourceBjdrdb()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceBjdrdb().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             rho, bx, by, bz, jx, jy, jz, divb,  
                             f=njit(SourceBjdrdb().fct)):
    
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    jxP, jyP, jzP = jx[ip, jp, kp], jy[ip, jp, kp], jz[ip, jp, kp]
    jxNP, jyNP, jzNP = jx[i, j, k], jy[i, j, k], jz[i, j, k]
    divbP, divbNP = divb[ip, jp, kp], divb[i, j, k]
    
    return (f(rhoP, rhoNP, jxP, jyP, jzP, bxNP, byNP, bzNP, divbP) 
            + f(rhoNP, rhoP, jxNP, jyNP, jzNP, bxP, byP, bzP, divbNP))


