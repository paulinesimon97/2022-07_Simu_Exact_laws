from typing import List
from numba import njit
import sympy as sp
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourcePisovdrdr(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("rho'", "rho",
                      "piso",
                      "vx", "vy", "vz",
                      "dxrho'", "dyrho'", "dzrho'"
                )
        self.fct = sp.lambdify(
            sp.symbols(quantities),
            self.expr,
            "numpy",
        )
        
    def set_sympy_expr(self):
        rhoP, rhoNP = sp.symbols(("rho'", "rho"))
        pisoNP = sp.symbols(("piso"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        dxrhoP, dyrhoP, dzrhoP = sp.symbols(("dxrho'", "dyrho'", "dzrho'"))
        
        scalprod = (vxNP * dxrhoP + vyNP * dyrhoP + vzNP * dzrhoP)
        drho = rhoP - rhoNP

        self.expr = drho * pisoNP * scalprod / rhoP 

    def calc(
        self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, piso, dxrho, dyrho, dzrho, **kwarg
    ) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, piso, dxrho, dyrho, dzrho)

    def variables(self) -> List[str]:
        return ["rho", "gradrho", "v", "piso"]
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr

def load():
    return SourcePisovdrdr()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourcePisovdrdr().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, vx, vy, vz, piso, dxrho, dyrho, dzrho, f=njit(SourcePisovdrdr().fct)):

    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    pisoNP, pisoP = piso[i, j, k], piso[ip, jp, kp]
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    dxrhoP, dyrhoP, dzrhoP = dxrho[ip, jp, kp], dyrho[ip, jp, kp], dzrho[ip, jp, kp]
    dxrhoNP, dyrhoNP, dzrhoNP = dxrho[i, j, k], dyrho[i, j, k], dzrho[i, j, k]

    return (f(rhoP, rhoNP, pisoNP, vxNP, vyNP, vzNP, dxrhoP, dyrhoP, dzrhoP) 
            + f(rhoNP, rhoP, pisoP, vxP, vyP, vzP, dxrhoNP, dyrhoNP, dzrhoNP) )

