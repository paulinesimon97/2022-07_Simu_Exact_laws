from typing import List
from numba import njit
import sympy as sp

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrpisov(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ( "rho'", "rho", "piso'", "piso",    
            "vx'", "vy'", "vz'", "vx", "vy", "vz")
        
        self.fctx = sp.lambdify(
            sp.symbols(quantities),
            self.exprx,
            "numpy",
        )
        self.fcty = sp.lambdify(
            sp.symbols(quantities),
            self.expry,
            "numpy",
        )
        self.fctz = sp.lambdify(
            sp.symbols(quantities),
            self.exprz,
            "numpy",
        )
    
    def set_sympy_expr(self):
        rhoP, rhoNP = sp.symbols(("rho'","rho"))
        pisoP, pisoNP = sp.symbols(("piso'","piso"))
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        
        rpNP = (rhoP + rhoNP) * pisoNP
        rpP = (rhoP + rhoNP) * pisoP
        
        self.exprx = rpNP * vxP - rpP * vxNP
        self.expry = rpNP * vyP - rpP * vyNP
        self.exprz = rpNP * vzP - rpP * vzNP
    
    def calc(self, vector:List[int], cube_size:List[int], rho, piso, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, piso, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','piso', 'v']

    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz
    
def load():
    return FluxDrpisov()

def print_expr():
    return FluxDrpisov().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, piso, vx, vy, vz,
                             fx=njit(FluxDrpisov().fctx),
                             fy=njit(FluxDrpisov().fcty),
                             fz=njit(FluxDrpisov().fctz)):
    
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    
    pisoP, pisoNP = piso[ip, jp, kp], piso[i, j, k]
        
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    
    outx = fx(
        rhoP, rhoNP, pisoP, pisoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP)
    
    outy = fy(
        rhoP, rhoNP, pisoP, pisoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP)
    
    outz = fz(
        rhoP, rhoNP, pisoP, pisoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP)
    
    return outx, outy, outz
    