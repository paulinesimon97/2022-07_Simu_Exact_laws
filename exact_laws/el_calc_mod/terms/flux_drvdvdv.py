from typing import List
from numba import njit
import sympy as sp

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrvdvdv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ( "rho'", "rho",
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
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
    
        dvx = vxP - vxNP
        dvy = vyP - vyNP
        dvz = vzP - vzNP
        
        drvx = rhoP * vxP - rhoNP * vxNP
        drvy = rhoP * vyP - rhoNP * vyNP
        drvz = rhoP * vzP - rhoNP * vzNP
    
        self.exprx = (drvx * dvx + drvy * dvy + drvz * dvz) * dvx
        self.expry = (drvx * dvx + drvy * dvy + drvz * dvz) * dvy
        self.exprz = (drvx * dvx + drvy * dvy + drvz * dvz) * dvz
    
    def calc(self, vector:List[int], cube_size:List[int], rho, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','v']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

def load():
    return FluxDrvdvdv()

def print_expr():
    return FluxDrvdvdv().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp,
                             rho, 
                             vx, vy, vz,   
                             fx=njit(FluxDrvdvdv().fctx),
                             fy=njit(FluxDrvdvdv().fcty),
                             fz=njit(FluxDrvdvdv().fctz)):
    
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    
    outx = fx(
        rhoP, rhoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP)
    
    outy = fy(
        rhoP,rhoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP)
    
    outz = fz(
        rhoP, rhoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP)
    
    return outx, outy, outz
    