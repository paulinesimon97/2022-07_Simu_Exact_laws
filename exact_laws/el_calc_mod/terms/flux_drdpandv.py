from typing import List
from numba import njit
import sympy as sp

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrdpandv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ( "rho'", "rho",
            "ppar'", "ppar", "pperp'", "pperp", "pm'", "pm",    
            "vx'", "vy'", "vz'", "vx", "vy", "vz",
            "bx'", "by'", "bz'", "bx", "by", "bz")
        
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
        pparP, pparNP = sp.symbols(("ppar'","ppar"))
        pperpP, pperpNP = sp.symbols(("pperp'","pperp"))
        pmP, pmNP = sp.symbols(("pm'","pm"))
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        bxP, byP, bzP = sp.symbols(("bx'", "by'", "bz'"))
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))

        dr = rhoP - rhoNP
        pressP = (pparP - pperpP) / pmP
        pressNP = (pparNP - pperpNP) / pmNP
        
        dpxx = pressP * bxP * bxP - pressNP * bxNP * bxNP
        dpyy = pressP * byP * byP - pressNP * byNP * byNP
        dpzz = pressP * bzP * bzP - pressNP * bzNP * bzNP
        dpxy = pressP * bxP * byP - pressNP * bxNP * byNP
        dpxz = pressP * bxP * bzP - pressNP * bxNP * bzNP
        dpyz = pressP * byP * bzP - pressNP * byNP * bzNP    
        
        dvx = vxP - vxNP
        dvy = vyP - vyNP
        dvz = vzP - vzNP
        
        self.exprx = dr * (dpxx * dvx + dpxy * dvy + dpxz * dvz)
        self.expry = dr * (dpxy * dvx + dpyy * dvy + dpyz * dvz)
        self.exprz = dr * (dpxz * dvx + dpyz * dvy + dpzz * dvz)
    
    def calc(self, vector:List[int], cube_size:List[int], rho, pperp, ppar, vx, vy, vz, pm, bx, by, bz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, pperp, ppar, vx, vy, vz, pm, bx, by, bz)

    def variables(self) -> List[str]:
        return ['rho','pgyr', 'pm', 'v', 'b']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

def load():
    return FluxDrdpandv()

def print_expr():
    return FluxDrdpandv().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             rho, pperp, ppar, 
                             vx, vy, vz, 
                             pm, bx, by, bz,                       
                             fx=njit(FluxDrdpandv().fctx),
                             fy=njit(FluxDrdpandv().fcty),
                             fz=njit(FluxDrdpandv().fctz)):
    
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    
    pparP, pparNP = ppar[ip, jp, kp], ppar[i, j, k]
    pperpP, pperpNP = pperp[ip, jp, kp], pperp[i, j, k]
    pmP, pmNP = pm[ip, jp, kp], pm[i, j, k]
        
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    
    outx = fx(
        rhoP, rhoNP,
        pparP, pparNP, pperpP, pperpNP, pmP, pmNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP, 
        bxP, byP, bzP, bxNP, byNP, bzNP)
    
    outy = fy(
        rhoP,rhoNP,
        pparP, pparNP, pperpP, pperpNP, pmP, pmNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP, 
        bxP, byP, bzP, bxNP, byNP, bzNP)
    
    outz = fz(
        rhoP, rhoNP,
        pparP, pparNP, pperpP, pperpNP, pmP, pmNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP, 
        bxP, byP, bzP, bxNP, byNP, bzNP)
    
    return outx, outy, outz
