from typing import List
from numba import njit
import sympy as sp

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDjbdrb(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ( "rho'", "rho",
                      "bx'", "by'", "bz'", "bx", "by", "bz",
                      "jx'", "jy'", "jz'", "jx", "jy", "jz")
        
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
        jxP, jyP, jzP = sp.symbols(("jx'", "jy'", "jz'"))
        jxNP, jyNP, jzNP = sp.symbols(("jx", "jy", "jz"))
        bxP, byP, bzP = sp.symbols(("bx'", "by'", "bz'"))
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))

        drjbx = (jyP * bzP - byP * jzP) - (jyNP * bzNP - byNP * jzNP)
        drjby = (jzP * bxP - bzP * jxP) - (jzNP * bxNP - bzNP * jxNP)
        drjbz = (jxP * byP - bxP * jyP) - (jxNP * byNP - bxNP * jyNP)
    
        dbx = (rhoP * bxP + rhoNP * bxNP)/2
        dby = (rhoP * byP + rhoNP * byNP)/2
        dbz = (rhoP * bzP + rhoNP * bzNP)/2
    
        self.exprx = drjby * dbz - drjbz * dby
        self.expry = drjbz * dbx - drjbx * dbz
        self.exprz = drjbx * dby - drjby * dbx
    
    def calc(self, vector:List[int], cube_size:List[int], rho, bx, by, bz, jx, jy, jz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, bx, by, bz, jx, jy, jz)

    def variables(self) -> List[str]:
        return ['rho','b','j']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

def load():
    return FluxDjbdrb()

def print_expr():
    return FluxDjbdrb().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp,
                             rho, 
                             bx, by, bz, 
                             jx, jy, jz,  
                             fx=njit(FluxDjbdrb().fctx),
                             fy=njit(FluxDjbdrb().fcty),
                             fz=njit(FluxDjbdrb().fctz)):
    
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    
    jxP, jyP, jzP = jx[ip, jp, kp], jy[ip, jp, kp], jz[ip, jp, kp]
    jxNP, jyNP, jzNP = jx[i, j, k], jy[i, j, k], jz[i, j, k]
    
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    
    outx = fx(
        rhoP, rhoNP,
        bxP, byP, bzP, bxNP, byNP, bzNP, 
        jxP, jyP, jzP, jxNP, jyNP, jzNP)
    
    outy = fy(
        rhoP,rhoNP,
        bxP, byP, bzP, bxNP, byNP, bzNP, 
        jxP, jyP, jzP, jxNP, jyNP, jzNP)
    
    outz = fz(
        rhoP, rhoNP,
        bxP, byP, bzP, bxNP, byNP, bzNP, 
        jxP, jyP, jzP, jxNP, jyNP, jzNP)
    
    return outx, outy, outz
    
    