from typing import List
from numba import njit
import sympy as sp

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxSs21(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("vx'", "vy'", "vz'", "bx'", "by'", "bz'", "rho'", "u'",
                 "vx", "vy", "vz", "bx", "by", "bz", "rho", "u"
                )
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
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        bxP, byP, bzP = sp.symbols(("bx'", "by'", "bz'"))
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))
        rhoP, rhoNP = sp.symbols(("rho'", "rho"))
        uP, uNP = sp.symbols(("u'", "u"))

        dvx = vxP - vxNP
        dvy = vyP - vyNP
        dvz = vzP - vzNP
        drhovx = rhoP * vxP - rhoNP * vxNP
        drhovy = rhoP * vyP - rhoNP * vyNP
        drhovz = rhoP * vzP - rhoNP * vzNP
        dbx = bxP - bxNP
        dby = byP - byNP
        dbz = bzP - bzNP
        drhobx = rhoP * bxP - rhoNP * bxNP
        drhoby = rhoP * byP - rhoNP * byNP
        drhobz = rhoP * bzP - rhoNP * bzNP
        drho = rhoP - rhoNP
        du = uP - uNP
        
        self.exprx = ((drhovx * dvx + drhovy * dvy + drhovz * dvz) * dvx 
                      + (drhobx * dbx + drhoby * dby + drhobz * dbz) * dvx 
                      + 2 * (drho * du ) * dvx
                      - (drhobx * dvx + drhoby * dvy + drhobz * dvz) * dbx
                      - (drhovx * dbx + drhovy * dby + drhovz * dbz) * dbx)
        self.expry = ((drhovx * dvx + drhovy * dvy + drhovz * dvz) * dvy
                      + (drhobx * dbx + drhoby * dby + drhobz * dbz) * dvy 
                      + 2 * (drho * du ) * dvy
                      - (drhobx * dvx + drhoby * dvy + drhobz * dvz) * dby
                      - (drhovx * dbx + drhovy * dby + drhovz * dbz) * dby)
        self.exprz = ((drhovx * dvx + drhovy * dvy + drhovz * dvz) * dvz
                      + (drhobx * dbx + drhoby * dby + drhobz * dbz) * dvz 
                      + 2 * (drho * du ) * dvz
                      - (drhobx * dvx + drhoby * dvy + drhobz * dvz) * dbz
                      - (drhovx * dbx + drhovy * dby + drhovz * dbz) * dbz)
    
    def calc(self, vector:List[int], cube_size:List[int], vx, vy, vz, bx, by, bz, rho, u, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, bx, by, bz, rho, u)

    def variables(self) -> List[str]:
        return ['v', 'b', 'rho', 'u']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

def load():
    return FluxSs21()

def print_expr():
    return FluxSs21().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             vx, vy, vz, bx, by, bz, rho, u, 
                             fx=njit(FluxSs21().fctx),
                             fy=njit(FluxSs21().fcty),
                             fz=njit(FluxSs21().fctz)):
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    uP, uNP = u[ip, jp, kp], u[i, j, k]
    
    outx = fx(vxP, vyP, vzP, bxP, byP, bzP, rhoP, uP, vxNP, vyNP, vzNP, bxNP, byNP, bzNP, rhoNP, uNP)
    outy = fy(vxP, vyP, vzP, bxP, byP, bzP, rhoP, uP, vxNP, vyNP, vzNP, bxNP, byNP, bzNP, rhoNP, uNP)
    outz = fz(vxP, vyP, vzP, bxP, byP, bzP, rhoP, uP, vxNP, vyNP, vzNP, bxNP, byNP, bzNP, rhoNP, uNP)
    
    return outx, outy, outz
