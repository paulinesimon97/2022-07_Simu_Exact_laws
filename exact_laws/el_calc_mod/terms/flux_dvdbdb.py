from typing import List
from numba import njit
import sympy as sp

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDvdbdb(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("vx'", "vy'", "vz'",
                 "vx", "vy", "vz",
                 "bx'", "by'", "bz'",
                 "bx", "by", "bz",
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
        IbxP, IbyP, IbzP = sp.symbols(("bx'", "by'", "bz'"))
        IbxNP, IbyNP, IbzNP = sp.symbols(("bx", "by", "bz"))

        dvx = vxP - vxNP
        dvy = vyP - vyNP
        dvz = vzP - vzNP
    
        dIbx = IbxP - IbxNP
        dIby = IbyP - IbyNP
        dIbz = IbzP - IbzNP
    
        self.exprx = (dvx * dIbx + dvy * dIby + dvz * dIbz) * dIbx
        self.expry = (dvx * dIbx + dvy * dIby + dvz * dIbz) * dIby
        self.exprz = (dvx * dIbx + dvy * dIby + dvz * dIbz) * dIbz
        
    def calc(self, vector:List[int], cube_size:List[int], vx, vy, vz, Ibx, Iby, Ibz, **kwarg) -> List[float]:
        #return calc_flux_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz, Ibx, Iby, Ibz)
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, Ibx, Iby, Ibz)

    def variables(self) -> List[str]:
        return ['Ib','v']

def load():
    return FluxDvdbdb()

def print_expr():
    sp.init_printing(use_latex=True)
    return FluxDvdbdb().exprx, FluxDvdbdb().expry, FluxDvdbdb().exprz

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             vx, vy, vz, 
                             Ibx, Iby, Ibz,  
                             fx=njit(FluxDvdbdb().fctx),
                             fy=njit(FluxDvdbdb().fcty),
                             fz=njit(FluxDvdbdb().fctz)):
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    IbxP, IbyP, IbzP = Ibx[ip, jp, kp], Iby[ip, jp, kp], Ibz[ip, jp, kp]
    IbxNP, IbyNP, IbzNP = Ibx[i, j, k], Iby[i, j, k], Ibz[i, j, k]
    
    outx = fx(
        vxP, vyP, vzP, 
        vxNP, vyNP, vzNP, 
        IbxP, IbyP, IbzP, 
        IbxNP, IbyNP, IbzNP
    )
    outy = fy(
        vxP, vyP, vzP, 
        vxNP, vyNP, vzNP, 
        IbxP, IbyP, IbzP, 
        IbxNP, IbyNP, IbzNP
    )
    outz = fz(
        vxP, vyP, vzP, 
        vxNP, vyNP, vzNP, 
        IbxP, IbyP, IbzP, 
        IbxNP, IbyNP, IbzNP
    )
    
    return outx, outy, outz

@njit
def calc_in_point(i, j, k, ip, jp, kp, vx, vy, vz, Ibx, Iby, Ibz):
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    dIbx = Ibx[ip,jp,kp] - Ibx[i,j,k]
    dIby = Iby[ip,jp,kp] - Iby[i,j,k]
    dIbz = Ibz[ip,jp,kp] - Ibz[i,j,k]
    
    fx = (dvx * dIbx + dvy * dIby + dvz * dIbz) * dIbx
    fy = (dvx * dIbx + dvy * dIby + dvz * dIbz) * dIby
    fz = (dvx * dIbx + dvy * dIby + dvz * dIbz) * dIbz
    
    return fx, fy, fz