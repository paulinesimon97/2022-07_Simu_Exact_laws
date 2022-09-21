from typing import List
from numba import njit
import sympy as sp

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDbdbdj(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("jx'", "jy'", "jz'", "jx", "jy", "jz",
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
        IjxP, IjyP, IjzP = sp.symbols(("jx'", "jy'", "jz'"))
        IjxNP, IjyNP, IjzNP = sp.symbols(("jx", "jy", "jz"))
        IbxP, IbyP, IbzP = sp.symbols(("bx'", "by'", "bz'"))
        IbxNP, IbyNP, IbzNP = sp.symbols(("bx", "by", "bz"))

        dIjx = IjxP - IjxNP
        dIjy = IjyP - IjyNP
        dIjz = IjzP - IjzNP
    
        dIbx = IbxP - IbxNP
        dIby = IbyP - IbyNP
        dIbz = IbzP - IbzNP
    
        self.exprx = (dIbx * dIbx + dIby * dIby + dIbz * dIbz) * dIjx
        self.expry = (dIbx * dIbx + dIby * dIby + dIbz * dIbz) * dIjy
        self.exprz = (dIbx * dIbx + dIby * dIby + dIbz * dIbz) * dIjz
        
    def calc(self, vector:List[int], cube_size:List[int], Ijx, Ijy, Ijz, Ibx, Iby, Ibz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, Ijx, Ijy, Ijz, Ibx, Iby, Ibz)

    def variables(self) -> List[str]:
        return ['Ib','Ij']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

def load():
    return FluxDbdbdj()

def print_expr():
    return FluxDbdbdj().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                            Ijx, Ijy, Ijz,
                             Ibx, Iby, Ibz,  
                             fx=njit(FluxDbdbdj().fctx),
                             fy=njit(FluxDbdbdj().fcty),
                             fz=njit(FluxDbdbdj().fctz)):
    
    IjxP, IjyP, IjzP = Ijx[ip, jp, kp], Ijy[ip, jp, kp], Ijz[ip, jp, kp]
    IjxNP, IjyNP, IjzNP = Ijx[i, j, k], Ijy[i, j, k], Ijz[i, j, k]
    
    IbxP, IbyP, IbzP = Ibx[ip, jp, kp], Iby[ip, jp, kp], Ibz[ip, jp, kp]
    IbxNP, IbyNP, IbzNP = Ibx[i, j, k], Iby[i, j, k], Ibz[i, j, k]
    
    outx = fx(
        IjxP, IjyP, IjzP, IjxNP, IjyNP, IjzNP, 
        IbxP, IbyP, IbzP, IbxNP, IbyNP, IbzNP)
    
    outy = fy(
        IjxP, IjyP, IjzP, IjxNP, IjyNP, IjzNP, 
        IbxP, IbyP, IbzP, IbxNP, IbyNP, IbzNP)
    
    outz = fz(
        IjxP, IjyP, IjzP, IjxNP, IjyNP, IjzNP, 
        IbxP, IbyP, IbzP, IbxNP, IbyNP, IbzNP)
    
    return outx, outy, outz

    
    