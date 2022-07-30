from typing import List
from numba import njit
import sympy as sp

from .abstract_term import AbstractTerm, calc_source_with_numba

class Bg17Vbj(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        self.fct = sp.lambdify(
            sp.symbols(
                ("vx'", "vy'", "vz'",
                 "vx", "vy", "vz",
                 "bx'", "by'", "bz'",
                 "bx", "by", "bz",
                 "jx'", "jy'", "jz'",
                 "jx", "jy", "jz"
                )),
            self.expr,
            "numpy",
        )

    def set_sympy_expr(self):
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        IbxP, IbyP, IbzP = sp.symbols(("bx'", "by'", "bz'"))
        IbxNP, IbyNP, IbzNP = sp.symbols(("bx", "by", "bz"))
        IjxP, IjyP, IjzP = sp.symbols(("jx'", "jy'", "jz'"))
        IjxNP, IjyNP, IjzNP = sp.symbols(("jx", "jy", "jz"))

        djx = IjxP - IjxNP
        djy = IjyP - IjyNP
        djz = IjzP - IjzNP

        vXbxP = vyP * IbzP - vzP * IbyP
        vXbyP = vzP * IbxP - vxP * IbzP
        vXbzP = vxP * IbyP - vyP * IbxP
        vXbxNP = vyNP * IbzNP - vzNP * IbyNP
        vXbyNP = vzNP * IbxNP - vxNP * IbzNP
        vXbzNP = vxNP * IbyNP - vyNP * IbxNP

        self.expr = (vXbxP - vXbxNP) * djx + (vXbyP - vXbyNP) * djy + (vXbzP - vXbzNP) * djz
    
    def calc(
        self, vector: List[int], cube_size: List[int], vx, vy, vz, Ibx, Iby, Ibz, Ijx, Ijy, Ijz, **kwarg
    ) -> List[float]:
        # return calc_source_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz, Ibx, Iby, Ibz, Ijx, Ijy, Ijz)
        return calc_source_with_numba(
            calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, Ibx, Iby, Ibz, Ijx, Ijy, Ijz
        )
        
    def variables(self) -> List[str]:
        return ['Ij','Ib','v']

def load():
    return Bg17Vbj()

def print_expr():
    sp.init_printing(use_latex=True)
    return Bg17Vbj().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             vx, vy, vz, 
                             Ibx, Iby, Ibz, 
                             Ijx, Ijy, Ijz, 
                             f=njit(Bg17Vbj().fct)):
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    IbxP, IbyP, IbzP = Ibx[ip, jp, kp], Iby[ip, jp, kp], Ibz[ip, jp, kp]
    IbxNP, IbyNP, IbzNP = Ibx[i, j, k], Iby[i, j, k], Ibz[i, j, k]
    IjxP, IjyP, IjzP = Ijx[ip, jp, kp], Ijy[ip, jp, kp], Ijz[ip, jp, kp]
    IjxNP, IjyNP, IjzNP = Ijx[i, j, k], Ijy[i, j, k], Ijz[i, j, k]
    return f(
        vxP, vyP, vzP, 
        vxNP, vyNP, vzNP, 
        IbxP, IbyP, IbzP, 
        IbxNP, IbyNP, IbzNP, 
        IjxP, IjyP, IjzP, 
        IjxNP, IjyNP, IjzNP
    )
    
@njit
def calc_in_point(i, j, k, ip, jp, kp, vx, vy, vz, Ibx, Iby, Ibz, Ijx, Ijy, Ijz):
    
    djx = Ijx[ip,jp,kp] - Ijx[i,j,k]
    djy = Ijy[ip,jp,kp] - Ijy[i,j,k]
    djz = Ijz[ip,jp,kp] - Ijz[i,j,k]
    
    vXbxP = vy[ip,jp,kp] * Ibz[ip,jp,kp] - vz[ip,jp,kp] * Iby[ip,jp,kp]  
    vXbyP = vz[ip,jp,kp] * Ibx[ip,jp,kp] - vx[ip,jp,kp] * Ibz[ip,jp,kp]  
    vXbzP = vx[ip,jp,kp] * Iby[ip,jp,kp] - vy[ip,jp,kp] * Ibx[ip,jp,kp]  
    vXbxNP = vy[i,j,k] * Ibz[i,j,k] - vz[i,j,k] * Iby[i,j,k]  
    vXbyNP = vz[i,j,k] * Ibx[i,j,k] - vx[i,j,k] * Ibz[i,j,k]  
    vXbzNP = vx[i,j,k] * Iby[i,j,k] - vy[i,j,k] * Ibx[i,j,k]  
    
    return (vXbxP - vXbxNP) * djx + (vXbyP - vXbyNP) * djy + (vXbzP - vXbzNP) * djz