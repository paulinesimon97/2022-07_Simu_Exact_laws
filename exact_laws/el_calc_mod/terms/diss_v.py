
from typing import List
from numba import njit
import sympy as sp

from .abstract_term import AbstractTerm, calc_source_with_numba

class DissV(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        self.fct = sp.lambdify(
            sp.symbols(
                ("rho'","rho",
                 "vx'", "vy'", "vz'", "vx", "vy", "vz",
                 "dx'", "dy'", "dz'", "dx", "dy", "dz",
                )),
            self.expr,
            "numpy",)

    def set_sympy_expr(self):
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        rhoP, rhoNP = sp.symbols(("rho'", "rho"))
        dxP, dyP, dzP = sp.symbols(("dx'", "dy'", "dz'"))
        dxNP, dyNP, dzNP = sp.symbols(("dx", "dy", "dz"))

        self.expr =  (rhoNP + rhoP)*(dxP*vxNP + dxNP*vxP + dyP*vyNP + dyNP*vyP + dzP*vzNP + dzNP*vzP)

    def calc(self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, hdkx, hdky, hdkz, **kwarg
        ) -> List[float]:
        return calc_source_with_numba(
            calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, hdkx, hdky, hdkz)

    def variables(self) -> List[str]:
        return ["hdk", "v", "rho"]
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr


def load():
    return DissV()


def print_expr():
    return DissV().print_expr()


@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             rho,
                             vx, vy, vz, 
                             hdkx, hdky, hdkz,
                             f=njit(DissV().fct)):
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    hdkxP, hdkyP, hdkzP = hdkx[ip, jp, kp], hdky[ip, jp, kp], hdkz[ip, jp, kp]
    hdkxNP, hdkyNP, hdkzNP = hdkx[i, j, k], hdky[i, j, k], hdkz[i, j, k]
    
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    
    return f(rhoP, rhoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP,
        hdkxP, hdkyP, hdkzP, hdkxNP, hdkyNP, hdkzNP
    )
