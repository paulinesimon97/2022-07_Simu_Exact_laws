from typing import List
from numba import njit
import sympy as sp

from .abstract_term import AbstractTerm, calc_source_with_numba


class Bg17Jbv(AbstractTerm):
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

        dvx = vxP - vxNP
        dvy = vyP - vyNP
        dvz = vzP - vzNP

        jXbxP = IjyP * IbzP - IjzP * IbyP
        jXbyP = IjzP * IbxP - IjxP * IbzP
        jXbzP = IjxP * IbyP - IjyP * IbxP
        jXbxNP = IjyNP * IbzNP - IjzNP * IbyNP
        jXbyNP = IjzNP * IbxNP - IjxNP * IbzNP
        jXbzNP = IjxNP * IbyNP - IjyNP * IbxNP

        self.expr = (jXbxP - jXbxNP) * dvx + (jXbyP - jXbyNP) * dvy + (jXbzP - jXbzNP) * dvz

    def calc(
        self, vector: List[int], cube_size: List[int], vx, vy, vz, Ibx, Iby, Ibz, Ijx, Ijy, Ijz, **kwarg
    ) -> List[float]:
        # return calc_source_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz, Ibx, Iby, Ibz, Ijx, Ijy, Ijz)
        return calc_source_with_numba(
            calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, Ibx, Iby, Ibz, Ijx, Ijy, Ijz
        )

    def variables(self) -> List[str]:
        return ["Ij", "Ib", "v"]


def load():
    return Bg17Jbv()


def print_expr():
    sp.init_printing(use_latex=True)
    return Bg17Jbv().expr


@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             vx, vy, vz, 
                             Ibx, Iby, Ibz, 
                             Ijx, Ijy, Ijz, 
                             f=njit(Bg17Jbv().fct)):
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
def calc_in_point(i, j, k, ip, jp, kp, 
                  vx, vy, vz, 
                  Ibx, Iby, Ibz, 
                  Ijx, Ijy, Ijz):

    dvx = vx[ip, jp, kp] - vx[i, j, k]
    dvy = vy[ip, jp, kp] - vy[i, j, k]
    dvz = vz[ip, jp, kp] - vz[i, j, k]

    jXbxP = Ijy[ip, jp, kp] * Ibz[ip, jp, kp] - Ijz[ip, jp, kp] * Iby[ip, jp, kp]
    jXbyP = Ijz[ip, jp, kp] * Ibx[ip, jp, kp] - Ijx[ip, jp, kp] * Ibz[ip, jp, kp]
    jXbzP = Ijx[ip, jp, kp] * Iby[ip, jp, kp] - Ijy[ip, jp, kp] * Ibx[ip, jp, kp]
    jXbxNP = Ijy[i, j, k] * Ibz[i, j, k] - Ijz[i, j, k] * Iby[i, j, k]
    jXbyNP = Ijz[i, j, k] * Ibx[i, j, k] - Ijx[i, j, k] * Ibz[i, j, k]
    jXbzNP = Ijx[i, j, k] * Iby[i, j, k] - Ijy[i, j, k] * Ibx[i, j, k]

    return (jXbxP - jXbxNP) * dvx + (jXbyP - jXbyNP) * dvy + (jXbzP - jXbzNP) * dvz
