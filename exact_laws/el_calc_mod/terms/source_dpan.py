from typing import List
from numba import njit
import sympy as sp
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceDpan(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("pperp'", "ppar'", "pm'"
                      "bx'", "by'", "bz'",
                      "dxvx'", "dyvx'", "dzvx'",
                      "dxvy'", "dyvy'", "dzvy'",
                      "dxvz'", "dyvz'", "dzvz'",
                      "dxvx", "dyvx", "dzvx",
                      "dxvy", "dyvy", "dzvy",
                      "dxvz", "dyvz", "dzvz"
                )
        self.fct = sp.lambdify(
            sp.symbols(quantities),
            self.expr,
            "numpy",
        )
        
    def set_sympy_expr(self):
        IpperpP, IpparP, IpmP = sp.symbols(("pperp'", "ppar'", "pm'"))
        IbxP, IbyP, IbzP = sp.symbols(("vx'", "vy'", "vz'"))
        dxvxP, dyvxP, dzvxP = sp.symbols(("dxvx'", "dyvx'", "dzvx'"))
        dxvyP, dyvyP, dzvyP = sp.symbols(("dxvy'", "dyvy'", "dzvy'"))
        dxvzP, dyvzP, dzvzP = sp.symbols(("dxvz'", "dyvz'", "dzvz'"))
        dxvxNP, dyvxNP, dzvxNP = sp.symbols(("dxvx", "dyvx", "dzvx"))
        dxvyNP, dyvyNP, dzvyNP = sp.symbols(("dxvy", "dyvy", "dzvy"))
        dxvzNP, dyvzNP, dzvzNP = sp.symbols(("dxvz", "dyvz", "dzvz"))

        ddxvx = dxvxP - dxvxNP
        ddyvx = dyvxP - dyvxNP
        ddzvx = dzvxP - dzvxNP
        ddxvy = dxvyP - dxvyNP
        ddyvy = dyvyP - dyvyNP
        ddzvy = dzvyP - dzvyNP
        ddxvz = dxvzP - dxvzNP
        ddyvz = dyvzP - dyvzNP
        ddzvz = dzvzP - dzvzNP

        pressP = (IpparP - IpperpP) / IpmP

        self.expr = pressP * (IbxP * (IbxP * ddxvx + IbyP * ddxvy + IbzP * ddxvz) + IbyP * (
                IbxP * ddyvx + IbyP * ddyvy + IbzP * ddyvz) + IbzP * (
                                IbxP * ddzvx + IbyP * ddzvy + IbzP * ddzvz)) 
        
    def calc(self, vector: List[int], cube_size: List[int],
             Ipperp, Ippar, Ipm,
             Ibx, Iby, Ibz,
             dxvx, dyvx, dzvx,
             dxvy, dyvy, dzvy,
             dxvz, dyvz, dzvz,
             **kwarg) -> (float):
        #return calc_source_with_numba(calc_in_point, *vector, *cube_size,
                                    #   Ipperp, Ippar, Ipm,
                                    #   Ibx, Iby, Ibz,
                                    #   dxvx, dyvx, dzvx,
                                    #   dxvy, dyvy, dzvy,
                                    #   dxvz, dyvz, dzvz)
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size,
                                      Ipperp, Ippar, Ipm,
                                      Ibx, Iby, Ibz,
                                      dxvx, dyvx, dzvx,
                                      dxvy, dyvy, dzvy,
                                      dxvz, dyvz, dzvz)

    def variables(self) -> List[str]:
        return ["Ipgyr", "Ipm", "gradv", "Ib"]


def load():
    return SourceDpan()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceDpan().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             Ipperp, Ippar, Ipm,
                             Ibx, Iby, Ibz,
                             dxvx, dyvx, dzvx,
                             dxvy, dyvy, dzvy,
                             dxvz, dyvz, dzvz,  
                             f=njit(SourceDpan().fct)):
    IpperpP, IpparP, IpmP = Ipperp[ip, jp, kp], Ippar[ip, jp, kp], Ipm[ip, jp, kp]
    IpperpNP, IpparNP, IpmNP = Ipperp[i, j, k], Ippar[i, j, k], Ipm[i, j, k]
    IbxP, IbyP, IbzP = Ibx[ip, jp, kp], Iby[ip, jp, kp], Ibz[ip, jp, kp]
    IbxNP, IbyNP, IbzNP = Ibx[i, j, k], Iby[i, j, k], Ibz[i, j, k]
    dxvxP, dyvxP, dzvxP = dxvx[ip, jp, kp], dyvx[ip, jp, kp], dzvx[ip, jp, kp]
    dxvyP, dyvyP, dzvyP = dxvy[ip, jp, kp], dyvy[ip, jp, kp], dzvy[ip, jp, kp]
    dxvzP, dyvzP, dzvzP = dxvz[ip, jp, kp], dyvz[ip, jp, kp], dzvz[ip, jp, kp]
    dxvxNP, dyvxNP, dzvxNP = dxvx[i, j, k], dyvx[i, j, k], dzvx[i, j, k]
    dxvyNP, dyvyNP, dzvyNP = dxvy[i, j, k], dyvy[i, j, k], dzvy[i, j, k]
    dxvzNP, dyvzNP, dzvzNP = dxvz[i, j, k], dyvz[i, j, k], dzvz[i, j, k]
    
    return (f(IpperpP, IpparP, IpmP, IbxP, IbyP, IbzP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP)
           + f(IpperpNP, IpparNP, IpmNP, IbxNP, IbyNP, IbzNP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP))

@njit
def calc_in_point(i, j, k, ip, jp, kp,
                  Ipperp, Ippar, Ipm,
                  Ibx, Iby, Ibz,
                  dxvx, dyvx, dzvx,
                  dxvy, dyvy, dzvy,
                  dxvz, dyvz, dzvz):
    ddxvx = dxvx[ip, jp, kp] - dxvx[i, j, k]
    ddyvx = dyvx[ip, jp, kp] - dyvx[i, j, k]
    ddzvx = dzvx[ip, jp, kp] - dzvx[i, j, k]
    ddxvy = dxvy[ip, jp, kp] - dxvy[i, j, k]
    ddyvy = dyvy[ip, jp, kp] - dyvy[i, j, k]
    ddzvy = dzvy[ip, jp, kp] - dzvy[i, j, k]
    ddxvz = dxvz[ip, jp, kp] - dxvz[i, j, k]
    ddyvz = dyvz[ip, jp, kp] - dyvz[i, j, k]
    ddzvz = dzvz[ip, jp, kp] - dzvz[i, j, k]

    pressP = (Ippar[ip, jp, kp] - Ipperp[ip, jp, kp]) / Ipm[ip, jp, kp]
    press = (Ippar[i, j, k] - Ipperp[i, j, k]) / Ipm[i, j, k]

    IbxP = Ibx[ip, jp, kp]
    IbyP = Iby[ip, jp, kp]
    IbzP = Ibz[ip, jp, kp]

    IbxNP = Ibx[i, j, k]
    IbyNP = Iby[i, j, k]
    IbzNP = Ibz[i, j, k]

    return pressP * (IbxP * (IbxP * ddxvx + IbyP * ddxvy + IbzP * ddxvz) + IbyP * (
            IbxP * ddyvx + IbyP * ddyvy + IbzP * ddyvz) + IbzP * (
                             IbxP * ddzvx + IbyP * ddzvy + IbzP * ddzvz)) - press * (
                   IbxNP * (IbxNP * ddxvx + IbyNP * ddxvy + IbzNP * ddxvz) + IbyNP * (
                   IbxNP * ddyvx + IbyNP * ddyvy + IbzNP * ddyvz) + IbzNP * (
                               IbxNP * ddzvx + IbyNP * ddzvy + IbzNP * ddzvz))
