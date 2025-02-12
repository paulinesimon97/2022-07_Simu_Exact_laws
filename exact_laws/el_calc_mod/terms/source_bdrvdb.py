from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba

class SourceBdrvdb(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("rho'", "rho",
                      "vx'", "vy'", "vz'",
                      "vx", "vy", "vz",
                      "bx", "by", "bz",
                      "divb'"
                )
        self.fct = sp.lambdify(
            sp.symbols(quantities),
            self.expr,
            "numpy",
        )
        
    def set_sympy_expr(self):
        rhoP, rhoNP = sp.symbols(("rho'", "rho"))
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))
        divbP = sp.symbols(("divb'"))

        drvx = rhoP * vxP - rhoNP * vxNP
        drvy = rhoP * vyP - rhoNP * vyNP
        drvz = rhoP * vzP - rhoNP * vzNP
        
        self.expr = (bxNP * drvx + byNP * drvy + bzNP * drvz) * divbP

    def calc(self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, bx, by, bz, divb, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, bx, by, bz, divb)
    
    def calc_fourier(self, rho, vx, vy, vz, bx, by, bz, divb, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, bx, by, bz, divb)

    def variables(self) -> List[str]:
        return ["rho", "v", "b", "divb"]

    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr

def load():
    return SourceBdrvdb()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceBdrvdb().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, vx, vy, vz, bx, by, bz, divb,f=njit(SourceBdrvdb().fct)):

    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    divbP, divbNP = divb[ip, jp, kp], divb[i, j, k]

    return (f(rhoP, rhoNP, vxP, vyP, vzP, vxNP, vyNP, vzNP, bxNP, byNP, bzNP, divbP) 
            + f(rhoNP, rhoP, vxNP, vyNP, vzNP, vxP, vyP, vzP, bxP, byP, bzP, divbNP))
    
def calc_with_fourier(rho, vx, vy, vz, bx, by, bz, divb):
    #A*dB*C'-A'*dB*C = A*(B'-B)*C'-A'*(B'-B)*C = A*B'*C' + A'B*C - A'*B'*C - A*B*C'
    fbx = ft.fft(bx)
    fby = ft.fft(by)
    fbz = ft.fft(bz)
    frvdx = ft.fft(rho*vx*divb)
    frvdy = ft.fft(rho*vy*divb)
    frvdz = ft.fft(rho*vz*divb)
    fd = ft.fft(divb)
    frvb = ft.fft(rho*vx*bx+rho*vy*by+rho*vz*bz)
    
    output = ft.ifft(frvdx*np.conj(fbx)+frvdy*np.conj(fby)+frvdz*np.conj(fbz)
                     +np.conj(frvdx)*fbx+np.conj(frvdy)*fby+np.conj(frvdz)*fbz
                     -frvb*np.conj(fd)-np.conj(frvb)*fd)
    return output/np.size(output)