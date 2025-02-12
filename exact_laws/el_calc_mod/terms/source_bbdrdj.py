from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceBbdrdj(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("rho'", "rho",
                      "bx'", "by'", "bz'",
                      "bx", "by", "bz",
                      "divj'", "divj"
                )
        self.fct = sp.lambdify(
            sp.symbols(quantities),
            self.expr,
            "numpy",
        )
        
    def set_sympy_expr(self):
        rhoP, rhoNP = sp.symbols(("rho'", "rho"))
        bxP, byP, bzP = sp.symbols(("bx'", "by'", "bz'"))
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))
        divjP, divjNP = sp.symbols(("divj'", "divj"))

        drho = rhoP - rhoNP
        psb = bxP * bxNP + byP * byNP + bzP * bzNP
        ddivj = divjP - divjNP
        
        self.expr = drho * psb * ddivj
        
    def calc(self, vector: List[int], cube_size: List[int],
             rho,
             bx, by, bz,
             divj, **kwarg) -> (float):
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size,
                                      rho, bx, by, bz, divj)
    
    def calc_fourier(self, rho, bx, by, bz, divj, **kwarg) -> List:
        return calc_with_fourier(rho, bx, by, bz, divj)

    def variables(self) -> List[str]:
        return ["rho", "b", "divj"]
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr

def load():
    return SourceBbdrdj()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceBbdrdj().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             rho, bx, by, bz, divj,  
                             f=njit(SourceBbdrdj().fct)):
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    divjP, divjNP = divj[ip, jp, kp], divj[i, j, k]
    
    return f(rhoP, rhoNP, bxP, byP, bzP, bxNP, byNP, bzNP, divjP, divjNP)

def calc_with_fourier(rho, bx, by, bz, divj): 
    #dA*B*C'*dD = BB'A'D' + BB'AD - BB'A'D - BB'AD'
    fbx = ft.fft(bx)
    fby = ft.fft(by)
    fbz = ft.fft(bz)
    frbdx = ft.fft(rho*bx*divj)
    frbdy = ft.fft(rho*by*divj)
    frbdz = ft.fft(rho*bz*divj)
    
    output = ft.ifft(frbdx*np.conj(fbx)+frbdy*np.conj(fby)+frbdz*np.conj(fbz)
                     +np.conj(frbdx)*fbx+np.conj(frbdy)*fby+np.conj(frbdz)*fbz)

    del(fbx,fby,fbz,frbdx,frbdy,frbdz)
    
    frbx = ft.fft(rho*bx)
    frby = ft.fft(rho*by)
    frbz = ft.fft(rho*bz)
    fbdx = ft.fft(bx*divj)
    fbdy = ft.fft(by*divj)
    fbdz = ft.fft(bz*divj)
    
    output -= ft.ifft(fbdx*np.conj(frbx)+fbdy*np.conj(frby)+fbdz*np.conj(frbz)
                     +np.conj(fbdx)*frbx+np.conj(fbdy)*frby+np.conj(fbdz)*frbz)
    return output/np.size(output)


