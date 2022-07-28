from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrppolv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], rho, ppol, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, rho, ppol, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','ppol', 'v']

def load():
    return FluxDrppolv()

@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, ppol, vx, vy, vz):
    
    rpNP = (rho[ip,jp,kp] + rho[i,j,k]) * ppol[i,j,k]
    rpP = (rho[ip,jp,kp] + rho[i,j,k]) * ppol[ip,jp,kp]
    
    fx = rpNP * vx[ip,jp,kp] - rpP * vx[i,j,k]
    fy = rpNP * vy[ip,jp,kp] - rpP * vy[i,j,k]
    fz = rpNP * vz[ip,jp,kp] - rpP * vz[i,j,k]
    
    return fx, fy, fz