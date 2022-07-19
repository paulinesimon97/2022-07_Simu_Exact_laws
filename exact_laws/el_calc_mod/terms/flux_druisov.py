from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDruisov(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], rho, uiso, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, rho, uiso, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','uiso', 'v']

def load():
    return FluxDruisov()

@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, uiso, vx, vy, vz):
    
    rPuNP = rho[ip,jp,kp] * uiso[i,j,k]
    rNPuP = rho[i,j,k] * uiso[ip,jp,kp]
    
    fx = rPuNP * vx[ip,jp,kp] - rNPuP * vx[i,j,k]
    fy = rPuNP * vy[ip,jp,kp] - rNPuP * vy[i,j,k]
    fz = rPuNP * vz[ip,jp,kp] - rNPuP * vz[i,j,k]
    
    return fx, fy, fz