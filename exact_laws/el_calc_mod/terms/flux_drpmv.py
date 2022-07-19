from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrpmv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], rho, pm, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, rho, pm, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','pm', 'v']

def load():
    return FluxDrpmv()

@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, pm, vx, vy, vz):
    
    rpmNP = (rho[ip,jp,kp] + rho[i,j,k]) * pm[i,j,k]
    rpmP = (rho[ip,jp,kp] + rho[i,j,k]) * pm[ip,jp,kp]
    
    fx = rpmNP * vx[ip,jp,kp] - rpmP * vx[i,j,k]
    fy = rpmNP * vy[ip,jp,kp] - rpmP * vy[i,j,k]
    fz = rpmNP * vz[ip,jp,kp] - rpmP * vz[i,j,k]
    
    return fx, fy, fz