from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrdpperpdv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], rho, pperp, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, rho, pperp, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','pgyr', 'v']

def load():
    return FluxDrdpperpdv()

@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, pperp, vx, vy, vz):
    
    dr = rho[ip,jp,kp] - rho[i,j,k]
    dpperp = pperp[ip,jp,kp] - pperp[i,j,k]
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    fx = dr * dpperp * dvx
    fy = dr * dpperp * dvy
    fz = dr * dpperp * dvz
    
    return fx, fy, fz