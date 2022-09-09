from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrdpperpcgldv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], rho, pperpcgl, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, rho, pperpcgl, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','pcgl', 'v']

def load():
    return FluxDrdpperpcgldv()

@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, pperpcgl, vx, vy, vz):
    
    dr = rho[ip,jp,kp] - rho[i,j,k]
    dpperp = pperpcgl[ip,jp,kp] - pperpcgl[i,j,k]
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    fx = dr * dpperp * dvx
    fy = dr * dpperp * dvy
    fz = dr * dpperp * dvz
    
    return fx, fy, fz