from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrdpisodv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], rho, piso, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, rho, piso, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','piso', 'v']

def load():
    return FluxDrdpisodv()

@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, piso, vx, vy, vz):
    
    dr = rho[ip,jp,kp] - rho[i,j,k]
    dpiso = piso[ip,jp,kp] - piso[i,j,k]
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    fx = dr * dpiso * dvx
    fy = dr * dpiso * dvy
    fz = dr * dpiso * dvz
    
    return fx, fy, fz