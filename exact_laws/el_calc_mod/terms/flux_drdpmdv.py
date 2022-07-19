from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrdpmdv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], rho, pm, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, rho, pm, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','pm', 'v']

def load():
    return FluxDrdpmdv()

@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, pm, vx, vy, vz):
    
    dr = rho[ip,jp,kp] - rho[i,j,k]
    dpm = pm[ip,jp,kp] - pm[i,j,k]
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    fx = dr * dpm * dvx
    fy = dr * dpm * dvy
    fz = dr * dpm * dvz
    
    return fx, fy, fz