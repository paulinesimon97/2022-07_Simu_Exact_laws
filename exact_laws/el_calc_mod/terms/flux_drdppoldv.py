from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrdppoldv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], rho, ppol, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, rho, ppol, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','ppol', 'v']

def load():
    return FluxDrdppoldv()

@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, ppol, vx, vy, vz):
    
    dr = rho[ip,jp,kp] - rho[i,j,k]
    dppol = ppol[ip,jp,kp] - ppol[i,j,k]
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    fx = dr * dppol * dvx
    fy = dr * dppol * dvy
    fz = dr * dppol * dvz
    
    return fx, fy, fz