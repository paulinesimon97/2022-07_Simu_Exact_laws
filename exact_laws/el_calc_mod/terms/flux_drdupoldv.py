from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrdupoldv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], rho, upol, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, rho, upol, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','upol', 'v']

def load():
    return FluxDrdupoldv()

@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, upol, vx, vy, vz):
    
    dr = rho[ip,jp,kp] - rho[i,j,k]
    dupol = upol[ip,jp,kp] - upol[i,j,k]
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    fx = dr * dupol * dvx
    fy = dr * dupol * dvy
    fz = dr * dupol * dvz
    
    return fx, fy, fz