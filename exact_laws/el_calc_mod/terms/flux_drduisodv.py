from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrduisodv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], rho, uiso, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, rho, uiso, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','uiso', 'v']

def load():
    return FluxDrduisodv()

@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, uiso, vx, vy, vz):
    
    dr = rho[ip,jp,kp] - rho[i,j,k]
    duiso = uiso[ip,jp,kp] - uiso[i,j,k]
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    fx = dr * duiso * dvx
    fy = dr * duiso * dvy
    fz = dr * duiso * dvz
    
    return fx, fy, fz