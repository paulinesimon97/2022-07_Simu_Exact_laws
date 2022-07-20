from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrdugyrdv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], rho, ugyr, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, rho, ugyr, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','ugyr', 'v']

def load():
    return FluxDrdugyrdv()

@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, ugyr, vx, vy, vz):
    
    dr = rho[ip,jp,kp] - rho[i,j,k]
    dugyr = ugyr[ip,jp,kp] - ugyr[i,j,k]
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    fx = dr * dugyr * dvx
    fy = dr * dugyr * dvy
    fz = dr * dugyr * dvz
    
    return fx, fy, fz