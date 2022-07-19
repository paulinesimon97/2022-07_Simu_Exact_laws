from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrvdvdv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], rho, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, rho, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','v']

def load():
    return FluxDrvdvdv()

@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, vx, vy, vz):
    
    drvx = rho[ip,jp,kp] * vx[ip,jp,kp] - rho[i,j,k] * vx[i,j,k]
    drvy = rho[ip,jp,kp] * vy[ip,jp,kp] - rho[i,j,k] * vy[i,j,k]
    drvz = rho[ip,jp,kp] * vz[ip,jp,kp] - rho[i,j,k] * vz[i,j,k]
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    fx = (drvx * dvx + drvy * dvy + drvz * dvz) * dvx
    fy = (drvx * dvx + drvy * dvy + drvz * dvz) * dvy
    fz = (drvx * dvx + drvy * dvy + drvz * dvz) * dvz
    
    return fx, fy, fz