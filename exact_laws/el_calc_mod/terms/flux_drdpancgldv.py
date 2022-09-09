from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrdpancgldv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], rho, pperpcgl, pparcgl, vx, vy, vz, pm, bx, by, bz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, rho, pperpcgl, pparcgl, vx, vy, vz, pm, bx, by, bz)

    def variables(self) -> List[str]:
        return ['rho','pcgl', 'pm', 'v', 'b']

def load():
    return FluxDrdpancgldv()

@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, pperpcgl, pparcgl, vx, vy, vz, pm, bx, by, bz):
    
    dr = rho[ip,jp,kp] - rho[i,j,k]
    pressP = (pparcgl[ip, jp, kp] - pperpcgl[ip, jp, kp]) / pm[ip, jp, kp]
    pressNP = (pparcgl[i, j, k] - pperpcgl[i, j, k]) / pm[i, j, k]
    
    dpxx = pressP * bx[ip,jp,kp] * bx[ip,jp,kp] - pressNP * bx[i,j,k] * bx[i,j,k]
    dpyy = pressP * by[ip,jp,kp] * by[ip,jp,kp] - pressNP * by[i,j,k] * by[i,j,k]
    dpzz = pressP * bz[ip,jp,kp] * bz[ip,jp,kp] - pressNP * bz[i,j,k] * bz[i,j,k]
    dpxy = pressP * bx[ip,jp,kp] * by[ip,jp,kp] - pressNP * bx[i,j,k] * by[i,j,k]
    dpxz = pressP * bx[ip,jp,kp] * bz[ip,jp,kp] - pressNP * bx[i,j,k] * bz[i,j,k]
    dpyz = pressP * by[ip,jp,kp] * bz[ip,jp,kp] - pressNP * by[i,j,k] * bz[i,j,k]    
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    fx = dr * (dpxx * dvx + dpxy * dvy + dpxz * dvz)
    fy = dr * (dpxy * dvx + dpyy * dvy + dpyz * dvz)
    fz = dr * (dpxz * dvx + dpyz * dvy + dpzz * dvz)
    
    return fx, fy, fz