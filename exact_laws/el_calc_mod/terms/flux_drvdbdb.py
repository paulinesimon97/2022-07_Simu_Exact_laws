from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrvdbdb(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], rho, vx, vy, vz, bx, by, bz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, rho, vx, vy, vz, bx, by, bz)

    def variables(self) -> List[str]:
        return ['rho','b','v']

def load():
    return FluxDrvdbdb()
    
@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, vx, vy, vz, bx, by, bz):
    
    dbx = bx[ip,jp,kp] - bx[i,j,k]
    dby = by[ip,jp,kp] - by[i,j,k]
    dbz = bz[ip,jp,kp] - bz[i,j,k]
    
    drvx = rho[ip,jp,kp] * vx[ip,jp,kp] - rho[i,j,k] * vx[i,j,k]
    drvy = rho[ip,jp,kp] * vy[ip,jp,kp] - rho[i,j,k] * vy[i,j,k]
    drvz = rho[ip,jp,kp] * vz[ip,jp,kp] - rho[i,j,k] * vz[i,j,k]
    
    fx = (drvx * dbx + drvy * dby + drvz * dbz) * dbx
    fy = (drvx * dbx + drvy * dby + drvz * dbz) * dby
    fz = (drvx * dbx + drvy * dby + drvz * dbz) * dbz
    
    return fx, fy, fz