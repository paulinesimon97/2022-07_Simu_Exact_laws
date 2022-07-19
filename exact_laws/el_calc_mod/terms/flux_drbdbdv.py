from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrbdbdv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], rho, vx, vy, vz, bx, by, bz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, rho, vx, vy, vz, bx, by, bz)

    def variables(self) -> List[str]:
        return ['rho','b','v']

def load():
    return FluxDrbdbdv()

@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, vx, vy, vz, bx, by, bz):
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    dbx = bx[ip,jp,kp] - bx[i,j,k]
    dby = by[ip,jp,kp] - by[i,j,k]
    dbz = bz[ip,jp,kp] - bz[i,j,k]
    
    drbx = rho[ip,jp,kp] * bx[ip,jp,kp] - rho[i,j,k] * bx[i,j,k]
    drby = rho[ip,jp,kp] * by[ip,jp,kp] - rho[i,j,k] * by[i,j,k]
    drbz = rho[ip,jp,kp] * bz[ip,jp,kp] - rho[i,j,k] * bz[i,j,k]
    
    fx = (drbx * dbx + drby * dby + drbz * dbz) * dvx
    fy = (drbx * dbx + drby * dby + drbz * dbz) * dvy
    fz = (drbx * dbx + drby * dby + drbz * dbz) * dvz
    
    return fx, fy, fz
    
    