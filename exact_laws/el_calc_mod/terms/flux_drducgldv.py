from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrducgldv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], rho, ucgl, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, rho, ucgl, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','ucgl', 'v']

def load():
    return FluxDrducgldv()

@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, ucgl, vx, vy, vz):
    
    dr = rho[ip,jp,kp] - rho[i,j,k]
    ducgl = ucgl[ip,jp,kp] - ucgl[i,j,k]
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    fx = dr * ducgl * dvx
    fy = dr * ducgl * dvy
    fz = dr * ducgl * dvz
    
    return fx, fy, fz