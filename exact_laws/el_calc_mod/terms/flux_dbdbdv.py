from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDbdbdv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], vx, vy, vz, Ibx, Iby, Ibz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz, Ibx, Iby, Ibz)

    def variables(self) -> List[str]:
        return ['Ib','v']

def load():
    return FluxDbdbdv()

@njit
def calc_in_point(i, j, k, ip, jp, kp, vx, vy, vz, Ibx, Iby, Ibz):
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    dIbx = Ibx[ip,jp,kp] - Ibx[i,j,k]
    dIby = Iby[ip,jp,kp] - Iby[i,j,k]
    dIbz = Ibz[ip,jp,kp] - Ibz[i,j,k]
    
    fx = (dIbx * dIbx + dIby * dIby + dIbz * dIbz) * dvx
    fy = (dIbx * dIbx + dIby * dIby + dIbz * dIbz) * dvy
    fz = (dIbx * dIbx + dIby * dIby + dIbz * dIbz) * dvz
    
    return fx, fy, fz
    
    