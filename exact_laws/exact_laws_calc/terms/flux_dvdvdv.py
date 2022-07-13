from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDvdvdv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc_old(self, values) -> (float or List[float]):
        return self.flux(("vx", "vy", "vz"), ("vx", "vy", "vz"), ("vx", "vy", "vz"), datadic=values)
    
    def calc(self, vector:List[int], cube_size:List[int], vx, vy, vz, **kwarg) -> List[float]:
        #return calc_source_with_numba(np.array(vector), np.array(cube_size), f2, vx)
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['v']

def load():
    return FluxDvdvdv()

@njit
def calc_in_point(i, j, k, ip, jp, kp, vx, vy, vz):
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    fx = (dvx * dvx + dvy * dvy + dvz * dvz) * dvx
    fy = (dvx * dvx + dvy * dvy + dvz * dvz) * dvy
    fz = (dvx * dvx + dvy * dvy + dvz * dvz) * dvz
    return fx, fy, fz