from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba

class Bg17Vwv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], vx, vy, vz, wx, wy, wz, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz, wx, wy, wz)

    def variables(self) -> List[str]:
        return ['w','v']

def load():
    return Bg17Vwv()
    
@njit
def calc_in_point(i, j, k, ip, jp, kp, vx, vy, vz, wx, wy, wz):
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    vXwxP = vy[ip,jp,kp] * wz[ip,jp,kp] - vz[ip,jp,kp] * wy[ip,jp,kp]  
    vXwyP = vz[ip,jp,kp] * wx[ip,jp,kp] - vx[ip,jp,kp] * wz[ip,jp,kp]  
    vXwzP = vx[ip,jp,kp] * wy[ip,jp,kp] - vy[ip,jp,kp] * wx[ip,jp,kp]  
    vXwxNP = vy[i,j,k] * wz[i,j,k] - vz[i,j,k] * wy[i,j,k]  
    vXwyNP = vz[i,j,k] * wx[i,j,k] - vx[i,j,k] * wz[i,j,k]  
    vXwzNP = vx[i,j,k] * wy[i,j,k] - vy[i,j,k] * wx[i,j,k]  
    
    return (vXwxP - vXwxNP) * dvx + (vXwyP - vXwyNP) * dvy + (vXwzP - vXwzNP) * dvz