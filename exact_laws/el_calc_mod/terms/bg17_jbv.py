from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba

class Bg17Jbv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], vx, vy, vz, Ibx, Iby, Ibz, Ijx, Ijy, Ijz, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz, Ibx, Iby, Ibz, Ijx, Ijy, Ijz)

    def variables(self) -> List[str]:
        return ['Ij','Ib','v']

def load():
    return Bg17Jbv()

@njit
def calc_in_point(i, j, k, ip, jp, kp, vx, vy, vz, Ibx, Iby, Ibz, Ijx, Ijy, Ijz):
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    jXbxP = Ijy[ip,jp,kp] * Ibz[ip,jp,kp] - Ijz[ip,jp,kp] * Iby[ip,jp,kp]  
    jXbyP = Ijz[ip,jp,kp] * Ibx[ip,jp,kp] - Ijx[ip,jp,kp] * Ibz[ip,jp,kp]  
    jXbzP = Ijx[ip,jp,kp] * Iby[ip,jp,kp] - Ijy[ip,jp,kp] * Ibx[ip,jp,kp]  
    jXbxNP = Ijy[i,j,k] * Ibz[i,j,k] - Ijz[i,j,k] * Iby[i,j,k]  
    jXbyNP = Ijz[i,j,k] * Ibx[i,j,k] - Ijx[i,j,k] * Ibz[i,j,k]  
    jXbzNP = Ijx[i,j,k] * Iby[i,j,k] - Ijy[i,j,k] * Ibx[i,j,k]  
    
    return (jXbxP - jXbxNP) * dvx + (jXbyP - jXbyNP) * dvy + (jXbzP - jXbzNP) * dvz
        