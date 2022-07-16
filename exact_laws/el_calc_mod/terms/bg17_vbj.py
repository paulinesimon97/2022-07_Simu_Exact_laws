from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba

class Bg17Vbj(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, vector:List[int], cube_size:List[int], vx, vy, vz, Ibx, Iby, Ibz, Ijx, Ijy, Ijz, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz, Ibx, Iby, Ibz, Ijx, Ijy, Ijz)

    def variables(self) -> List[str]:
        return ['Ij','Ib','v']

def load():
    return Bg17Vbj()
    
@njit
def calc_in_point(i, j, k, ip, jp, kp, vx, vy, vz, Ibx, Iby, Ibz, Ijx, Ijy, Ijz):
    
    djx = Ijx[ip,jp,kp] - Ijx[i,j,k]
    djy = Ijy[ip,jp,kp] - Ijy[i,j,k]
    djz = Ijz[ip,jp,kp] - Ijz[i,j,k]
    
    vXbxP = vy[ip,jp,kp] * Ibz[ip,jp,kp] - vz[ip,jp,kp] * Iby[ip,jp,kp]  
    vXbyP = vz[ip,jp,kp] * Ibx[ip,jp,kp] - vx[ip,jp,kp] * Ibz[ip,jp,kp]  
    vXbzP = vx[ip,jp,kp] * Iby[ip,jp,kp] - vy[ip,jp,kp] * Ibx[ip,jp,kp]  
    vXbxNP = vy[i,j,k] * Ibz[i,j,k] - vz[i,j,k] * Iby[i,j,k]  
    vXbyNP = vz[i,j,k] * Ibx[i,j,k] - vx[i,j,k] * Ibz[i,j,k]  
    vXbzNP = vx[i,j,k] * Iby[i,j,k] - vy[i,j,k] * Ibx[i,j,k]  
    
    return (vXbxP - vXbxNP) * djx + (vXbyP - vXbyNP) * djy + (vXbzP - vXbzNP) * djz