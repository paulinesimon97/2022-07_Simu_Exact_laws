from typing import List
from numba import njit
import numpy as np
from .abstract_term import AbstractTerm, calc_flux_with_numba, calc_flux_with_numba_test1, calc_flux_with_numba_test2


class FluxTest(AbstractTerm):
    def __init__(self):
        pass
        
    def calc1(self, vector:List[int], cube_size:List[int], vx, vy, vz, **kwarg) -> List[float]:
        #return calc_source_with_numba(np.array(vector), np.array(cube_size), f2, vx)
        return calc_flux_with_numba_test1(calc_x_in_point, calc_y_in_point, calc_z_in_point, *vector, *cube_size, vx, vy, vz)
    
    def calc2(self, vector:List[int], cube_size:List[int], vx, vy, vz, **kwarg) -> List[float]:
        #return calc_source_with_numba(np.array(vector), np.array(cube_size), f2, vx)
        return calc_flux_with_numba_test2(calc_in_point, *vector, *cube_size, vx, vy, vz)
    
    def calc(self, vector:List[int], cube_size:List[int], vx, vy, vz, **kwarg) -> List[float]:
        #return calc_source_with_numba(np.array(vector), np.array(cube_size), f2, vx)
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['v']

def load():
    return FluxTest()

@njit
def calc_x_in_point(i, j, k, ip, jp, kp, vx, vy, vz):
    return vx[i,j,k] * vx[ip,jp,kp] 

@njit
def calc_y_in_point(i, j, k, ip, jp, kp, vx, vy, vz):
    return vy[i,j,k] * vy[ip,jp,kp] 

@njit
def calc_z_in_point(i, j, k, ip, jp, kp, vx, vy, vz):
    return vz[i,j,k] * vz[ip,jp,kp] 

@njit 
def calc_in_point(i, j, k, ip, jp, kp, vx, vy, vz):
    f1 = vx[i,j,k] * vx[ip,jp,kp] 
    f2 = vy[i,j,k] * vy[ip,jp,kp] 
    f3 = vz[i,j,k] * vz[ip,jp,kp] 
    return f1, f2, f3
    
 
    