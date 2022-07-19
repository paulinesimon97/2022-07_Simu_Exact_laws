from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class FluxDrbbdv(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], rho, bx, by, bz, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, rho, bx, by, bz, divv)

    def variables(self) -> List[str]:
        return ["rho", "b", "divv"]


def load():
    return FluxDrbbdv()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, bx, by, bz, divv):
    
    bPbNP = bx[ip, jp, kp] * bx[i, j, k] + by[ip, jp, kp] * by[i, j, k] + bz[ip, jp, kp] * bz[i, j, k]

    return  (rho[ip, jp, kp] + rho[i, j, k]) * bPbNP * (divv[ip, jp, kp] + divv[i, j, k])
