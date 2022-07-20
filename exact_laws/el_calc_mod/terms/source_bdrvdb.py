from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceBdrvdb(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, bx, by, bz, divb, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, rho, vx, vy, vz, bx, by, bz, divb)

    def variables(self) -> List[str]:
        return ["rho", "v", "b", "divb"]


def load():
    return SourceBdrvdb()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, vx, vy, vz, bx, by, bz, divb):

    drvx = rho[ip,jp,kp] * vx[ip,jp,kp] - rho[i,j,k] * vx[i,j,k]
    drvy = rho[ip,jp,kp] * vy[ip,jp,kp] - rho[i,j,k] * vy[i,j,k]
    drvz = rho[ip,jp,kp] * vz[ip,jp,kp] - rho[i,j,k] * vz[i,j,k]

    return (
        (bx[i, j, k] * drvx + by[i, j, k] * drvy + bz[i, j, k] * drvz) * divb[ip, jp, kp]
        - (bx[ip, jp, kp] * drvx + by[ip, jp, kp] * drvy + bz[ip, jp, kp] * drvz) * divb[i, j, k]
    )
