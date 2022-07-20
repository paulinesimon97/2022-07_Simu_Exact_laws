from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRbdvdb(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, bx, by, bz, divb, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, rho, vx, vy, vz, bx, by, bz, divb)

    def variables(self) -> List[str]:
        return ["rho", "v", "b", "divb"]


def load():
    return SourceRbdvdb()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, vx, vy, vz, bx, by, bz, divb):

    dvx = vx[ip, jp, kp] - vx[i, j, k]
    dvy = vy[ip, jp, kp] - vy[i, j, k]
    dvz = vz[ip, jp, kp] - vz[i, j, k]

    return (
        rho[i, j, k] * (bx[i, j, k] * dvx + by[i, j, k] * dvy + bz[i, j, k] * dvz) * divb[ip, jp, kp]
        - rho[ip, jp, kp] * (bx[ip, jp, kp] * dvx + by[ip, jp, kp] * dvy + bz[ip, jp, kp] * dvz) * divb[i, j, k]
    )
