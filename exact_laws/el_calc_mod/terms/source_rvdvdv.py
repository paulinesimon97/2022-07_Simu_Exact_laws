from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRvdvdv(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, rho, vx, vy, vz, divv)

    def variables(self) -> List[str]:
        return ["rho", "v", "divv"]


def load():
    return SourceRvdvdv()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, vx, vy, vz, divv):

    dvx = vx[ip, jp, kp] - vx[i, j, k]
    dvy = vy[ip, jp, kp] - vy[i, j, k]
    dvz = vz[ip, jp, kp] - vz[i, j, k]

    return (
        rho[i, j, k] * (vx[i, j, k] * dvx + vy[i, j, k] * dvy + vz[i, j, k] * dvz) * divv[ip, jp, kp]
        - rho[ip, jp, kp] * (vx[ip, jp, kp] * dvx + vy[ip, jp, kp] * dvy + vz[ip, jp, kp] * dvz) * divv[i, j, k]
    )
