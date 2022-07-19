from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class FluxBdrbdv(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], rho, bx, by, bz, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, rho, bx, by, bz, divv)

    def variables(self) -> List[str]:
        return ["rho", "b", "divv"]


def load():
    return FluxBdrbdv()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, bx, by, bz, divv):

    drbx = rho[ip, jp, kp] * bx[ip, jp, kp] - rho[i, j, k] * bx[i, j, k]
    drby = rho[ip, jp, kp] * by[ip, jp, kp] - rho[i, j, k] * by[i, j, k]
    drbz = rho[ip, jp, kp] * bz[ip, jp, kp] - rho[i, j, k] * bz[i, j, k]

    return (
        (bx[i, j, k] * drbx + by[i, j, k] * drby + bz[i, j, k] * drbz) * divv[ip, jp, kp]
        - (bx[ip, jp, kp] * drbx + by[ip, jp, kp] * drby + bz[ip, jp, kp] * drbz) * divv[i, j, k]
    )
