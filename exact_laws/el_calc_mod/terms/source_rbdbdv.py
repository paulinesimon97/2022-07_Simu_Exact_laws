from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class FluxRbdbdv(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], rho, bx, by, bz, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, rho, bx, by, bz, divv)

    def variables(self) -> List[str]:
        return ["rho", "b", "divv"]


def load():
    return FluxRbdbdv()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, bx, by, bz, divv):

    dbx = bx[ip, jp, kp] - bx[i, j, k]
    dby = by[ip, jp, kp] - by[i, j, k]
    dbz = bz[ip, jp, kp] - bz[i, j, k]

    return (
        rho[i, j, k] * (bx[i, j, k] * dbx + by[i, j, k] * dby + bz[i, j, k] * dbz) * divv[ip, jp, kp]
        - rho[ip, jp, kp] * (bx[ip, jp, kp] * dbx + by[ip, jp, kp] * dby + bz[ip, jp, kp] * dbz) * divv[i, j, k]
    )
