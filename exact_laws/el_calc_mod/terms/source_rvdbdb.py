from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRvdbdb(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, bx, by, bz, divb, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, rho, vx, vy, vz, bx, by, bz, divb)

    def variables(self) -> List[str]:
        return ["rho", "v", "b", "divb"]


def load():
    return SourceRvdbdb()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, vx, vy, vz, bx, by, bz, divb):

    dbx = bx[ip, jp, kp] - bx[i, j, k]
    dby = by[ip, jp, kp] - by[i, j, k]
    dbz = bz[ip, jp, kp] - bz[i, j, k]

    return (
        rho[i, j, k] * (vx[i, j, k] * dbx + vy[i, j, k] * dby + vz[i, j, k] * dbz) * divb[ip, jp, kp]
        - rho[ip, jp, kp] * (vx[ip, jp, kp] * dbx + vy[ip, jp, kp] * dby + vz[ip, jp, kp] * dbz) * divb[i, j, k]
    )
