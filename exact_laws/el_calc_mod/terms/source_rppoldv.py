from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRppoldv(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], rho, ppol, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, rho, ppol, divv)

    def variables(self) -> List[str]:
        return ["rho", "ppol", "divv"]


def load():
    return SourceRppoldv()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, ppol, divv):

    return (
        rho[i, j, k] * ppol[ip, jp, kp] * divv[ip, jp, kp]
        + rho[ip, jp, kp] * ppol[i, j, k] * divv[i, j, k]
    )
