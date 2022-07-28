from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRdupoldv(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], rho, upol, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, rho, upol, divv)

    def variables(self) -> List[str]:
        return ["rho", "upol", "divv"]


def load():
    return SourceRdupoldv()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, upol, divv):

    return (
        rho[i, j, k] * (upol[ip, jp, kp] - upol[i, j, k]) * divv[ip, jp, kp]
        - rho[ip, jp, kp] * (upol[ip, jp, kp] - upol[i, j, k]) * divv[i, j, k]
    )
