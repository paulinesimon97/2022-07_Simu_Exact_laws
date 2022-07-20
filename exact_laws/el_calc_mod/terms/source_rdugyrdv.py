from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRdugyrdv(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], rho, ugyr, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, rho, ugyr, divv)

    def variables(self) -> List[str]:
        return ["rho", "ugyr", "divv"]


def load():
    return SourceRdugyrdv()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, ugyr, divv):

    return (
        rho[i, j, k] * (ugyr[ip, jp, kp] - ugyr[i, j, k]) * divv[ip, jp, kp]
        - rho[ip, jp, kp] * (ugyr[ip, jp, kp] - ugyr[i, j, k]) * divv[i, j, k]
    )
