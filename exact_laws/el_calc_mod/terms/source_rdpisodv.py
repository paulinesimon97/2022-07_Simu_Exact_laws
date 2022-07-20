from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRdpisodv(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], rho, piso, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, rho, piso, divv)

    def variables(self) -> List[str]:
        return ["rho", "piso", "divv"]


def load():
    return SourceRdpisodv()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, piso, divv):

    return (
        rho[i, j, k] * (piso[ip, jp, kp] - piso[i, j, k]) * divv[ip, jp, kp]
        - rho[ip, jp, kp] * (piso[ip, jp, kp] - piso[i, j, k]) * divv[i, j, k]
    )
