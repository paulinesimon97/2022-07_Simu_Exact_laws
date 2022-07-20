from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRdpperpdv(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], rho, pperp, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, rho, pperp, divv)

    def variables(self) -> List[str]:
        return ["rho", "pgyr", "divv"]


def load():
    return SourceRdpperpdv()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, pperp, divv):

    return (
        rho[i, j, k] * (pperp[ip, jp, kp] - pperp[i, j, k]) * divv[ip, jp, kp]
        - rho[ip, jp, kp] * (pperp[ip, jp, kp] - pperp[i, j, k]) * divv[i, j, k]
    )
