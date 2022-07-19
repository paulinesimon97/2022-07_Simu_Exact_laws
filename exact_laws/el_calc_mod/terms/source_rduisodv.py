from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class FluxRduisodv(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], rho, uiso, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, rho, uiso, divv)

    def variables(self) -> List[str]:
        return ["rho", "uiso", "divv"]


def load():
    return FluxRduisodv()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, uiso, divv):

    return (
        rho[i, j, k] * (uiso[ip, jp, kp] - uiso[i, j, k]) * divv[ip, jp, kp]
        - rho[ip, jp, kp] * (uiso[ip, jp, kp] - uiso[i, j, k]) * divv[i, j, k]
    )
