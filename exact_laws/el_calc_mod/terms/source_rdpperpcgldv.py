from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRdpperpcgldv(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], rho, pperpcgl, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, rho, pperpcgl, divv)

    def variables(self) -> List[str]:
        return ["rho", "pcgl", "divv"]


def load():
    return SourceRdpperpcgldv()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, pperpcgl, divv):

    return (
        rho[i, j, k] * (pperpcgl[ip, jp, kp] - pperpcgl[i, j, k]) * divv[ip, jp, kp]
        - rho[ip, jp, kp] * (pperpcgl[ip, jp, kp] - pperpcgl[i, j, k]) * divv[i, j, k]
    )
