from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRducgldv(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], rho, ucgl, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, rho, ucgl, divv)

    def variables(self) -> List[str]:
        return ["rho", "ucgl", "divv"]


def load():
    return SourceRducgldv()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, ucgl, divv):

    return (
        rho[i, j, k] * (ucgl[ip, jp, kp] - ucgl[i, j, k]) * divv[ip, jp, kp]
        - rho[ip, jp, kp] * (ucgl[ip, jp, kp] - ucgl[i, j, k]) * divv[i, j, k]
    )
