from typing import List
from numba import njit
import numpy as np
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceTest(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], vx, vy, vz, **kwarg) -> (float):
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['v']


def load():
    return SourceTest()


@njit
def calc_in_point(i, j, k, ip, jp, kp, vx, vy, vz):
    return vx[i, j, k] * vy[ip, jp, kp] * vz[i, j, k]
