from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class SourcePisovdrdr(AbstractTerm):
    def __init__(self):
        pass

    def calc(
        self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, piso, dxrho, dyrho, dzrho, **kwarg
    ) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, rho, vx, vy, vz, piso, dxrho, dyrho, dzrho)

    def variables(self) -> List[str]:
        return ["rho", "gradrho", "v", "piso"]


def load():
    return SourcePisovdrdr()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, vx, vy, vz, piso, dxrho, dyrho, dzrho):

    vNPgradrhoP = vx[i, j, k] * dxrho[ip, jp, kp] + vy[i, j, k] * dyrho[ip, jp, kp] + vz[i, j, k] * dzrho[ip, jp, kp]
    vPgradrhoNP = vx[ip, jp, kp] * dxrho[i, j, k] + vy[ip, jp, kp] * dyrho[i, j, k] + vz[ip, jp, kp] * dzrho[i, j, k]
    drho = rho[ip, jp, kp] - rho[i, j, k]

    return drho * piso[i, j, k] * vNPgradrhoP / rho[ip, jp, kp] - drho * piso[ip, jp, kp] * vPgradrhoNP / rho[i, j, k]
