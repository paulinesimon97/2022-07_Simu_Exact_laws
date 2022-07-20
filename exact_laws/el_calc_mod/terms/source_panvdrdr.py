from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class SourcePanvdrdr(AbstractTerm):
    def __init__(self):
        pass

    def calc(
        self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, pperp, ppar, pm, bx, by, bz, dxrho, dyrho, dzrho, **kwarg
    ) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, rho, vx, vy, vz, pperp, ppar, pm, bx, by, bz, dxrho, dyrho, dzrho)

    def variables(self) -> List[str]:
        return ["rho", "gradrho", "v", "pgyr", "pm", "b"]


def load():
    return SourcePanvdrdr()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, vx, vy, vz, pperp, ppar, pm, bx, by, bz, dxrho, dyrho, dzrho):

    pressP = (ppar[ip, jp, kp] - pperp[ip, jp, kp]) / pm[ip, jp, kp]
    pressNP = (ppar[i, j, k] - pperp[i, j, k]) / pm[i, j, k]

    vpNPgradrhoP = (
        vx[i, j, k] * bx[i, j, k] * bx[i, j, k] * dxrho[ip, jp, kp]
        + vy[i, j, k] * bx[i, j, k] * by[i, j, k] * dxrho[ip, jp, kp]
        + vz[i, j, k] * bx[i, j, k] * bz[i, j, k] * dxrho[ip, jp, kp]
        + vx[i, j, k] * bx[i, j, k] * by[i, j, k] * dyrho[ip, jp, kp]
        + vy[i, j, k] * by[i, j, k] * by[i, j, k] * dyrho[ip, jp, kp]
        + vz[i, j, k] * by[i, j, k] * bz[i, j, k] * dyrho[ip, jp, kp]
        + vx[i, j, k] * bx[i, j, k] * bz[i, j, k] * dzrho[ip, jp, kp]
        + vy[i, j, k] * by[i, j, k] * bz[i, j, k] * dzrho[ip, jp, kp]
        + vz[i, j, k] * bz[i, j, k] * bz[i, j, k] * dzrho[ip, jp, kp]
    )
    vpPgradrhoNP = (
        vx[ip, jp, kp] * bx[ip, jp, kp] * bx[ip, jp, kp] * dxrho[i, j, k]
        + vy[ip, jp, kp] * bx[ip, jp, kp] * by[ip, jp, kp] * dxrho[i, j, k]
        + vz[ip, jp, kp] * bx[ip, jp, kp] * bz[ip, jp, kp] * dxrho[i, j, k]
        + vx[ip, jp, kp] * bx[ip, jp, kp] * by[ip, jp, kp] * dyrho[i, j, k]
        + vy[ip, jp, kp] * by[ip, jp, kp] * by[ip, jp, kp] * dyrho[i, j, k]
        + vz[ip, jp, kp] * by[ip, jp, kp] * bz[ip, jp, kp] * dyrho[i, j, k]
        + vx[ip, jp, kp] * bx[ip, jp, kp] * bz[ip, jp, kp] * dzrho[i, j, k]
        + vy[ip, jp, kp] * by[ip, jp, kp] * bz[ip, jp, kp] * dzrho[i, j, k]
        + vz[ip, jp, kp] * bz[ip, jp, kp] * bz[ip, jp, kp] * dzrho[i, j, k]
    )
    drho = rho[ip, jp, kp] - rho[i, j, k]

    return drho * pressNP * vpNPgradrhoP / rho[ip, jp, kp] - drho * pressP * vpPgradrhoNP / rho[i, j, k]
