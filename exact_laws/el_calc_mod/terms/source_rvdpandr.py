from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRvdpandr(AbstractTerm):
    def __init__(self):
        pass

    def calc(
        self,
        vector: List[int],
        cube_size: List[int],
        rho,
        vx,
        vy,
        vz,
        pperp,
        ppar,
        pm,
        bx,
        by,
        bz,
        dxrho,
        dyrho,
        dzrho,
        **kwarg
    ) -> List[float]:
        return calc_source_with_numba(
            calc_in_point, *vector, *cube_size, rho, vx, vy, vz, pperp, ppar, pm, bx, by, bz, dxrho, dyrho, dzrho
        )

    def variables(self) -> List[str]:
        return ["rho", "gradrho", "v", "pgyr", "pm", "b"]


def load():
    return SourceRvdpandr()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, vx, vy, vz, pperp, ppar, pm, bx, by, bz, dxrho, dyrho, dzrho):

    pressP = (ppar[ip, jp, kp] - pperp[ip, jp, kp]) / pm[ip, jp, kp]
    pressNP = (ppar[i, j, k] - pperp[i, j, k]) / pm[i, j, k]

    dpxx = pressP * bx[ip, jp, kp] * bx[ip, jp, kp] - pressNP * bx[i, j, k] * bx[i, j, k]
    dpxy = pressP * bx[ip, jp, kp] * by[ip, jp, kp] - pressNP * bx[i, j, k] * by[i, j, k]
    dpxz = pressP * bx[ip, jp, kp] * bz[ip, jp, kp] - pressNP * bx[i, j, k] * bz[i, j, k]
    dpyy = pressP * by[ip, jp, kp] * by[ip, jp, kp] - pressNP * by[i, j, k] * by[i, j, k]
    dpyz = pressP * by[ip, jp, kp] * bz[ip, jp, kp] - pressNP * by[i, j, k] * bz[i, j, k]
    dpzz = pressP * bz[ip, jp, kp] * bz[ip, jp, kp] - pressNP * bz[i, j, k] * bz[i, j, k]

    vNPdpgradrhoP = (
        vx[i, j, k] * dpxx * dxrho[ip, jp, kp]
        + vy[i, j, k] * dpxy * dxrho[ip, jp, kp]
        + vz[i, j, k] * dpxz * dxrho[ip, jp, kp]
        + vx[i, j, k] * dpxy * dyrho[ip, jp, kp]
        + vy[i, j, k] * dpyy * dyrho[ip, jp, kp]
        + vz[i, j, k] * dpyz * dyrho[ip, jp, kp]
        + vx[i, j, k] * dpxz * dzrho[ip, jp, kp]
        + vy[i, j, k] * dpyz * dzrho[ip, jp, kp]
        + vz[i, j, k] * dpzz * dzrho[ip, jp, kp]
    )
    vPdpgradrhoNP = (
        vx[ip, jp, kp] * dpxx * dxrho[i, j, k]
        + vy[ip, jp, kp] * dpxy * dxrho[i, j, k]
        + vz[ip, jp, kp] * dpxz * dxrho[i, j, k]
        + vx[ip, jp, kp] * dpxy * dyrho[i, j, k]
        + vy[ip, jp, kp] * dpyy * dyrho[i, j, k]
        + vz[ip, jp, kp] * dpyz * dyrho[i, j, k]
        + vx[ip, jp, kp] * dpxz * dzrho[i, j, k]
        + vy[ip, jp, kp] * dpyz * dzrho[i, j, k]
        + vz[ip, jp, kp] * dpzz * dzrho[i, j, k]
    )

    return rho[i, j, k] * vNPdpgradrhoP / rho[ip, jp, kp] - rho[ip, jp, kp] * vPdpgradrhoNP / rho[i, j, k]
