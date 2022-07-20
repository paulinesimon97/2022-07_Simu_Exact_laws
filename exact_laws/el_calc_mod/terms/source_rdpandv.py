from typing import List
from numba import njit
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRdpandv(AbstractTerm):
    def __init__(self):
        pass

    def calc(
        self,
        vector: List[int],
        cube_size: List[int],
        rho,
        pperp,
        ppar,
        pm,
        bx,
        by,
        bz,
        dxvx,
        dyvx,
        dzvx,
        dxvy,
        dyvy,
        dzvy,
        dxvz,
        dyvz,
        dzvz,
        **kwarg
    ) -> (float):
        return calc_source_with_numba(
            calc_in_point,
            *vector,
            *cube_size,
            rho,
            pperp,
            ppar,
            pm,
            bx,
            by,
            bz,
            dxvx,
            dyvx,
            dzvx,
            dxvy,
            dyvy,
            dzvy,
            dxvz,
            dyvz,
            dzvz
        )

    def variables(self) -> List[str]:
        return ["rho", "pgyr", "pm", "gradv", "b"]


def load():
    return SourceRdpandv()


@njit
def calc_in_point(
    i, j, k, ip, jp, kp, rho, pperp, ppar, pm, bx, by, bz, dxvx, dyvx, dzvx, dxvy, dyvy, dzvy, dxvz, dyvz, dzvz
):

    pressP = (ppar[ip, jp, kp] - pperp[ip, jp, kp]) / pm[ip, jp, kp]
    pressNP = (ppar[i, j, k] - pperp[i, j, k]) / pm[i, j, k]

    dpxx = pressP * bx[ip, jp, kp] * bx[ip, jp, kp] - pressNP * bx[i, j, k] * bx[i, j, k]
    dpxy = pressP * bx[ip, jp, kp] * by[ip, jp, kp] - pressNP * bx[i, j, k] * by[i, j, k]
    dpxz = pressP * bx[ip, jp, kp] * bz[ip, jp, kp] - pressNP * bx[i, j, k] * bz[i, j, k]
    dpyy = pressP * by[ip, jp, kp] * by[ip, jp, kp] - pressNP * by[i, j, k] * by[i, j, k]
    dpyz = pressP * by[ip, jp, kp] * bz[ip, jp, kp] - pressNP * by[i, j, k] * bz[i, j, k]
    dpzz = pressP * bz[ip, jp, kp] * bz[ip, jp, kp] - pressNP * bz[i, j, k] * bz[i, j, k]

    return rho[i, j, k] * (
        dpxx * dxvx[ip, jp, kp]
        + dpxy * dyvx[ip, jp, kp]
        + dpxz * dzvx[ip, jp, kp]
        + dpxy * dxvy[ip, jp, kp]
        + dpyy * dyvy[ip, jp, kp]
        + dpyz * dzvy[ip, jp, kp]
        + dpxz * dxvz[ip, jp, kp]
        + dpyz * dyvz[ip, jp, kp]
        + dpzz * dzvz[ip, jp, kp]
    ) - rho[ip, jp, kp] * (
        dpxx * dxvx[i, j, k]
        + dpxy * dyvx[i, j, k]
        + dpxz * dzvx[i, j, k]
        + dpxy * dxvy[i, j, k]
        + dpyy * dyvy[i, j, k]
        + dpyz * dzvy[i, j, k]
        + dpxz * dxvz[i, j, k]
        + dpyz * dyvz[i, j, k]
        + dpzz * dzvz[i, j, k]
    )
