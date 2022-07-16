from typing import List
from numba import njit
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceDpan(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int],
             Ipperp, Ippar, Ipm,
             Ibx, Iby, Ibz,
             dxvx, dyvx, dzvx,
             dxvy, dyvy, dzvy,
             dxvz, dyvz, dzvz,
             **kwarg) -> (float):
        return calc_source_with_numba(calc_in_point, *vector, *cube_size,
                                      Ipperp, Ippar, Ipm,
                                      Ibx, Iby, Ibz,
                                      dxvx, dyvx, dzvx,
                                      dxvy, dyvy, dzvy,
                                      dxvz, dyvz, dzvz)

    def variables(self) -> List[str]:
        return ["Ipgyr", "Ipm", "gradv", "Ib"]


def load():
    return SourceDpan()


@njit
def calc_in_point(i, j, k, ip, jp, kp,
                  Ipperp, Ippar, Ipm,
                  Ibx, Iby, Ibz,
                  dxvx, dyvx, dzvx,
                  dxvy, dyvy, dzvy,
                  dxvz, dyvz, dzvz):
    ddxvx = dxvx[ip, jp, kp] - dxvx[i, j, k]
    ddyvx = dyvx[ip, jp, kp] - dyvx[i, j, k]
    ddzvx = dzvx[ip, jp, kp] - dzvx[i, j, k]
    ddxvy = dxvy[ip, jp, kp] - dxvy[i, j, k]
    ddyvy = dyvy[ip, jp, kp] - dyvy[i, j, k]
    ddzvy = dzvy[ip, jp, kp] - dzvy[i, j, k]
    ddxvz = dxvz[ip, jp, kp] - dxvz[i, j, k]
    ddyvz = dyvz[ip, jp, kp] - dyvz[i, j, k]
    ddzvz = dzvz[ip, jp, kp] - dzvz[i, j, k]

    pressP = (Ippar[ip, jp, kp] - Ipperp[ip, jp, kp]) / Ipm[ip, jp, kp]
    press = (Ippar[i, j, k] - Ipperp[i, j, k]) / Ipm[i, j, k]

    IbxP = Ibx[ip, jp, kp]
    IbyP = Iby[ip, jp, kp]
    IbzP = Ibz[ip, jp, kp]

    IbxNP = Ibx[i, j, k]
    IbyNP = Iby[i, j, k]
    IbzNP = Ibz[i, j, k]

    return pressP * (IbxP * (IbxP * ddxvx + IbyP * ddxvy + IbzP * ddxvz) + IbyP * (
            IbxP * ddyvx + IbyP * ddyvy + IbzP * ddyvz) + IbzP * (
                             IbxP * ddzvx + IbyP * ddzvy + IbzP * ddzvz)) - press * (
                   IbxNP * (IbxNP * ddxvx + IbyNP * ddxvy + IbzNP * ddxvz) + IbyNP * (
                   IbxNP * ddyvx + IbyNP * ddyvy + IbzNP * ddyvz) + IbzNP * (
                               IbxNP * ddzvx + IbyNP * ddzvy + IbzNP * ddzvz))
