from typing import List
import numpy as np
import numexpr as ne
from numba import njit
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceDpan(AbstractTerm):
    def __init__(self):
        pass

    def source_dp(datadic, meth=1):

        exprP = f"(IpparP - IpperpP) / (IpmP)"
        expr = f"- (Ippar - Ipperp) / (Ipm)"

        if meth == 1:
            pdualPP, pdualP1, pdual1P, pdual11 = f"", f"", f"", f""
            for i in ("x", "y", "z"):
                for j in ("x", "y", "z"):
                    pdualPP += f"Ib{i}P * Ib{j}P * d{i}v{j}P +"
                    pdualP1 += f"Ib{i}P * Ib{j}P * d{i}v{j} +"
                    pdual1P += f"Ib{i}  * Ib{j}  * d{i}v{j}P +"
                    pdual11 += f"Ib{i}  * Ib{j}  * d{i}v{j} +"
            tab = ne.evaluate(f"{exprP} * (({pdualPP[:-1]}) - ({pdualP1[:-1]}))".lstrip(), local_dict=datadic)
            out = np.sum(tab)
            tab = ne.evaluate(f"{expr} * (({pdual1P[:-1]}) - ({pdual11[:-1]}))".lstrip(), local_dict=datadic)
            out = out + np.sum(tab)

        else:
            dualP, dual1 = f"", f""
            for i in ("x", "y", "z"):
                for j in ("x", "y", "z"):
                    dualP += f"Ib{i}P * Ib{j}P * (d{i}v{j}P - d{i}v{j})+"
                    dual1 += f"Ib{i}  * Ib{j}  * (d{i}v{j}P - d{i}v{j})+"
            tab = ne.evaluate(f"{exprP} * ({dualP[:-1]})".lstrip(), local_dict=datadic)
            out = np.sum(tab)
            tab = ne.evaluate(f"{expr} * ({dual1[:-1]})".lstrip(), local_dict=datadic)
            out = out + np.sum(tab)

        return out

    def calc_old(self, values) -> (float or List[float]):
        return self.source_dp(datadic=values)

    def calc(self, vector: List[int], cube_size: List[int],
             Ipperp, Ippar, Ipm,
             Ibx, Iby, Ibz,
             dxvx, dyvx, dzvx,
             dxvy, dyvy, dzvy,
             dxvz, dyvz, dzvz,
             **kwarg) -> (float):
        # return calc_source_with_numba(np.array(vector), np.array(cube_size), f2, vx)
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
