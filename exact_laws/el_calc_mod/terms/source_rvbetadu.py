from typing import List
from numba import njit

from .abstract_term import AbstractTerm, calc_source_with_numba


class FluxRvbetadu(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, pm, piso, dxuiso, dyuiso, dzuiso, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, rho, vx, vy, vz, pm, piso, dxuiso, dyuiso, dzuiso)

    def variables(self) -> List[str]:
        return ["rho", "graduiso", "v", "pm", "piso"]


def load():
    return FluxRvbetadu()


@njit
def calc_in_point(i, j, k, ip, jp, kp, rho, vx, vy, vz, pm, piso, dxuiso, dyuiso, dzuiso):
    
    vNPgraduP = vx[i, j, k] * dxuiso[ip, jp, kp] + vy[i, j, k] * dyuiso[ip, jp, kp] + vz[i, j, k] * dzuiso[ip, jp, kp]
    vPgraduNP = vx[ip, jp, kp] * dxuiso[i, j, k] + vy[ip, jp, kp] * dyuiso[i, j, k] + vz[ip, jp, kp] * dzuiso[i, j, k]

    return (
        rho[i, j, k] * pm[ip, jp, kp] / piso[ip, jp, kp] * vNPgraduP
        + rho[ip, jp, kp] * pm[i, j, k] / piso[i, j, k] * vPgraduNP
    )
