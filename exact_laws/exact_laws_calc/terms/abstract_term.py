from typing import List
import numpy as np
import numexpr as ne
from numba import njit
from functools import reduce


class AbstractTerm:
    def __init__(self):
        pass

    def calc(self, *args, **kwargs) -> (float or List[float]):
        raise NotImplementedError("You have to reimplement this method")

    def variables(self) -> List[str]:
        raise NotImplementedError("You have to reimplement this method")

    # def flux_gen(self, data_1, data_2, data_3, data_4=None, datadic={}):
    #     """general expression for flux of the form (data_1 . data_2) data_3
    #     ex for K41 : self.flux_gen(('vx','vy','vz'),('vx','vy','vz'),('vx','vy','vz'))
    #     need : data_1, data_2, data_3 and conjugates in local_dict.keys()
    #     return an iterator on the flux components summed on all the points of the original grid"""

    #     if data_4 == None:  # calcul flux term with 3 quantities (ex: dv.dvdv)
    #         scalar_product = ""
    #         for i in range(len(data_1)):
    #             scalar_product += f" ({data_1[i]}P - {data_1[i]}) * ({data_2[i]}P - {data_2[i]}) +"
    #         for j in range(len(data_3)):
    #             tab = ne.evaluate(
    #                 f"   ({scalar_product[:-1]}) * ({data_3[j]}P - {data_3[j]}) ".lstrip(),
    #                 local_dict=datadic,
    #             )
    #             yield np.sum(tab)  # np.sum(np.sort(tab.flatten()))

    #     elif data_4 == "rho":  # calcul compressible flux term  (ex: drv.dvdv)
    #         scalar_product = ""
    #         for i in range(len(data_1)):
    #             scalar_product += (
    #                 f" ({data_4}P * {data_1[i]}P      "
    #                 f"- {data_4}  * {data_1[i]})      "
    #                 f"* ({data_2[i]}P - {data_2[i]}) +"
    #             )
    #         for i in range(len(data_3)):
    #             tab = ne.evaluate(
    #                 f"   ({scalar_product[:-1]}) * ({data_3[i]}P - {data_3[i]})".lstrip(),
    #                 local_dict=datadic,
    #             )
    #             yield np.sum(tab)  # np.sum(np.sort(tab.flatten()))

    #     elif data_4 == "pan":
    #         scalar_productP = ""
    #         scalar_product = ""
    #         for i in range(len(data_3)):
    #             scalar_productP += f"{data_2[i]}P * ({data_3[i]}P - {data_3[i]}) +"
    #             scalar_product += f"{data_2[i]}  * ({data_3[i]}P - {data_3[i]}) +"
    #         for i in range(len(data_2)):
    #             tab = ne.evaluate(
    #                 f"({data_1[0]}P - {data_1[0]})                 "
    #                 f" * ((pparP - pperpP) / pmP * {data_2[i]}P    "
    #                 f"   * {scalar_productP[:-1]}                  "
    #                 f"  - (ppar  - pperp)  / pm  * {data_2[i]}     "
    #                 f"   * {scalar_product[:-1]})                  ".lstrip(),
    #                 local_dict=datadic,
    #             )
    #             yield np.sum(tab)  # np.sum(np.sort(tab.flatten()))

    # def flux(self, data_1, data_2, data_3, data_4=None, datadic={}):
    #     """general expression for flux of the form (data_1 . data_2) data_3
    #     ex for K41 : self.flux(('vx','vy','vz'),('vx','vy','vz'),('vx','vy','vz'))
    #     need : data_1, data_2, data_3 and conjugates in local_dict.keys()
    #     return a list of the flux components summed on all the points of the original grid"""
    #     return np.array(list(self.flux_gen(data_1, data_2, data_3, data_4, datadic=datadic)))

    # def source_iso(self, func, d="", datadic={}):

    #     P = "P"
    #     NP = " "
    #     return np.sum(ne.evaluate(func(NP, P, d).lstrip(), local_dict=datadic)) + np.sum(
    #         ne.evaluate(func(P, NP, d).lstrip(), local_dict=datadic)
    #     )

    # def source_an(self, func, d="", datadic={}):

    #     P = "P"
    #     NP = " "

    #     if d == "v":
    #         # prim
    #         tab = np.sum(ne.evaluate(func(NP, P, P).lstrip(), local_dict=datadic))
    #         # non prim
    #         tab -= np.sum(ne.evaluate(func(NP, P, NP).lstrip(), local_dict=datadic))
    #         # conjugate prim
    #         tab -= np.sum(ne.evaluate(func(P, NP, P).lstrip(), local_dict=datadic))
    #         # conjugate non prim
    #         tab += np.sum(ne.evaluate(func(P, NP, NP).lstrip(), local_dict=datadic))

    #     elif d == "r":
    #         # term 1
    #         tab = np.sum(ne.evaluate(func(NP, P, "n").lstrip(), local_dict=datadic))
    #         # term 1 conjugate
    #         tab += np.sum(ne.evaluate(func(P, NP, "n").lstrip(), local_dict=datadic))
    #         # term 2 prim
    #         tab -= np.sum(ne.evaluate(func(NP, P, P).lstrip(), local_dict=datadic))
    #         # term 2 non prim
    #         tab += np.sum(ne.evaluate(func(NP, P, NP).lstrip(), local_dict=datadic))
    #         # term 2 conjugate prim
    #         tab += np.sum(ne.evaluate(func(P, NP, P).lstrip(), local_dict=datadic))
    #         # term 2 conjugate non prim
    #         tab -= np.sum(ne.evaluate(func(P, NP, NP).lstrip(), local_dict=datadic))

    #     return tab

    # def BG17_term(data_1, data_2, data_3, datadic={}):
    #     """general expression for BG17 terms in of the form delta(data_1 x data_2).delta(data_3)
    #     need : data_1, data_2, data_3 and conjugates in local_dict.keys()
    #     return the term summed on all the points of the original grid"""
    #     d0 = (
    #         f"  {data_1[1]}P * {data_2[2]}P - {data_1[2]}P * {data_2[1]}P "
    #         f"- {data_1[1]}  * {data_2[2]}  + {data_1[2]}  * {data_2[1]}  "
    #     )
    #     d1 = (
    #         f"  {data_1[2]}P * {data_2[0]}P - {data_1[0]}P * {data_2[2]}P "
    #         f"- {data_1[2]}  * {data_2[0]}  + {data_1[0]}  * {data_2[2]}  "
    #     )
    #     d2 = (
    #         f"  {data_1[0]}P * {data_2[1]}P - {data_1[1]}P * {data_2[0]}P "
    #         f"- {data_1[0]}  * {data_2[1]}  + {data_1[1]}  * {data_2[0]}  "
    #     )
    #     tab = ne.evaluate(
    #         f"  ({d0}) * ({data_3[0]}P - {data_3[0]}) "
    #         f"+ ({d1}) * ({data_3[1]}P - {data_3[1]}) "
    #         f"+ ({d2}) * ({data_3[2]}P - {data_3[2]}) ".lstrip(),
    #         local_dict=datadic,
    #     )
    #     return np.sum(tab)  # np.sum(np.sort(tab.flatten()))


def load():
    return AbstractTerm()


@njit
def calc_source_with_numba(funct, dx, dy, dz, Nx, Ny, Nz, *quantities):
    acc = 0.0

    # face N0
    offset_dirr0 = Nx - dx
    offset_dirr1 = Ny - dy
    offset_dirr2 = Nz - dz

    # cube
    for i in range(Nx - dx):
        for j in range(Ny - dy):
            for k in range(Nz - dz):
                acc += funct(i, j, k, i + dx, j + dy, k + dz, *quantities)

    # face N0
    for i in range(dx):
        for j in range(Ny - dy):
            for k in range(Nz - dz):
                acc += funct(i + offset_dirr0, j, k, i, j + dy, k + dz, *quantities)

    # face N1
    for i in range(Nx - dx):
        for j in range(dy):
            for k in range(Nz - dz):
                acc += funct(i, j + offset_dirr1, k, i + dx, j, k + dz, *quantities)

    # face N2
    for i in range(Nx - dx):
        for j in range(Ny - dy):
            for k in range(dz):
                acc += funct(i, j, k + offset_dirr2, i + dx, j + dy, k, *quantities)

    # arrète N01
    for i in range(dx):
        for j in range(dy):
            for k in range(Nz - dz):
                acc += funct(i + offset_dirr0, j + offset_dirr1, k, i, j, k + dz, *quantities)

    # arrète N02
    for i in range(dx):
        for j in range(Ny - dy):
            for k in range(dz):
                acc += funct(i + offset_dirr0, j, k + offset_dirr2, i, j + dy, k, *quantities)

    # arrète N12
    for i in range(Nx - dx):
        for j in range(dy):
            for k in range(dz):
                acc += funct(i, j + offset_dirr1, k + offset_dirr2, i + dx, j, k, *quantities)

    # sommet
    for i in range(dx):
        for j in range(dy):
            for k in range(dz):
                acc += funct(i + offset_dirr0, j + offset_dirr1, k + offset_dirr2, i, j, k, *quantities)

    return acc / (Nx * Ny * Nz)


@njit
def calc_flux_with_numba_test1(funct_x, funct_y, funct_z, dx, dy, dz, Nx, Ny, Nz, *quantities):
    acc_x = 0.0
    acc_y = 0.0
    acc_z = 0.0

    # face N0
    offset_dirr0 = Nx - dx
    offset_dirr1 = Ny - dy
    offset_dirr2 = Nz - dz

    # cube
    for i in range(Nx - dx):
        for j in range(Ny - dy):
            for k in range(Nz - dz):
                acc_x += funct_x(i, j, k, i + dx, j + dy, k + dz, *quantities)
                acc_y += funct_y(i, j, k, i + dx, j + dy, k + dz, *quantities)
                acc_z += funct_z(i, j, k, i + dx, j + dy, k + dz, *quantities)

    # face N0
    for i in range(dx):
        for j in range(Ny - dy):
            for k in range(Nz - dz):
                acc_x += funct_x(i + offset_dirr0, j, k, i, j + dy, k + dz, *quantities)
                acc_y += funct_y(i + offset_dirr0, j, k, i, j + dy, k + dz, *quantities)
                acc_z += funct_z(i + offset_dirr0, j, k, i, j + dy, k + dz, *quantities)

    # face N1
    for i in range(Nx - dx):
        for j in range(dy):
            for k in range(Nz - dz):
                acc_x += funct_x(i, j + offset_dirr1, k, i + dx, j, k + dz, *quantities)
                acc_y += funct_y(i, j + offset_dirr1, k, i + dx, j, k + dz, *quantities)
                acc_z += funct_z(i, j + offset_dirr1, k, i + dx, j, k + dz, *quantities)

    # face N2
    for i in range(Nx - dx):
        for j in range(Ny - dy):
            for k in range(dz):
                acc_x += funct_x(i, j, k + offset_dirr2, i + dx, j + dy, k, *quantities)
                acc_y += funct_y(i, j, k + offset_dirr2, i + dx, j + dy, k, *quantities)
                acc_z += funct_z(i, j, k + offset_dirr2, i + dx, j + dy, k, *quantities)

    # arrète N01
    for i in range(dx):
        for j in range(dy):
            for k in range(Nz - dz):
                acc_x += funct_x(i + offset_dirr0, j + offset_dirr1, k, i, j, k + dz, *quantities)
                acc_y += funct_y(i + offset_dirr0, j + offset_dirr1, k, i, j, k + dz, *quantities)
                acc_z += funct_z(i + offset_dirr0, j + offset_dirr1, k, i, j, k + dz, *quantities)

    # arrète N02
    for i in range(dx):
        for j in range(Ny - dy):
            for k in range(dz):
                acc_x += funct_x(i + offset_dirr0, j, k + offset_dirr2, i, j + dy, k, *quantities)
                acc_y += funct_y(i + offset_dirr0, j, k + offset_dirr2, i, j + dy, k, *quantities)
                acc_z += funct_z(i + offset_dirr0, j, k + offset_dirr2, i, j + dy, k, *quantities)

    # arrète N12
    for i in range(Nx - dx):
        for j in range(dy):
            for k in range(dz):
                acc_x += funct_x(i, j + offset_dirr1, k + offset_dirr2, i + dx, j, k, *quantities)
                acc_y += funct_y(i, j + offset_dirr1, k + offset_dirr2, i + dx, j, k, *quantities)
                acc_z += funct_z(i, j + offset_dirr1, k + offset_dirr2, i + dx, j, k, *quantities)

    # sommet
    for i in range(dx):
        for j in range(dy):
            for k in range(dz):
                acc_x += funct_x(i + offset_dirr0, j + offset_dirr1, k + offset_dirr2, i, j, k, *quantities)
                acc_y += funct_y(i + offset_dirr0, j + offset_dirr1, k + offset_dirr2, i, j, k, *quantities)
                acc_z += funct_z(i + offset_dirr0, j + offset_dirr1, k + offset_dirr2, i, j, k, *quantities)

    return [acc_x / (Nx * Ny * Nz), acc_y / (Nx * Ny * Nz), acc_z / (Nx * Ny * Nz)]

@njit
def calc_flux_with_numba_test2(funct, dx, dy, dz, Nx, Ny, Nz, *quantities):
    acc_x = 0.0
    acc_y = 0.0
    acc_z = 0.0

    # face N0
    offset_dirr0 = Nx - dx
    offset_dirr1 = Ny - dy
    offset_dirr2 = Nz - dz

    # cube
    for i in range(Nx - dx):
        for j in range(Ny - dy):
            for k in range(Nz - dz):
                acc_x += funct(i, j, k, i + dx, j + dy, k + dz, *quantities)[0]
                acc_y += funct(i, j, k, i + dx, j + dy, k + dz, *quantities)[1]
                acc_z += funct(i, j, k, i + dx, j + dy, k + dz, *quantities)[2]

    # face N0
    for i in range(dx):
        for j in range(Ny - dy):
            for k in range(Nz - dz):
                acc_x += funct(i + offset_dirr0, j, k, i, j + dy, k + dz, *quantities)[0]
                acc_y += funct(i + offset_dirr0, j, k, i, j + dy, k + dz, *quantities)[1]
                acc_z += funct(i + offset_dirr0, j, k, i, j + dy, k + dz, *quantities)[2]

    # face N1
    for i in range(Nx - dx):
        for j in range(dy):
            for k in range(Nz - dz):
                acc_x += funct(i, j + offset_dirr1, k, i + dx, j, k + dz, *quantities)[0]
                acc_y += funct(i, j + offset_dirr1, k, i + dx, j, k + dz, *quantities)[1]
                acc_z += funct(i, j + offset_dirr1, k, i + dx, j, k + dz, *quantities)[2]

    # face N2
    for i in range(Nx - dx):
        for j in range(Ny - dy):
            for k in range(dz):
                acc_x += funct(i, j, k + offset_dirr2, i + dx, j + dy, k, *quantities)[0]
                acc_y += funct(i, j, k + offset_dirr2, i + dx, j + dy, k, *quantities)[1]
                acc_z += funct(i, j, k + offset_dirr2, i + dx, j + dy, k, *quantities)[2]

    # arrète N01
    for i in range(dx):
        for j in range(dy):
            for k in range(Nz - dz):
                acc_x += funct(i + offset_dirr0, j + offset_dirr1, k, i, j, k + dz, *quantities)[0]
                acc_y += funct(i + offset_dirr0, j + offset_dirr1, k, i, j, k + dz, *quantities)[1]
                acc_z += funct(i + offset_dirr0, j + offset_dirr1, k, i, j, k + dz, *quantities)[2]

    # arrète N02
    for i in range(dx):
        for j in range(Ny - dy):
            for k in range(dz):
                acc_x += funct(i + offset_dirr0, j, k + offset_dirr2, i, j + dy, k, *quantities)[0]
                acc_y += funct(i + offset_dirr0, j, k + offset_dirr2, i, j + dy, k, *quantities)[1]
                acc_z += funct(i + offset_dirr0, j, k + offset_dirr2, i, j + dy, k, *quantities)[2]

    # arrète N12
    for i in range(Nx - dx):
        for j in range(dy):
            for k in range(dz):
                acc_x += funct(i, j + offset_dirr1, k + offset_dirr2, i + dx, j, k, *quantities)[0]
                acc_y += funct(i, j + offset_dirr1, k + offset_dirr2, i + dx, j, k, *quantities)[1]
                acc_z += funct(i, j + offset_dirr1, k + offset_dirr2, i + dx, j, k, *quantities)[2]

    # sommet
    for i in range(dx):
        for j in range(dy):
            for k in range(dz):
                acc_x += funct(i + offset_dirr0, j + offset_dirr1, k + offset_dirr2, i, j, k, *quantities)[0]
                acc_y += funct(i + offset_dirr0, j + offset_dirr1, k + offset_dirr2, i, j, k, *quantities)[1]
                acc_z += funct(i + offset_dirr0, j + offset_dirr1, k + offset_dirr2, i, j, k, *quantities)[2]

    return [acc_x / (Nx * Ny * Nz), acc_y / (Nx * Ny * Nz), acc_z / (Nx * Ny * Nz)]

@njit
def calc_flux_with_numba(funct, dx, dy, dz, Nx, Ny, Nz, *quantities):
    acc_x = 0.0
    acc_y = 0.0
    acc_z = 0.0

    # face N0
    offset_dirr0 = Nx - dx
    offset_dirr1 = Ny - dy
    offset_dirr2 = Nz - dz

    # cube
    for i in range(Nx - dx):
        for j in range(Ny - dy):
            for k in range(Nz - dz):
                x, y, z = funct(i, j, k, i + dx, j + dy, k + dz, *quantities)
                acc_x += x
                acc_y += y
                acc_z += z

    # face N0
    for i in range(dx):
        for j in range(Ny - dy):
            for k in range(Nz - dz):
                x, y, z = funct(i + offset_dirr0, j, k, i, j + dy, k + dz, *quantities)
                acc_x += x
                acc_y += y
                acc_z += z

    # face N1
    for i in range(Nx - dx):
        for j in range(dy):
            for k in range(Nz - dz):
                x, y, z = funct(i, j + offset_dirr1, k, i + dx, j, k + dz, *quantities)
                acc_x += x
                acc_y += y
                acc_z += z

    # face N2
    for i in range(Nx - dx):
        for j in range(Ny - dy):
            for k in range(dz):
                x, y, z = funct(i, j, k + offset_dirr2, i + dx, j + dy, k, *quantities)
                acc_x += x
                acc_y += y
                acc_z += z

    # arrète N01
    for i in range(dx):
        for j in range(dy):
            for k in range(Nz - dz):
                x, y, z = funct(i + offset_dirr0, j + offset_dirr1, k, i, j, k + dz, *quantities)
                acc_x += x
                acc_y += y
                acc_z += z

    # arrète N02
    for i in range(dx):
        for j in range(Ny - dy):
            for k in range(dz):
                x, y, z = funct(i + offset_dirr0, j, k + offset_dirr2, i, j + dy, k, *quantities)
                acc_x += x
                acc_y += y
                acc_z += z

    # arrète N12
    for i in range(Nx - dx):
        for j in range(dy):
            for k in range(dz):
                x, y, z = funct(i, j + offset_dirr1, k + offset_dirr2, i + dx, j, k, *quantities)
                acc_x += x
                acc_y += y
                acc_z += z

    # sommet
    for i in range(dx):
        for j in range(dy):
            for k in range(dz):
                x, y, z = funct(i + offset_dirr0, j + offset_dirr1, k + offset_dirr2, i, j, k, *quantities)
                acc_x += x
                acc_y += y
                acc_z += z

    return [acc_x / (Nx * Ny * Nz), acc_y / (Nx * Ny * Nz), acc_z / (Nx * Ny * Nz)]

