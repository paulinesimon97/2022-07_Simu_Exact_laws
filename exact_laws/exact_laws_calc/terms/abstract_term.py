from typing import List
import numpy as np
import numexpr as ne

class AbstractTerm:
    def __init__(self):
        pass
    
    def calc(self, *args, **kwargs) -> (float or List[float]):
        raise NotImplementedError("You have to reimplement this method")

    def variables(self) -> List[str]:
        raise NotImplementedError("You have to reimplement this method") 
    
    def flux_gen(self, data_1, data_2, data_3, data_4=None, datadic={}):
        """general expression for flux of the form (data_1 . data_2) data_3
        ex for K41 : self.flux_gen(('vx','vy','vz'),('vx','vy','vz'),('vx','vy','vz'))
        need : data_1, data_2, data_3 and conjugates in local_dict.keys()
        return an iterator on the flux components summed on all the points of the original grid"""

        if data_4 == None:  # calcul flux term with 3 quantities (ex: dv.dvdv)
            scalar_product = ""
            for i in range(len(data_1)):
                scalar_product += f" ({data_1[i]}P - {data_1[i]}) * ({data_2[i]}P - {data_2[i]}) +"
            for j in range(len(data_3)):
                tab = ne.evaluate(
                    f"   ({scalar_product[:-1]}) * ({data_3[j]}P - {data_3[j]}) ".lstrip(),
                    local_dict=datadic,
                )
                yield np.sum(tab)  # np.sum(np.sort(tab.flatten()))

        elif data_4 == "rho":  # calcul compressible flux term  (ex: drv.dvdv)
            scalar_product = ""
            for i in range(len(data_1)):
                scalar_product += (
                    f" ({data_4}P * {data_1[i]}P      "
                    f"- {data_4}  * {data_1[i]})      "
                    f"* ({data_2[i]}P - {data_2[i]}) +"
                )
            for i in range(len(data_3)):
                tab = ne.evaluate(
                    f"   ({scalar_product[:-1]}) * ({data_3[i]}P - {data_3[i]})".lstrip(),
                    local_dict=datadic,
                )
                yield np.sum(tab)  # np.sum(np.sort(tab.flatten()))

        elif data_4 == "pan":
            scalar_productP = ""
            scalar_product = ""
            for i in range(len(data_3)):
                scalar_productP += f"{data_2[i]}P * ({data_3[i]}P - {data_3[i]}) +"
                scalar_product += f"{data_2[i]}  * ({data_3[i]}P - {data_3[i]}) +"
            for i in range(len(data_2)):
                tab = ne.evaluate(
                    f"({data_1[0]}P - {data_1[0]})                 "
                    f" * ((pparP - pperpP) / pmP * {data_2[i]}P    "
                    f"   * {scalar_productP[:-1]}                  "
                    f"  - (ppar  - pperp)  / pm  * {data_2[i]}     "
                    f"   * {scalar_product[:-1]})                  ".lstrip(),
                    local_dict=datadic,
                )
                yield np.sum(tab)  # np.sum(np.sort(tab.flatten()))

    def flux(self, data_1, data_2, data_3, data_4=None, datadic={}):
        """general expression for flux of the form (data_1 . data_2) data_3
        ex for K41 : self.flux(('vx','vy','vz'),('vx','vy','vz'),('vx','vy','vz'))
        need : data_1, data_2, data_3 and conjugates in local_dict.keys()
        return a list of the flux components summed on all the points of the original grid"""
        return np.array(list(self.flux_gen(data_1, data_2, data_3, data_4, datadic=datadic)))

    def source_iso(self, func, d="", datadic={}):

        P = "P"
        NP = " "
        return np.sum(ne.evaluate(func(NP, P, d).lstrip(), local_dict=datadic)) + np.sum(
            ne.evaluate(func(P, NP, d).lstrip(), local_dict=datadic)
        )

    def source_an(self, func, d="", datadic={}):

        P = "P"
        NP = " "

        if d == "v":
            # prim
            tab = np.sum(ne.evaluate(func(NP, P, P).lstrip(), local_dict=datadic))
            # non prim
            tab -= np.sum(ne.evaluate(func(NP, P, NP).lstrip(), local_dict=datadic))
            # conjugate prim
            tab -= np.sum(ne.evaluate(func(P, NP, P).lstrip(), local_dict=datadic))
            # conjugate non prim
            tab += np.sum(ne.evaluate(func(P, NP, NP).lstrip(), local_dict=datadic))

        elif d == "r":
            # term 1
            tab = np.sum(ne.evaluate(func(NP, P, "n").lstrip(), local_dict=datadic))
            # term 1 conjugate
            tab += np.sum(ne.evaluate(func(P, NP, "n").lstrip(), local_dict=datadic))
            # term 2 prim
            tab -= np.sum(ne.evaluate(func(NP, P, P).lstrip(), local_dict=datadic))
            # term 2 non prim
            tab += np.sum(ne.evaluate(func(NP, P, NP).lstrip(), local_dict=datadic))
            # term 2 conjugate prim
            tab += np.sum(ne.evaluate(func(P, NP, P).lstrip(), local_dict=datadic))
            # term 2 conjugate non prim
            tab -= np.sum(ne.evaluate(func(P, NP, NP).lstrip(), local_dict=datadic))

        return tab
    
    def BG17_term(data_1, data_2, data_3, datadic={}):
        """general expression for BG17 terms in of the form delta(data_1 x data_2).delta(data_3)
        need : data_1, data_2, data_3 and conjugates in local_dict.keys()
        return the term summed on all the points of the original grid"""
        d0 = (
            f"  {data_1[1]}P * {data_2[2]}P - {data_1[2]}P * {data_2[1]}P "
            f"- {data_1[1]}  * {data_2[2]}  + {data_1[2]}  * {data_2[1]}  "
        )
        d1 = (
            f"  {data_1[2]}P * {data_2[0]}P - {data_1[0]}P * {data_2[2]}P "
            f"- {data_1[2]}  * {data_2[0]}  + {data_1[0]}  * {data_2[2]}  "
        )
        d2 = (
            f"  {data_1[0]}P * {data_2[1]}P - {data_1[1]}P * {data_2[0]}P "
            f"- {data_1[0]}  * {data_2[1]}  + {data_1[1]}  * {data_2[0]}  "
        )
        tab = ne.evaluate(
            f"  ({d0}) * ({data_3[0]}P - {data_3[0]}) "
            f"+ ({d1}) * ({data_3[1]}P - {data_3[1]}) "
            f"+ ({d2}) * ({data_3[2]}P - {data_3[2]}) ".lstrip(),
            local_dict=datadic,
        )
        return np.sum(tab)  # np.sum(np.sort(tab.flatten()))
