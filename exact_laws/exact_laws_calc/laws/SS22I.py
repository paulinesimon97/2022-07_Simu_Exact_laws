from ..abstract_law import AbstractLaw
from typing import List


class Ss22i(AbstractLaw):
    def __init__(self):
        a = 1

    def terms(self, terms, funct, args, coeff, param_origin, helm=False):
        # term 'flux_dvdvdv'
        t = "flux_dvdvdv"
        if helm == True:
            t = t + "_helm"
        terms.append(t)
        funct[t] = self.flux
        if helm == False:
            args[t] = [("vx", "vy", "vz"), ("vx", "vy", "vz"), ("vx", "vy", "vz")]
        else:
            args[t] = [("Ivx", "Ivy", "Ivz"), ("Ivx", "Ivy", "Ivz"), ("Ivx", "Ivy", "Ivz")]
        coeff[f"div_{t}"] = -param_origin["rho_mean"] / 4

        # term 'flux_dbdbdv'
        t = "flux_dbdbdv"
        if helm == True:
            t = t + "_helm"
        terms.append(t)
        funct[t] = self.flux
        if helm == False:
            args[t] = [("Ibx", "Iby", "Ibz"), ("Ibx", "Iby", "Ibz"), ("vx", "vy", "vz")]
        else:
            args[t] = [("Ibx", "Iby", "Ibz"), ("Ibx", "Iby", "Ibz"), ("Ivx", "Ivy", "Ivz")]
        coeff[f"div_{t}"] = -param_origin["rho_mean"] / 4

        # term 'flux_dvdbdb'
        t = "flux_dvdbdb"
        if helm == True:
            t = t + "_helm"
        terms.append(t)
        funct[t] = self.flux
        if helm == False:
            args[t] = [("vx", "vy", "vz"), ("Ibx", "Iby", "Ibz"), ("Ibx", "Iby", "Ibz")]
        else:
            args[t] = [("Ivx", "Ivy", "Ivz"), ("Ibx", "Iby", "Ibz"), ("Ibx", "Iby", "Ibz")]
        coeff[f"div_{t}"] = param_origin["rho_mean"] / 2

        return terms, funct, args, coeff

    def variables(self) -> List[str]:
        return ['v', 'Ib']


def load():
    return Ss22i()
