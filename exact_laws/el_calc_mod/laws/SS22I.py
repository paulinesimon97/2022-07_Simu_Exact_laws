from typing import List

from .abstract_law import AbstractLaw

class Ss22i(AbstractLaw):
    def __init__(self):
        self.terms = ["flux_dvdvdv","flux_dbdbdv","flux_dvdbdb"]
        pass
    
    def terms_and_coeffs(self,physical_params):
        coeffs = {} 
        coeffs["div_flux_dvdvdv"] = - physical_params["rho_mean"] / 4
        coeffs["div_flux_dbdbdv"] = - physical_params["rho_mean"] / 4
        coeffs["div_flux_dbdbdv"] = physical_params["rho_mean"] / 2
        return self.terms, coeffs

    def variables(self) -> List[str]:
        return self.list_variables(self.terms)

def load():
    return Ss22i()
