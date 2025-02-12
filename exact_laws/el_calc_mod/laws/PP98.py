from typing import List

from .abstract_law import AbstractLaw

class Pp98(AbstractLaw):
    def __init__(self):
        self.terms = ["flux_dvdvdv","flux_dbdbdv","flux_dvdbdb"]
        pass
    
    def terms_and_coeffs(self,physical_params):
        coeffs = {} 
        coeffs["div_flux_dvdvdv"] = - physical_params["rho_mean"] / 4
        coeffs["div_flux_dbdbdv"] = - physical_params["rho_mean"] / 4
        coeffs["div_flux_dvdbdb"] = physical_params["rho_mean"] / 2
        return self.terms, coeffs

    def variables(self) -> List[str]:
        return self.list_variables(self.terms)

def load():
    return Pp98()
