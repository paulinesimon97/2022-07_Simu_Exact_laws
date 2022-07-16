from typing import List

from .abstract_law import AbstractLaw

class Bg17(AbstractLaw):
    def __init__(self):
        self.terms = ["BG17_vwv","BG17_jbv","BG17_vbj"]
        pass
    
    def terms_and_coeffs(self,physical_params):
        coeffs = {} 
        coeffs["BG17_vwv"] = physical_params["rho_mean"] / 2
        coeffs["BG17_jbv"] = physical_params["rho_mean"] / 2
        coeffs["BG17_vbj"] = physical_params["rho_mean"] / 2
        return self.terms, coeffs

    def variables(self) -> List[str]:
        return self.list_variables(self.terms)

def load():
    return Bg17()
