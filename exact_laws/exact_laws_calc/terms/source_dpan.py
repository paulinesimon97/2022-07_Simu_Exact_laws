from typing import List
import numpy as np
import numexpr as ne

from .abstract_term import AbstractTerm

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
    
    def calc(self, values) -> (float or List[float]):
        return self.source_dp(datadic=values)

    def variables(self) -> List[str]:
        return ['Ipgyr','Ipm','gradv','Ib']

def load():
    return SourceDpan()
    
    