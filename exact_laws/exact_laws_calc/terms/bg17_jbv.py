from typing import List

from .abstract_term import AbstractTerm

class Bg17Jbv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, values) -> (float or List[float]):
        return self.BG17_term(("Ijx", "Ijy", "Ijz"), ("Ibx", "Iby", "Ibz"), ("vx", "vy", "vz"), datadic=values)

    def variables(self) -> List[str]:
        return ['Ij','Ib','v']

def load():
    return Bg17Jbv()
    
    