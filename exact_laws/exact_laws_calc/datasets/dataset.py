import logging
from ..grids import Grid

class Dataset:
    """Classe de données ayant pour attributs:
        - quantities[dict{name:np.array}]: contient les tableaux de valeurs des quantités
        - grid[Grid object]: contient les informations sur la grille
        - params[dict]: contient d'autres informations"""

    def __init__(self, params={}, quantities={}, grid=Grid()):
        """Initialisation of the mother class data"""
        self.params = params
        self.quantities = quantities  # dictionary 
        self.grid = grid  # grid
    
    def describ(self,name:str) -> str:
        message = f"\n\t - Params:"
        for k in self.params.keys():
            message += f"\n\t\t - {k} = {self.params[k]}"
        message += f"\n\t - Quantities:"
        for k in self.quantities.keys():
            #tab = self.quantities[k]
            message += f"\n\t\t - {k}"  # = {np.mean(np.sort(tab.copy().reshape(np.product(tab.shape))))}"))
        message += f"\n\t - Grid: {self.describ().replace('\t', '\t\t')}"
        return message
    
    def check(self,name:str) -> None:
        message = message = f"Check Dataset object {name}:"
        message += self.describ()
        logging.info(message)
        
            