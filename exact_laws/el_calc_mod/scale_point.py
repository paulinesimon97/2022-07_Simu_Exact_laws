import numpy as np


class ScalePoint:
    """Classe contenant les informations nécessaires sur une grille de données
    Mère de la classe contenant les informations nécessaires sur la grille finale (grille d'échelles)
    """

    def __init__(self, dx: int, dy: int, dz: int, role="prim"):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.adjacents = [[],[],[]]
        self.role = role
        self.dejavu = False
        self.loc_vector = ()
        self.loc_scalar = ()
        self.lpar = 0 #necessaire ? 
        self.lperp = 0 #necessaire ? 
        self.c = [] #necessaire ? 

    def set_adjacents(self, N, nb_by_dirr):
        deviations = list(np.arange(-nb_by_dirr, nb_by_dirr + 1, 1))
        deviations.remove(0)
        for dirr in range(3):
            for d in deviations:
                central_point = [self.dx, self.dy, self.dz]
                adj_point = central_point.copy()
                adj_point[dirr] += (
                    d
                    - (N[dirr] * ((adj_point[dirr] + d) >= (N[dirr] / 2)))
                    + (N[dirr] * ((adj_point[dirr] + d) <= (-N[dirr] / 2)))
                )
                self.adjacents[dirr].append(ScalePoint(*adj_point, role="sec"))

    def set_dejavu(self, dejavu=True):
        self.dejavu = dejavu
        
    def get_adjacents(self):
        return self.adjacents
    
