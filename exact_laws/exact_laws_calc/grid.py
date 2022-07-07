class Grid:
    """Classe contenant les informations nécessaires sur une grille de données
    Mère de la classe contenant les informations nécessaires sur la grille finale (grille d'échelles)
    """

    def __init__(self, N, L, c):
        """Initialisation of the mother class grid"""
        self.N = N  # Echantillonnage (Nombre de points)
        self.L = L  # Largeur totale réelle
        self.c = c  # real case length

    def check(self, mpi):
        """Display the principal parameter of the grid object"""
        mpi.pprint(f"Grid (x,y,z) :\n   - N : {self.N}\n   - L : {self.L}\n   - c : {self.c}\n")