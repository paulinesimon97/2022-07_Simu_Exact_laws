import numpy as np
import random

from . import Grid

class Grid_scale_logcyl(Grid):
    """Classe contenant les informations nécessaires sur la grille finale (grille d'échelles)"""

    def __init__(self, N, L, c, Nmax_scale, Nmax_list, kind="cls"):
        """Initialisation of class grid_scale_logcyl
        directions : lperp (norme dans le plan polaire),z (coupe parallèle)
        N : list of maximum number of points in each dimension of the scale grid (argument = possible, attribut final = final)
        indexes selected according to point_min and N, then N is reevaluated"""
        grid = {}

        # liste des échelles parallèles (indexes des coupes parallèles)
        grid["lz"] = np.unique(np.logspace(0, np.log10(N[2]), Nmax_scale, endpoint=True, dtype=int))
        grid["lz"] = np.array(np.append([0], grid["lz"]))
        N_par = np.shape((grid["lz"]))[0]

        # liste des échelles perp (norme polaire)
        N_perp = np.min(N[0:2])
        grid["lperp"] = np.unique(np.logspace(0, np.log10(N_perp), Nmax_scale, endpoint=True, dtype=int))
        grid["lperp"] = np.array(np.append([0], grid["lperp"]))
        N_perp = np.shape((grid["lperp"]))[0]

        # init liste des points présent sur un même arc de rayon lperp +/- une marge
        grid["listperp"] = [[] for r in range(N_perp)]

        # init liste des écarts entre les points d'un arc et l'arc
        grid["listnorm"] = [[] for r in range(N_perp)]

        # init nombre de points présent sur un même arc
        grid["count"] = np.zeros((N_perp))

        # precision : marge maximale acceptée entre un point et l'arc associé
        precision = [
            np.min(((grid["lperp"][r + 1] - grid["lperp"][r]) / 2, (grid["lperp"][r] - grid["lperp"][r - 1]) / 2))
            for r in range(1, N_perp - 1)
        ]
        precision = np.append([(grid["lperp"][1] - grid["lperp"][0]) / 2], precision)  # pour le premier point
        precision = np.append(precision, [(grid["lperp"][-1] - grid["lperp"][-2]) / 2])  # pour le dernier point

        # distribution des points cartésiens suivant les arcs
        limX = np.min((grid["lperp"][-1], N[0]))  # limite d'intéret en x
        limY = np.min((grid["lperp"][-1], N[1]))  # limite d'intéret en y
        for x in range(-limX, limX + 1):  # boucle sur l'ensemble des points d'échelle d'intéret (y imbriqué dans x)
            for y in range(-limY, limY + 1):
                n = np.sqrt(x * x + y * y)  # norme du point
                for r in range(N_perp):  # boucle sur l'ensemble des normes sélectionnées
                    if n >= grid["lperp"][r] - precision[r] and n <= grid["lperp"][r] + precision[r]:
                        grid["listperp"][r].append(
                            [x, y]
                        )  # distribution des points en fonction de leur présence près d'un arc de rayon lperp
                        grid["listnorm"][r].append(np.abs(grid["lperp"][r] - n))
                        grid["count"][r] += 1
                        continue

        # réduction aléatoire (rdm) ou au plus près (cls) du nombre de points près d'un arc lperp à Nmax_list points si supérieur
        for r in range(N_perp):
            if grid["count"][r] > Nmax_list:
                if kind == "cls":
                    grid["listperp"][r] = [x for _, x in sorted(zip(grid["listnorm"][r], grid["listperp"][r]))][
                        :Nmax_list
                    ]
                elif kind == "rdm":
                    grid["listperp"][r] = random.sample(grid["listperp"][r], Nmax_list)
                grid["count"][r] = Nmax_list

        # creation de la grille
        Grid.__init__(self, np.array([N_par, N_perp, np.max(grid["count"])], dtype=int), L, c)
        self.grid = grid
        # N tel que [nombre de coupes, nombre de rayons dans le plan polaire, nombre maximal de direction dans le plan polaire par rayon]