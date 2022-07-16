import numpy as np
import random

from .incgrid import IncGrid

def build_logregular_cylindrical_incremental_grid(original_grid, Nmax_scale, Nmax_list, kind="cls"):
    axis = ['lz','lperp','listperp']
    grid = {}
    
    # liste des échelles parallèles (indexes des coupes parallèles)
    grid["lz"] = np.unique(np.logspace(0, np.log10(original_grid.N[2]), Nmax_scale, endpoint=True, dtype=int))
    grid["lz"] = np.array(np.append([0], grid["lz"]))
    N_par = np.shape((grid["lz"]))[0]

    # liste des échelles perp (norme polaire)
    N_perp = np.min(original_grid.N[0:2])
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
    limX = np.min((grid["lperp"][-1], original_grid.N[0]))  # limite d'intéret en x
    limY = np.min((grid["lperp"][-1], original_grid.N[1]))  # limite d'intéret en y
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
                grid["listnorm"][r] = [x for x,_ in sorted(zip(grid["listnorm"][r], grid["listperp"][r]))][
                    :Nmax_list
                ]
                
            elif kind == "rdm":
                grid["listperp"][r] = random.sample(grid["listperp"][r], Nmax_list)
            grid["count"][r] = Nmax_list
    
    return IncGrid(original_grid, N=np.array([N_par, N_perp, np.max(grid["count"])], dtype=int), axis=axis, coords=grid, kind=kind)

def load(original_grid, Nmax_scale, Nmax_list, kind, **kargs):
    return build_logregular_cylindrical_incremental_grid(original_grid, Nmax_scale, Nmax_list, kind)

def build_listcoords(incgrid, nb_sec_by_dirr=1, **kargs):
        N = incgrid.spatial_grid.N
        
        list_prim = []
        for z in incgrid.coords["lz"]:
            for perp in range(len(incgrid.coords["lperp"])):
                for vect_perp in incgrid.coords["listperp"][perp]:
                    list_prim.append((vect_perp[0], vect_perp[1], z))
        set_prim = set(list_prim)
        list_prim = list(set_prim)
        
        list_sec = []
        for vect_prim in list_prim:
            for dirr in range(len(N)):
                for i in range(-nb_sec_by_dirr, nb_sec_by_dirr+1):
                    if i != 0:
                        vect = list(vect_prim)
                        vect[dirr] = (vect[dirr] + i) % N[dirr]
                        vect = (*vect,)
                        if not vect in set_prim:
                            list_sec.append(vect)
        list_sec = list(set(list_sec))
        return list_prim, list_sec, nb_sec_by_dirr

def reorganise_quantities(incgrid, output_grid, output_quantities, nb_sec_by_dirr=1):
    output = {}
    shape_scalar = incgrid.N
    shape_vector = [*shape_scalar, 3]
    shape_termdiv = [*shape_vector, 2*nb_sec_by_dirr]
    list_flux = []
    list_other = []
    for k in output_quantities:
        if k.startswith('flux'):
            output[k] = np.zeros(shape_vector)
            list_flux.append(k)
            output['term_div_'+k] = np.zeros(shape_termdiv)
        else: 
            output[k] = np.zeros(shape_scalar)
            list_other.append(k)
    for ind_z, z in enumerate(incgrid.coords["lz"]):
        for ind_perp in range(incgrid.N[1]):
            for ind_vect, vect_perp in enumerate(incgrid.coords["listperp"][ind_perp]):
                vect_prim = (vect_perp[0], vect_perp[1], z)
                prim_index = output_grid.coords['listprim'].index(vect_prim)
                for t in list_flux: 
                    output[t][ind_z,ind_perp,ind_vect] = [*output_quantities[t][0][prim_index]]
                for t in list_other:
                    output[t][ind_z,ind_perp,ind_vect] = output_quantities[t][0][prim_index]
                sec_list = list(np.arange(-nb_sec_by_dirr, nb_sec_by_dirr+1,1))
                sec_list.remove(0)
                for dirr in range(len(incgrid.spatial_grid.N)):
                    for ind_point, value in enumerate(sec_list):
                        vect = list(vect_prim)
                        vect[dirr] = (vect[dirr] + value) % incgrid.spatial_grid.N[dirr]
                        vect = tuple(vect)
                        loc = -1
                        try:
                            index = output_grid.coords['listprim'].index(vect)
                            loc = 0
                        except:
                            index = output_grid.coords['listsec'].index(vect)
                            loc = 1
                        for t in list_flux: 
                            output['term_div_'+t][ind_z,ind_perp,ind_vect,dirr,ind_point] = output_quantities[t][loc][index][dirr]
    return output
        


    
