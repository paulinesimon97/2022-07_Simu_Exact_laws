import logging
import numpy as np
import numexpr as ne
import h5py as h5
import configparser
from functools import reduce

from ..running_tools.mpi_wrap import Mpi
from ..running_tools.save_wrap import Save
from .datasets import Dataset, read_standard_file 
from .datasets.output_calc_laws import OutputCalcLaws
from .grids import Grid
from .laws import LAWS
from .terms import TERMS
from ..mathematical_tools.derivation import cdiff
from .values_useful_for_calc_at_scale import ValuesUsefulForCalcAtScale
                    
def calc_terms_old_way(original_dataset,output_dataset,mpi,save):
    logging.info("Calculation output data INIT")
    useful_at_scale = ValuesUsefulForCalcAtScale(
        original_dataset, mpi
    )  # Insertion des données initiales dans le dictionnaire servant pour le calcul à vecteur fixé
    useful_at_scale.check(mpi)
    mpi.barrier()
    
    logging.info("Calculation output data BEG")
    # vecteur servant à obtenir le cube translaté en z qui sera distribuée aux processurs si parallélisation
    vector_dir0 = [0, 0, 0]  
     # vecteur servant à obtenir les translations du cube dans le plan polaire
    vector = [0, 0, 0]  

    # Test pour vérifier s'il est nécessaire de faire une divergence locale et donc de calculer les points autour du point d'intéret
    div = False
    for k in output_dataset.quantities.keys():
        if "term_div" in k:
            div = True
            continue

    state = output_dataset.state

    # Boucle sur les plans
    for ind_dir0 in range(state, output_dataset.grid.N[0]):
        logging.info(f"Calculation output data state {ind_dir0} INIT")

        # Insertion des données déplacées suivant z dans le dictionnaire servant pour le calcul à vecteur fixé
        vector_dir0[2] = output_dataset.grid.grid["lz"][ind_dir0]
        useful_at_scale.set_data_dir0(original_dataset, vector_dir0, mpi)
        mpi.barrier()

        i = -1
        # Boucle sur les rayons dans les plans polaires
        for lperp in range(output_dataset.grid.N[1]):

            # Distribution des calculs 1 vecteur => 1 processeur si parallélisation
            for vect in range(len(output_dataset.grid.grid["listperp"][lperp])):
                i += 1
                if mpi.group_rank == i % mpi.group_size:
                    # Insertion des données déplacées dans le plan polaire dans le dictionnaire servant pour le calcul à vecteur fixé
                    vector[0] = output_dataset.grid.grid["listperp"][lperp][vect][0]
                    vector[1] = output_dataset.grid.grid["listperp"][lperp][vect][1]
                    useful_at_scale.set_data_prim(vector)
                    # Calcul à vecteur fixé
                    indices = [ind_dir0, lperp, vect]
                    
                    for t in output_dataset.params['terms']:
                        output_dataset.quantities[t][indices[0], indices[1], indices[2]] = TERMS[t].calc(values=useful_at_scale.datadic)
                        
                    for t in output_dataset.params['terms']:
                        if t.startswith("flux"):
                            output_dataset.quantities[f"term_div_{t}"][
                                indices[0], indices[1], indices[2], indices[3], indices[4]
                            ] = TERMS[t].calc(values=useful_at_scale.datadic)[indices[3]]

                    if div == True:
                        for d in range(3):
                            for p in [-1, 1]:
                                vector_div = np.copy(vector)
                                vector_div[d] = vector[d] + p
                                useful_at_scale.set_data_prim(vector_div)
                                if p == -1:
                                    p = 0
                                indices = [ind_dir0, lperp, vect, d, p]
                            
                                for t in output_dataset.params['terms']:
                                    output_dataset.quantities[t][indices[0], indices[1], indices[2]] = TERMS[t].calc(values=useful_at_scale.datadic)
                            
                                for t in output_dataset.params['terms']:
                                    if t.startswith("flux"):
                                        output_dataset.quantities[f"term_div_{t}"][
                                            indices[0], indices[1], indices[2], indices[3], indices[4]
                                        ] = TERMS[t].calc(values=useful_at_scale.datadic)[indices[3]]
        mpi.barrier()
        logging.info(f"Calculation output data state {ind_dir0} END")
        output_dataset.state += 1

        save.save(output_dataset,'data_output',rank=f"{mpi.rank}",state=f'state {ind_dir0}')

    logging.info(f"Filtre output data INIT")
    for k in output_dataset.quantities.keys():
        output_dataset.quantities[k] = np.where(
            np.abs(output_dataset.quantities[k]) < 1e-15, 0 * output_dataset.quantities[k], output_dataset.quantities[k]
        )
    logging.info(f"Filtre output data END")

    logging.info(f"Reduction output data INIT")
    for k in output_dataset.quantities.keys():
        if mpi.size != 1:
            total = mpi.comm.reduce(output_dataset.quantities[k], op=mpi.op, root=0)
        if mpi.rank == 0:
            denom = reduce(lambda x, y: x * y, original_dataset.grid.N)
            output_dataset.quantities[k] = total / denom
    logging.info(f"Reduction output data END")

    logging.info(f"Divergence output data INIT")
    if mpi.rank == 0:
        for k in output_dataset.quantities.keys():
            if k.startswith("div_"):
                case_vec = output_dataset.grid.c
                local_dict = {
                    "fx": [
                        output_dataset.quantities["term_" + k][:, :, :, 0, 0],
                        output_dataset.quantities["term_" + k][:, :, :, 0, -1],
                    ],
                    "fy": [
                        output_dataset.quantities["term_" + k][:, :, :, 1, 0],
                        output_dataset.quantities["term_" + k][:, :, :, 1, -1],
                    ],
                    "fz": [
                        output_dataset.quantities["term_" + k][:, :, :, 2, 0],
                        output_dataset.quantities["term_" + k][:, :, :, 2, -1],
                    ],
                }
                local_dict["dx"] = cdiff(
                    local_dict["fx"], length_case=case_vec[0], precision=2, period=False, point=True
                )
                local_dict["dy"] = cdiff(
                    local_dict["fy"], length_case=case_vec[1], precision=2, period=False, point=True
                )
                local_dict["dz"] = cdiff(
                    local_dict["fz"], length_case=case_vec[2], precision=2, period=False, point=True
                )
                output_dataset.quantities[k] = ne.evaluate(f"dx+dy+dz", local_dict=local_dict)
    logging.info(f"Divergence output data END")

    # for k in data_output.datadic.keys():
    #    if mpi.rank == 0: print(k,' ',np.min(data_output.datadic[k]),' ',np.max(data_output.datadic[k]))
    logging.info("Calculation output data END")
    
    
def calc_terms(dataset,grid_prim,grid_sec,terms):
    # dataset contient au moins les quantités nécessaires au calcul
    # grid_prim = [Np][3] et grid_sec = [Ns][3] contiennent des vecteurs à calculer sous le format [x,y,z]
    # terms contient les noms des termes à calculer 
    # sortie : {[term_scalaire]:[Np],array([term_flux]):[array([Np,3]),array([Ns,3])]}    
    logging.info("INIT Calculation of the correlation functions")
    output = {}
    terms_flux = []
    for term in terms:
        if term.startswith('flux'):
            terms_flux.append(term)
            output[term] = [[],[]]
        else:
            output[term] = []
    for vector in grid_prim:
        for term in terms:
            output[term].append(TERMS[term].calc(vector,dataset.grid.N,**dataset.quantities))
    for vector in grid_sec:
        for term in terms_flux:
            output[term].append(TERMS[term].calc(vector,dataset.grid.N,**dataset.quantities))
    logging.info("END Calculation of a correlation functions")
    return output

            
        
                
            
    
    
    
    