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

'''

[INPUT_DATA]
path = ../../data_process/TestEff_CGL2/
name = OCA_CGL2_cycle0_TestEff_bin2

[OUTPUT_DATA]
path = ./
name = EL_logcyl40_cls100


[RUN_PARAMS]
nblayer = 8
nbbuf = 4
save = None

[GRID_PARAMS]
Nmax_scale = 40
Nmax_list = 100
kind = cls
coord = logcyl

'''

def list_terms_and_coeffs(laws,physical_params):
    terms = []
    coeffs = {}
    for law in laws:
        terms_for_law, coeffs[law] = LAWS[law].terms_and_coeffs(physical_params)
        terms += terms_for_law
    return list(set(terms)), coeffs

def fill_output_at_scale(output_dataset, indices, values, div=False):
        """Remplissage des cubes au point d'indices 'indices' à partir des dictionnaires funcdic et argdic
        si div = False : remplissage des cubes flux et sources
        si div = True : remplissage des cubes term_div
        """
        if div == False:
            for t in output_dataset.params['terms']:
                output_dataset.quantities[t][indices[0], indices[1], indices[2]] = TERMS[t].calc(values=values.datadic)
        else:
            for t in output_dataset.params['terms']:
                if t.startswith("flux"):
                    output_dataset.quantities[f"term_div_{t}"][
                        indices[0], indices[1], indices[2], indices[3], indices[4]
                    ] = TERMS[t].calc(values=values.datadic)[indices[3]]


def initialise_original_dataset(input_filename,mpi):
    logging.info("Initialisation original data INIT")
    original_dataset = Dataset()
    laws = 0
    grid = 0
    if mpi.rank == 0:
        original_dataset.quantities, original_dataset.params, laws, grid = read_standard_file(input_filename)
    mpi.barrier
    original_dataset.params = mpi.bcast(original_dataset.params)
    laws = mpi.bcast(laws)
    grid = mpi.bcast(grid)
    original_dataset.grid = Grid.from_dict(grid)
    logging.info("Initialisation original data END")
    return original_dataset, laws

def initialise_output_dataset(grid,original_dataset):
    logging.info(f"Initialisation output data INIT")
    params = {}
    params['laws'] = laws
    params['terms'], coeffs = list_terms_and_coeffs(laws, original_dataset.params)
    output_dataset = OutputCalcLaws(params, coeffs, grid, original_dataset.grid)
    logging.info(f"Initialisation output data END")
    return output_dataset

def calc_exact_laws(original_dataset,output_dataset,mpi,save):
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
                    fill_output_at_scale(output_dataset, indices, useful_at_scale)

                    if div == True:
                        for d in range(3):
                            for p in [-1, 1]:
                                vector_div = np.copy(vector)
                                vector_div[d] = vector[d] + p
                                useful_at_scale.set_data_prim(vector_div)
                                if p == -1:
                                    p = 0
                                indices = [ind_dir0, lperp, vect, d, p]
                                fill_output_at_scale(output_dataset, indices, useful_at_scale, div=True)
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

def calc_exact_laws_from_config(config_file,mpi=Mpi()):
    
    config = configparser.ConfigParser()
    config.read(config_file)
    mpi.set_nblayer(int(eval(config['RUN_PARAMS']['nblayer'])))
    mpi.set_bufnum(int(eval(config['RUN_PARAMS']['nbbuf'])))
    save = Save()
    save.configure(eval(config['RUN_PARAMS']["save"]),mpi.time_deb,mpi.rank)
    input_filename = f"{config['INPUT_DATA']['path']}/{config['INPUT_DATA']['name']}.h5"
    output_filename = f"{config['OUTPUT_DATA']['path']}/{config['INPUT_DATA']['name']}_{config['OUTPUT_DATA']['name']}.h5"
    input_grid = {}
    input_grid["coord"] = config['GRID_PARAMS']["coord"]
    input_grid["kind"] = config['GRID_PARAMS']["kind"]
    input_grid["Nmax_scale"] = int(eval(config['GRID_PARAMS']["Nmax_scale"]))
    input_grid["Nmax_list"] = int(eval(config['GRID_PARAMS']["Nmax_list"]))
    
    # Init Original_dataset
    if save.already:
        original_dataset = save.download('data_origin',rank=f"{0 + 1 * (mpi.rank != 0)}")
    else:
        initialise_original_dataset(input_filename,mpi)
        if mpi.rank in [0,1]:
            save.save(original_dataset,'data_origin',rank=f"{mpi.rank}")  
    original_dataset.check('original_dataset')
    mpi.barrier    

    # Init Output_dataset 
    if save.already:
        output_dataset = save.download('data_output',rank=f"{mpi.rank}")
    else:                 
        initialise_output_dataset(input_grid,original_dataset)
        save.save(output_dataset,'data_output',rank=f"{mpi.rank}")
    output_dataset.check('output_dataset')
    mpi.barrier 

    # ## CALCUL LOI EXACTE
    calc_exact_laws(original_dataset,output_dataset,mpi,save)

    # ## Enregistrement données finales
    logging.info(f"Record final result in {output_filename} INIT")
    if mpi.rank == 0:
        output_dataset.record_to_h5file(output_filename)
        with h5.File(output_filename, "a") as file:
            file.create_group("param_origin")
            for k in original_dataset.params.keys():
                file["param_origin"].create_dataset(k, data=original_dataset.params[k])
    logging.info(f"Record final result END")
