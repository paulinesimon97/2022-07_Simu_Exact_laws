import logging
import numpy as np
import configparser

from .datasets import load_from_standard_file, load, record_incdataset_to_h5file
from .grids import load_incgrid_from_grid, load_listgrid_from_incgrid, div_on_incgrid, reorganise_quantities, reformat_grid_compatible_to_h5
from .terms import TERMS
from .laws import LAWS
from ..preprocessing import process_on_standard_h5_file


def initialise_original_dataset(input_filename):
    logging.info("INIT Initialisation original data")
    original_dataset, laws, terms = load_from_standard_file(input_filename)
    logging.info("END Initialisation original data ")
    return original_dataset, laws, terms


def initialise_output_dataset(incremental_grid, original_dataset, laws, terms):
    logging.info(f"INIT Initialisation output data ")
    params = {}
    params['laws'] = laws
    params['terms'], params['coeffs'] = list_terms_and_coeffs(laws, terms, original_dataset.params)
    params['state'] = {'index': 0, 'list': 'prim'}
    grid = grid_prim_and_sec(params['terms'], incremental_grid)
    quantities = init_ouput_quantities(grid.coords['listprim'], grid.coords['listsec'], params['terms'])
    output_dataset = load(quantities=quantities, grid=grid, params=params)
    logging.info(f"END Initialisation output data ")
    return output_dataset


def list_terms_and_coeffs(laws, terms, physical_params):
    list_terms = terms.copy()
    coeffs = {}
    for law in laws:
        terms_for_law, coeffs[law] = LAWS[law].terms_and_coeffs(physical_params)
        list_terms.append(*terms_for_law)
    return list(set(list_terms)), coeffs


def grid_prim_and_sec(terms, incremental_grid):
    nb_sec_by_dirr = 0
    for term in terms:
        if term.startswith('flux'):
            nb_sec_by_dirr = 1
    return load_listgrid_from_incgrid(coord=incremental_grid.kind.split('_')[0],
                                      incgrid=incremental_grid,
                                      nb_sec_by_dirr=nb_sec_by_dirr)


def init_ouput_quantities(listprim, listsec, terms):
    output_quantities = {}
    Nprim = len(listprim)
    Nsec = len(listsec)
    for term in terms:
        if term.startswith('flux'):
            output_quantities[term] = [np.zeros((Nprim, 3)), np.zeros((Nsec, 3))]
        else:
            output_quantities[term] = [np.zeros((Nprim))]
    return output_quantities


def calc_terms(dataset, output_dataset, run_config, save, backup_setp=10):
    # dataset contient au moins les quantités nécessaires au calcul
    # grid_prim = [Np][3] et grid_sec = [Ns][3] contiennent des vecteurs à calculer sous le format [x,y,z]
    # terms contient les noms des termes à calculer 
    # sortie : {[term_scalaire]:[Np],array([term_flux]):[array([Np,3]),array([Ns,3])]}

    terms = output_dataset.params['terms']
    output_quantities = output_dataset.quantities
    saved_list = output_dataset.params['state']['list']

    logging.info("INIT Calculation of terms")
    Ndat = dataset.grid.N

    if saved_list == 'prim':
        state_index = output_dataset.params['state']['index']
        vectors = [(index, vector) for index, vector in enumerate(output_dataset.grid.coords['listprim']) if
                   (index >= state_index) and ((index % run_config.size) == run_config.rank)]
        backup_ind = 0
        for index, vector in vectors:
            if backup_ind == 0: 
                logging.info(f'BEG {backup_ind+1}/{len(vectors)} of listprim attributed to proc {run_config.rank}')
            for term in terms:
                output_quantities[term][0][state_index + index] = TERMS[term].calc(vector, Ndat,
                                                                                   **dataset.quantities)
            if backup_ind % backup_setp == 0: 
                logging.info(f'... END {backup_ind+1}/{len(vectors)} of listprim attributed to proc {run_config.rank}')
                output_dataset.params['state']['index'] = index + 1
                save(output_dataset, 'data_output', rank=f"{run_config.rank}")
            backup_ind +=1
        output_dataset.params['state']['index'] = 0
        saved_list = 'sec'

    if saved_list == 'sec':
        terms_flux = list(filter(lambda e: e.startswith('flux'), terms))
        state_index = output_dataset.params['state']['index']
        vectors = [(index, vector) for index, vector in enumerate(output_dataset.grid.coords['listsec']) if
                   (index >= state_index) and ((index % run_config.size) == run_config.rank)]
        backup_ind = 0
        for index, vector in vectors:
            if backup_ind == 0: 
                logging.info(f'BEG {backup_ind+1}/{len(vectors)} of listprim attributed to proc {run_config.rank}')
            for term in terms_flux:
                output_quantities[term][1][state_index + index] = TERMS[term].calc(vector, Ndat,
                                                                                   **dataset.quantities)
            if backup_ind % backup_setp == 0: 
                logging.info(f'... END {backup_ind+1}/{len(vectors)} of listsec attributed to proc {run_config.rank}')
                output_dataset.params['state']['index'] = index + 1
                output_dataset.params['state']['list'] = 'sec'
                save(output_dataset, 'data_output', rank=f"{run_config.rank}")
            backup_ind +=1
    run_config.barrier()
    del (output_dataset.params['state'])
    logging.info("END Calculation of terms")


def reduction_output(quantities, run_config):
    logging.info(f"INIT Reduction output data")
    output = {}
    for k in quantities.keys():
        output[k] = run_config.reduce(quantities[k])
    logging.info(f"END Reduction output data")
    return output

def save_output_dataset_on_incgrid(output_filename, output_dataset, incremental_grid, original_dataset, coord):
    logging.info(f"INIT Record final result in {output_filename} ")
    nb_sec_by_dirr = 0
    for term in output_dataset.quantities:
        if term.startswith('flux'):
            nb_sec_by_dirr = 1
    incgrid_quantities = reorganise_quantities(coord =coord,  
                                               incgrid = incremental_grid, 
                                               output_grid = output_dataset.grid, 
                                               output_quantities = output_dataset.quantities,
                                               nb_sec_by_dirr = nb_sec_by_dirr)
    incgrid_h5 = reformat_grid_compatible_to_h5(coord=coord, incgrid= incremental_grid)
    params = original_dataset.params
    for k in output_dataset.params:
        params[k] = output_dataset.params[k]
    incgrid_dataset = load(quantities=incgrid_quantities, grid=incgrid_h5, params=params)
    record_incdataset_to_h5file(output_filename, incgrid_dataset)
    logging.info(f"END Record final result ")


def calc_exact_laws_from_config(config_file, run_config, backup):
    '''
    config_file example:
        [INPUT_DATA]
        path = ../../data_process/TestEff_CGL2/
        name = OCA_CGL2_cycle0_TestEff_bin2

        [OUTPUT_DATA]
        path = ./
        name = EL_logcyl40_cls100


        [RUN_PARAMS]
        config = NOP
        numbap = O
        nblayer = 8
        nbbuf = 4
        backup = None

        [GRID_PARAMS]
        Nmax_scale = 40
        Nmax_list = 100
        kind = cls
        coord = logcyl
    '''

    config = configparser.ConfigParser()
    config.read(config_file)

    # translate information useful for the computation
    input_filename = f"{config['INPUT_DATA']['path']}/{config['INPUT_DATA']['name']}.h5"
    output_filename = f"{config['OUTPUT_DATA']['path']}/{config['INPUT_DATA']['name']}_{config['OUTPUT_DATA']['name']}.h5"
    input_grid = {}
    input_grid["coord"] = config['GRID_PARAMS']["coord"]
    input_grid["kind"] = config['GRID_PARAMS']["kind"]
    input_grid["Nmax_scale"] = int(eval(config['GRID_PARAMS']["Nmax_scale"]))
    input_grid["Nmax_list"] = int(eval(config['GRID_PARAMS']["Nmax_list"]))
    
    message = (
        f"Begin calc exact law with config:"
        f"\n\t - input_file: {input_filename}"
        f"\n\t - output_file: {output_filename}"
    )
    logging.info(message)

    # Init Original_dataset
    if backup.already:
        original_dataset = backup.download('data_origin')
    else:
        original_dataset, laws, terms = initialise_original_dataset(input_filename)
        if run_config.rank == 0:
            backup.save(original_dataset, 'data_origin')
    original_dataset.check('original_dataset')

    # Init Incremental_grid
    if backup.already:
        incremental_grid = backup.download('inc_grid')
    else:
        incremental_grid = load_incgrid_from_grid(original_grid=original_dataset.grid, **input_grid)
        if run_config.rank == 0:
            backup.save(incremental_grid, 'inc_grid')
    incremental_grid.check('incremental_grid')

    # Init Output_dataset 
    if backup.already:
        output_dataset = backup.download('data_output', rank=f"{run_config.rank}")
    else:
        output_dataset = initialise_output_dataset(incremental_grid, original_dataset, laws, terms)
        backup.save(output_dataset, 'data_output', rank=f"{run_config.rank}")
    output_dataset.check('output_dataset')

    # ## CALCUL LOI EXACTE
    if 'state' in output_dataset.params:
        calc_terms(
            dataset=original_dataset,
            output_dataset=output_dataset,
            run_config=run_config,
            save=backup.save
        )
    backup.save(output_dataset, 'data_output', rank=f"{run_config.rank}")
    
    # ## DIVERGENCE
    div_quantities = div_on_incgrid(incremental_grid, output_dataset)
    for k in div_quantities.keys():
        output_dataset.quantities[k] = [div_quantities[k]]
    backup.save(output_dataset, 'data_output', rank=f"{run_config.rank}")

    # ## RASSEMBLEMENT
    output_dataset.quantities = reduction_output(output_dataset.quantities, run_config)
    if run_config.rank == 0: 
        backup.save(output_dataset, 'data_output_final')

    # ## Enregistrement données finales sur la grille incrémentale
    if run_config.rank == 0:
        save_output_dataset_on_incgrid(output_filename, output_dataset, incremental_grid, original_dataset, input_grid["coord"])
        process_on_standard_h5_file.check_file(output_filename)
    
