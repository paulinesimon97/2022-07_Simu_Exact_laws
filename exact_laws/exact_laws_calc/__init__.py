import logging
import numpy as np
import configparser

from ..running_tools.mpi_wrap import Mpi
from ..running_tools.save_wrap import Save
from .datasets import load_from_standard_file, load
from .grids import load_incgrid_from_grid, load_listgrid_from_incgrid, div_on_incgrid
from .terms import TERMS
from .laws import LAWS

def initialise_original_dataset(input_filename,mpi):
    logging.info("Initialisation original data INIT")
    original_dataset, laws, terms = load_from_standard_file(input_filename)
    logging.info("Initialisation original data END")
    return original_dataset, laws, terms

def initialise_output_dataset(incremental_grid, original_dataset, laws, terms):
    logging.info(f"Initialisation output data INIT")
    params = {}
    params['laws'] = laws
    params['terms'], params['coeffs'] = list_terms_and_coeffs(laws, terms, original_dataset.params)
    params['state'] = {'index':0, 'list': 'prim'}
    grid = grid_prim_and_sec(params['terms'],incremental_grid)
    quantities = init_ouput_quantities(grid.coords['listprim'],grid.coords['listsec'],params['terms'])
    output_dataset = load(quantities=quantities, grid=grid, params=params)
    logging.info(f"Initialisation output data END")
    return output_dataset

def list_terms_and_coeffs(laws,terms,physical_params):
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
    return load_listgrid_from_incgrid(coord = incremental_grid.kind.split('_')[0],
                                      incgrid = incremental_grid,
                                      nb_sec_by_dirr = nb_sec_by_dirr)
    
def init_ouput_quantities(listprim, listsec, terms):
    output_quantities = {}
    Nprim = len(listprim)
    Nsec = len(listsec)
    for term in terms:
        if term.startswith('flux'):
            output_quantities[term] = [np.zeros((Nprim,3)),np.zeros((Nsec,3))]
        else:
            output_quantities[term] = [np.zeros((Nprim,3))]
    return output_quantities

def calc_terms(dataset, listprim, listsec, terms, output_quantities, saved_index, saved_list, mpi):
    # dataset contient au moins les quantités nécessaires au calcul
    # grid_prim = [Np][3] et grid_sec = [Ns][3] contiennent des vecteurs à calculer sous le format [x,y,z]
    # terms contient les noms des termes à calculer 
    # sortie : {[term_scalaire]:[Np],array([term_flux]):[array([Np,3]),array([Ns,3])]}    
    logging.info("INIT Calculation of the correlation functions")
    Ndat = dataset.grid.N
    terms_flux = []
    for term in terms:
        if term.startswith('flux'):
            terms_flux.append(term)
    if saved_list == 'prim':
        for index, vector in enumerate(listprim):
            if index % mpi.size*10 == 0 :
                mpi.barrier()
                logging(f'... End {saved_index + index} of listprim')
                yield output_quantities, saved_index + index, 'prim'
            if saved_index + index % mpi.size == mpi.rank:
                for term in terms:
                    output_quantities[term][0][saved_index + index] = TERMS[term].calc(vector,Ndat,**dataset.quantities) 
        saved_list = 'sec'
        saved_index=0
    mpi.barrier()
    if saved_list == 'sec':
        for index, vector in enumerate(listsec):
            if index % mpi.size*10 == 0 :
                mpi.barrier()
                logging(f'... End {saved_index + index} of listprim')
                yield output_quantities, saved_index + index, 'sec'
            if saved_index + index % mpi.size == mpi.rank:
                for term in terms_flux:
                    output_quantities[term][1][saved_index + index] = TERMS[term].calc(vector,Ndat,**dataset.quantities)
    mpi.barrier()
    logging.info("END Calculation of a correlation functions")
    return output_quantities, saved_index + index, 'end'

def reduction_output(quantities,mpi):
    logging.info(f"Reduction output data INIT")
    output = {}
    for k in quantities.keys():
        output[k] = mpi.reduce(quantities[k])
    logging.info(f"Reduction output data END")
    return output    

def calc_exact_laws_from_config(config_file,mpi=Mpi()):
    '''
    config_file example:
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
    
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # configure the potential parallelisation process (add old way params)
    mpi.set_nblayer(int(eval(config['RUN_PARAMS']['nblayer'])))
    mpi.set_bufnum(int(eval(config['RUN_PARAMS']['nbbuf'])))
    
    # configure the saving process (always valid way)
    save = Save()
    save.configure(eval(config['RUN_PARAMS']["save"]),mpi.time_deb,mpi.rank)
    
    # translate information useful for the computation
    input_filename = f"{config['INPUT_DATA']['path']}/{config['INPUT_DATA']['name']}.h5"
    output_filename = f"{config['OUTPUT_DATA']['path']}/{config['INPUT_DATA']['name']}_{config['OUTPUT_DATA']['name']}.h5"
    input_grid = {}
    input_grid["coord"] = config['GRID_PARAMS']["coord"]
    input_grid["kind"] = config['GRID_PARAMS']["kind"]
    input_grid["Nmax_scale"] = int(eval(config['GRID_PARAMS']["Nmax_scale"]))
    input_grid["Nmax_list"] = int(eval(config['GRID_PARAMS']["Nmax_list"]))
    
    # Init Original_dataset
    if save.already:
        original_dataset = save.download('data_origin')
    else:
        original_dataset, laws, terms = initialise_original_dataset(input_filename,mpi)
        if mpi.rank == 0:
            save.save(original_dataset,'data_origin')  
    original_dataset.check('original_dataset') 
    
    # Init Incremental_grid
    if save.already:
        incremental_grid = save.download('inc_grid')
    else: 
        incremental_grid = load_incgrid_from_grid(original_grid = original_dataset.grid,**input_grid)
        if mpi.rank == 0:
            save.save(incremental_grid,'inc_grid')

    # Init Output_dataset 
    if save.already:
        output_dataset = save.download('data_output',rank=f"{mpi.rank}")
    else:                 
        output_dataset = initialise_output_dataset(incremental_grid,original_dataset,laws,terms)
        save.save(output_dataset,'data_output',rank=f"{mpi.rank}")
    output_dataset.check('output_dataset')

    # ## CALCUL LOI EXACTE
    while output_dataset.params['state']['list'] != 'end':
        output_dataset.quantities, output_dataset.params['state']['index'], output_dataset.params['state']['list'] = calc_terms(
            dataset = original_dataset,
            listprim = output_dataset.grid.coords['listprim'][output_dataset.params['state']['index']:],
            listsec = output_dataset.grid.coords['listsec'][output_dataset.params['state']['index']:], 
            terms = output_dataset.params['terms'],
            output_quantities = output_dataset.quantities, 
            saved_index = output_dataset.params['state']['index'], 
            saved_list = output_dataset.params['state']['list'], 
            mpi = mpi)
        save.save(output_dataset,'data_output',rank=f"{mpi.rank}")        
    del(output_dataset.params['state'])
    
    # ## DIVERGENCE
    div_quantities = div_on_incgrid(incremental_grid,output_dataset)
    for k in div_quantities.keys():
        output_dataset.quantities[k] = [div_quantities[k]]
        
    # ## RASSEMBLEMENT
    output_dataset.quantities = reduction_output(output_dataset.quantities,mpi)
    save.save(output_dataset,'data_output_final',rank=f"{mpi.rank}")  
    
    
    # # ## Enregistrement données finales
    # logging.info(f"Record final result in {output_filename} INIT")
    # if mpi.rank == 0:
    #     output_dataset.record_to_h5file(output_filename)
    #     with h5.File(output_filename, "a") as file:
    #         file.create_group("param_origin")
    #         for k in original_dataset.params.keys():
    #             file["param_origin"].create_dataset(k, data=original_dataset.params[k])
    # logging.info(f"Record final result END")