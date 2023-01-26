import logging
import numpy as np
import os
import h5py as h5
import shutil

from .datasets import load, record_incdataset_to_h5file
from .grids import load_outputgrid_from_incgrid
from .terms import TERMS
from .laws import LAWS
from .grids import div_on_incgrid, reorganise_quantities, reformat_grid_compatible_to_h5

def initialise_output_dataset(incremental_grid, original_dataset, laws, terms):
    logging.info("INIT Initialisation output data ")
    params = {}
    params['laws'] = laws
    params['terms'], params['coeffs'] = list_terms_and_coeffs(laws, terms, original_dataset.params)
    params['state'] = {'nb_term_done': 0, 'nb_term_rec': 0, 'nb_terms': len(params['terms'])}
    grid = load_outputgrid_from_incgrid(coord=incremental_grid.kind.split('_')[0], incgrid=incremental_grid)
    quantities = {}
    output_dataset = load(quantities=quantities, grid=grid, params=params)
    logging.info("END Initialisation output data ")
    return output_dataset

def list_terms_and_coeffs(laws, terms, physical_params):
    list_terms = terms.copy()
    coeffs = {}
    for law in laws:
        terms_for_law, coeffs[law] = LAWS[law].terms_and_coeffs(physical_params)
        list_terms += terms_for_law
    return list(set(list_terms)), coeffs

def calc_term(dataset, output_dataset):
    # dataset contient au moins les quantités nécessaires au calcul
    # grid_prim = [Np][3] et grid_sec = [Ns][3] contiennent des vecteurs à calculer sous le format [x,y,z]
    # terms contient les noms des termes à calculer 
    # sortie : {[term_scalaire]:[Np],array([term_flux]):[array([Np,3]),array([Ns,3])]}

    terms = output_dataset.params['terms']
    output_quantities = output_dataset.quantities
    ind_term = output_dataset.params['state']['nb_term_done']

    logging.info(f"INIT Calculation of term {terms[ind_term]}")
    output_quantities[terms[ind_term]] = TERMS[terms[ind_term]].calc_fourier(**dataset.quantities)
    logging.info(f"END Calculation of term {terms[ind_term]}")
    
def reduction_output(quantities, run_config):
    logging.info("INIT Reduction output data")
    output = {}
    for k in quantities.keys():
        output[k] = run_config.reduce(quantities[k])
    logging.info("END Reduction output data")
    return output
    
def save_output_dataset_on_incgrid(output_filename, output_dataset, incremental_grid, original_dataset, coord):
    incgrid_quantities = reorganise_quantities(coord =coord,  
                                                incgrid = incremental_grid, 
                                                output_grid = output_dataset.grid, 
                                                output_quantities = output_dataset.quantities)
    if os.path.isfile(output_filename) :
        logging.info(f"INIT Record datasets in {output_filename} ")
        with h5.File(output_filename, "a") as f:
            for key in incgrid_quantities:
                if key not in f.keys():
                    f.create_dataset(key, data=incgrid_quantities[key])
                del(output_dataset.quantities[key])
            del(f['params']['state']['nb_term_done'],f['params']['state']['nb_term_rec'])
            f['params']['state'].create_dataset('nb_term_done', data=output_dataset.params['state']['nb_term_done'])
            f['params']['state'].create_dataset('nb_term_rec', data=output_dataset.params['state']['nb_term_rec'] +1)
    else : 
        logging.info(f"INIT Record params and grid in {output_filename} ")
        incgrid_h5 = reformat_grid_compatible_to_h5(coord=coord, incgrid= incremental_grid)
        params = original_dataset.params
        for k in output_dataset.params:
            params[k] = output_dataset.params[k]
        incgrid_dataset = load(quantities=incgrid_quantities, grid=incgrid_h5, params=params)
        record_incdataset_to_h5file(output_filename, incgrid_dataset)
    logging.info("END Record ")
   
def apply_method(original_dataset, incremental_grid, coord, laws, terms, output_filename, run_config, backup): 
    # Init Output_dataset 
    if backup.already:
        output_dataset = backup.download('data_output', rank=f"{run_config.rank}")
    else:
        output_dataset = initialise_output_dataset(incremental_grid, original_dataset, laws, terms)
        backup.save(output_dataset, 'data_output', rank=f"{run_config.rank}")
    output_dataset.check('output_dataset')
    
    if run_config.rank == 0:
        save_output_dataset_on_incgrid(output_filename, output_dataset, incremental_grid, original_dataset, coord)
    
    output_dataset.params['state']['nb_term_rec'] = output_dataset.params['state']['nb_term_done']
    
    logging.info("INIT creat temporary .h5 file")
    if run_config.rank == 0:
        shutil.copy2(output_filename,output_filename+'temp')
    logging.info("END creat temporary .h5 file")
    
    # ## CALCUL LOI EXACTE
    while output_dataset.params['state']['nb_term_rec'] != output_dataset.params['state']['nb_terms'] :
        if output_dataset.params['state']['nb_term_rec']%5 == 0 :
            logging.info("INIT update .h5 file and backup")
            if run_config.rank == 0:
                shutil.copy2(output_filename+'temp',output_filename)
                backup.save(output_dataset, 'data_output', rank=f"{run_config.rank}")
            logging.info("END update .h5 file and backup")
    
        calc_term(
            dataset=original_dataset,
            output_dataset=output_dataset
        )    
        # ## DIVERGENCE
        div_quantities = div_on_incgrid(coord, incremental_grid, output_dataset)
        for k in div_quantities.keys():
            output_dataset.quantities[k] = div_quantities[k]
        
        output_dataset.params['state']['nb_term_done'] += 1
        
        # ## ENREGISTREMENT
        if run_config.rank == 0:
            save_output_dataset_on_incgrid(output_filename+'temp', output_dataset, incremental_grid, original_dataset, coord)
        
        output_dataset.params['state']['nb_term_rec'] += 1
    
    logging.info("INIT final update .h5 file")
    if run_config.rank == 0:
        shutil.copy2(output_filename+'temp',output_filename)
    logging.info("END final update .h5 file")
        
        
        
