import logging
from .running_tools import mpi_wrap

mpi = mpi_wrap.Mpi()
mpi.configure_log('calc_exact_law')

from exact_laws.exact_laws_calc import main
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config-file", help="config file", default='input_calc.txt')
parser.add_argument("-q", "--list-laws", help="List available laws", action="store_true")
args = parser.parse_args()

version = "09/07/2022"

if __name__ == "__main__":
    
    if args.list_quantities:
        from exact_laws.exact_laws_calc.laws import  LAWS
        print(list(LAWS.keys()))
        exit(0)
        
    logging.info(f"Run of {__file__} version {version}\n")
    main(config_file=args.config_file,mpi=mpi)
    logging.info(f"Exit")