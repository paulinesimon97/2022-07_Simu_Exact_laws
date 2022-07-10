import logging
import argparse
from exact_laws.running_tools import mpi_wrap
from exact_laws.exact_laws_calc import calc_exact_laws_from_config

mpi = mpi_wrap.Mpi()
mpi.configure_log('calc_exact_law')

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config-file", help="config file", default='example_input_calc.txt')
parser.add_argument("-q", "--list-laws", help="List available laws", action="store_true")
args = parser.parse_args()

version = "09/07/2022"

if __name__ == "__main__":
    
    if args.list_laws:
        from exact_laws.exact_laws_calc.laws import  LAWS
        print(list(LAWS.keys()))
        exit(0)
        
    logging.info(f"Run of {__file__} version {version}\n")
    calc_exact_laws_from_config(config_file=args.config_file,mpi=mpi)
    logging.info(f"Exit")
