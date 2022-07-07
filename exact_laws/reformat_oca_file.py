from exact_laws.preprocessing.reformat_OCA_file import main
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config-file", help="config file", default='input_process.txt')
args = parser.parse_args()

if __name__ == "__main__":
    main(config_file=args.config_file)
