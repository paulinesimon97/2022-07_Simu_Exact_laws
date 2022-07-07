#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Pauline SIMON
# dernière modification : 27/06/2022
version = "27/06/2022"

import sys
import numpy as np
import h5py as h5
from datetime import datetime
from contextlib import redirect_stdout

import preprocessing.reformating_OCA_file as reformat
import preprocessing.scan_file as sf

if __name__ == "__main__":
    now = datetime.today()
    with open(f"output_process_{now.strftime('%d%m%Y_%H%M')}.txt", "w") as f:
        with redirect_stdout(f):
            print(
                f"Run of Data_process.py version {version} the {now.strftime('%d/%m/%Y')} at {now.strftime('%H:%M')}.\n"
            )
            sys.stdout.flush()
            inputdic = reformat.inputfile_to_dict("input_process.txt")
            binning = inputdic["bin"]
            inputdic["bin"] = 1
            file_process = reformat.data_process_OCA(inputdic)
            if binning != 1:
                inputdic["bin"] = binning
                sf.data_binning(file_process, inputdic)
            print(f"End the {datetime.today().strftime('%d/%m/%Y')} at {datetime.today().strftime('%H:%M')}")