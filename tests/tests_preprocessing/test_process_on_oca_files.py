import pytest
from .. import not_implemented_warning as NIW
from exact_laws.preprocessing.process_on_oca_files import *

class TestExtractSimuParamFromOCAFile:
    
    def test(self):
        NIW(extract_simu_param_from_OCA_file)

class TestExtractQuantitiesFromOCAFile:
    
    def test(self):
        NIW(extract_quantities_from_OCA_file)

class TestListQuantities:
    
    def test(self):
        NIW(list_quantities)

class TestFromOCAFilesToStandardH5File:
    
    def test(self):
        NIW(from_OCA_files_to_standard_h5_file)

class TestReformatOCAFiles:
    
    def test(self):
        NIW(reformat_oca_files)
    
