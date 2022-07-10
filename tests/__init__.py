import logging
import pytest
import warnings
    
def not_implemented_warning(fct_to_test):
    warnings.warn(f"No test about {fct_to_test.__name__} implemented yet.")
    
    
