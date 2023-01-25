import scipy as scp
import numpy as np

def fft(tab,way='scipy'): 
    if 'scipy' in way:
        return scp.fft.rfftn(tab)
    elif 'numpy' in way:
        return np.fft.rfftn(tab)
    
def ifft(tab,way='scipy'): 
    if 'scipy' in way:
        return scp.fft.irfftn(tab)
    elif 'numpy' in way:
        return np.fft.irfftn(tab)

