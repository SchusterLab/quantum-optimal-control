import numpy

def CtoRMat(M):
    return np.asarray(np.bmat([[M.real,-M.imag],[M.imag,M.real]]))
