import numpy as np

def CtoRMat(M):
    return np.asarray(np.bmat([[M.real,-M.imag],[M.imag,M.real]]))

def CtoRVec(V):
    new_v =[]
    new_v.append(V.real)
    new_v.append(V.imag)
    return np.reshape(new_v,[2*len(V)])
        
