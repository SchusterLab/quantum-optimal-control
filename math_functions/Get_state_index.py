import numpy as np

def Get_State_index(bareindex,dressed):
    if len(dressed) > 0:
        return dressed.index(bareindex)
    else:
        return bareindex