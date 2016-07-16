import tensorflow as tf
import numpy as np
import scipy.linalg as la
from core.TensorflowState import TensorflowState
from system.SystemParametersGeneral import SystemParametersGeneral
from math_functions.c_to_r_mat import CtoRMat
from runtime_functions.ConvergenceGeneral import ConvergenceGeneral
from runtime_functions.run_session import run_session
from math_functions.Get_state_index import Get_State_index
from main_grape.Grape import Grape

import random as rd
import time
from IPython import display

def Evolve(H0,Hops,U0,total_time,steps,psi0,initial_guess,U=None,draw=None,Hnames = None):
    
    flag = True
    if Hnames == None:
        for ii in range (len(Hops)):
            Hnames.append(str(ii))
    
    if U == None:
        flag = False
        U = U0
    convergence = {'rate':0, 'update_step':1, 'max_iterations':0,\
               'conv_target':1e-8,'learning_rate_decay':1}
    
    Grape(H0,Hops,Hnames,U,U0,total_time,steps,psi0,convergence, draw= draw, initial_guess = initial_guess, evolve = True, evolve_error = flag,Unitary_error = 1e-20)
