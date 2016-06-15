import tensorflow as tf
import scipy.linalg as la
from core.TensorflowState import TensorflowState
from system.SystemParametersGeneral import SystemParametersGeneral
from math_functions.c_to_r_mat import CtoRMat
from runtime_functions.ConvergenceGeneral import ConvergenceGeneral
from runtime_functions.run_session import run_session
from math_functions.Get_state_index import Get_State_index


import random as rd
import time
from IPython import display

def Grape(H0,Hops,U,U0,total_time,steps,states_forbidden_list,states_concerned_list,convergence, reg_coeffs,D,Modulation,Interpolation,multi_mode , maxA ,use_gpu = True):
    
    
    
    class SystemParameters(SystemParametersGeneral):
        
        def __init__(self):
            SystemParametersGeneral.__init__(self,H0,Hops,U,U0,total_time,steps,states_forbidden_list,states_concerned_list,D,Modulation,Interpolation,multi_mode,maxA)
        
    sys_para = SystemParameters()
    if use_gpu:
        dev = '/gpu:0'
    else:
        dev = '/cpu:0'
            
    with tf.device(dev):
        tfs = TensorflowState(sys_para,use_gpu)
        graph = tfs.build_graph()
    
    class Convergence(ConvergenceGeneral):
        def __init__(self):
        # paramters
            self.sys_para = SystemParameters()
            self.Modulation = Modulation
            self.Interpolation = Interpolation

            self.rate = convergence['rate']
            self.update_step = convergence['update_step']
            self.conv_target = convergence['conv_target']
            self.max_iterations = convergence['max_iterations']

            self.learning_rate_decay = convergence['learning_rate_decay']


            reg_coeffs = {'alpha' : 0.01, 'z':0.01, 'dwdt':0.0001,'d2wdt2':0.001*0.0001, 'inter':100}
            self.reg_alpha_coeff = reg_coeffs['alpha']

            self.z_reg_alpha_coeff = reg_coeffs['z']

            self.dwdt_reg_alpha_coeff = reg_coeffs['dwdt']
            self.d2wdt2_reg_alpha_coeff = reg_coeffs['d2wdt2']

            self.inter_reg_alpha_coeff = reg_coeffs['inter']

            self.reset_convergence()
    conv = Convergence()
    
    try:
        run_session(tfs,graph,conv,sys_para)
    except KeyboardInterrupt:
        display.clear_output()
        
    
   