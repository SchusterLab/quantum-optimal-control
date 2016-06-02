import tensorflow as tf
import scipy.linalg as la
from core.TensorflowState import TensorflowState
from system.SystemParametersGeneral import SystemParametersGeneral
from math_functions.c_to_r_mat import CtoRMat
from runtime_functions.ConvergenceGeneral import ConvergenceGeneral
from runtime_functions.run_session import run_session
from math_functions.Get_state_index import Get_State_index

import numpy as np
import random as rd
import time


'''
qubit_pi
two_modes_cnot
'''
Modulation = True
Interpolation = True
D = True
simulation_system = "qubit_pi"


if simulation_system == "qubit_pi":
    class SystemParameters(SystemParametersGeneral):
        
        def __init__(self):
            total_time = 5.0
            SystemParametersGeneral.__init__(self,total_time, Modulation, Interpolation, D)
            self.init_states()
            
            
        def init_states(self):
            self.g00=Get_State_index(0,self.dressed)
            self.e00=Get_State_index(self.mode_state_num**2,self.dressed)
            # concerned states
            self.states_concerned_list = [0,self.mode_state_num**2]
            self.init_vectors()
            
            # Initialize initial and target states in numpy vector
            
            self.init_operators()
            if D:
                Ct_c= np.identity(self.state_num) - np.outer(self.v_c[:,self.g00], np.conjugate((self.v_c[:,self.g00])))
                Ct_c= Ct_c - np.outer(self.v_c[:,self.e00], np.conjugate((self.v_c[:,self.e00])))
                Ct_c= Ct_c + np.outer(self.v_c[:,self.e00], np.conjugate((self.v_c[:,self.g00])))
                Ct_c= Ct_c + np.outer(self.v_c[:,self.g00], np.conjugate((self.v_c[:,self.e00])))
            else:
                Ut_c = la.expm((0-1j)*self.total_time*self.H0_c)
                Ct_c = Ut_c.copy()

                Ct_c[self.e00,self.g00] = 1
                Ct_c[self.g00,self.e00] = 1
                Ct_c[self.g00,self.g00] = 0
                Ct_c[self.e00,self.e00] = 0
            

            self.initial_state = self.identity
            self.target_state = CtoRMat(Ct_c)



if simulation_system == "two_modes_cnot":
    class SystemParameters(SystemParametersGeneral):
        
        def __init__(self):
            total_time = 30
            SystemParametersGeneral.__init__(self,total_time, Modulation, Interpolation, D)
            self.init_states()
            
            
        def init_states(self):
            # concerned states
            self.g00=Get_State_index(0,self.dressed)
            self.g01=Get_State_index(1,self.dressed)
            self.g10=Get_State_index(self.mode_state_num,self.dressed)
            self.g11=Get_State_index(self.mode_state_num+1,self.dressed)
            self.states_concerned_list = [0,1,self.mode_state_num,self.mode_state_num+1] #g00,g01,g10,g11
            self.init_vectors()
            self.init_operators()
            
            # Initialize initial and target states in numpy vector           
            if D:
                Ct_c= np.identity(self.state_num) - np.outer(self.v_c[:,self.g11], np.conjugate((self.v_c[:,self.g11])))
                Ct_c= Ct_c - np.outer(self.v_c[:,self.g10], np.conjugate((self.v_c[:,self.g10])))
                Ct_c= Ct_c + np.outer(self.v_c[:,self.g11], np.conjugate((self.v_c[:,self.g10])))
                Ct_c= Ct_c + np.outer(self.v_c[:,self.g10], np.conjugate((self.v_c[:,self.g11])))
            else:
                Ut_c = la.expm((0-1j)*self.total_time*self.H0_c)
                Ct_c = Ut_c.copy()
            
                Ct_c[self.g11,self.g10] = 1
                Ct_c[self.g10,self.g11] = 1
                Ct_c[self.g11,self.g11] = 0
                Ct_c[self.g10,self.g10] = 0
            
            self.initial_state = self.identity
            self.target_state = CtoRMat(Ct_c)
            


sys_para = SystemParameters()

with tf.device('/cpu:0'):
    tfs = TensorflowState(sys_para,use_gpu = False)
    graph = tfs.build_graph()


class Convergence(ConvergenceGeneral):

        
    def __init__(self):
        # paramters
        self.sys_para = SystemParameters()
        self.Modulation = Modulation
        self.Interpolation = Interpolation
        self.rate = 0.01
        self.update_step = 10
        self.conv_target = 1e-8
        self.max_iterations = 5000
        
        self.learning_rate_decay = self.max_iterations/2
        
        self.reg_alpha_coeff = 0.01
        
        self.z_reg_alpha_coeff = 0.01
        
        self.dwdt_reg_alpha_coeff = 0.0001
        self.d2wdt2_reg_alpha_coeff = 0.001*0.0001
        
        self.inter_reg_alpha_coeff = 100.0
        
        self.reset_convergence()
conv = Convergence()


run_session(tfs,graph,conv,sys_para,show_plots=False)
