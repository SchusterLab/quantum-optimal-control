import tensorflow as tf
import numpy as np
import scipy.linalg as la
from core.TensorflowState import TensorflowState
from system.SystemParametersGeneral import SystemParametersGeneral
from runtime_functions.ConvergenceGeneral import ConvergenceGeneral
from runtime_functions.run_session import run_session



import random as rd
import time
from IPython import display

from helper_functions.datamanagement import H5File
import os


def Grape(H0,Hops,Hnames,U,total_time,steps,states_concerned_list,convergence = None, U0= None, penalty_coeffs = None,multi_mode = None, maxA = None ,use_gpu= True, draw= None, forbidden = None, initial_guess = None, evolve_only = False, evolve_error = False,show_plots = True, H_time_scales = None, unitary_error=1e-4, method = 'Adam',state_transfer = False, switch = True,no_scaling = False, freq_unit = 'GHz', limit_dc = None, limit_dc_segment_num= 1, gate = None, forbid_dressed = True, save = True, data_path = None):
    
    
    if freq_unit == 'GHz':
        time_unit = 'ns'
    elif freq_unit == 'MHz':
        time_unit = 'micro s'
    elif freq_unit == 'KHz':
        time_unit = 'ms'
    elif freq_unit == 'Hz':
        time_unit = 's'
        
    file_path = None
    
    if save:
        if gate == None:
            raise ValueError('Grape function input: gate, is not specified.')

        if data_path == None:
            raise ValueError('Grape function input: data_path, is not specified.')

        #file_name = gate + ' total_time: ' + str(total_time)+', steps: ' + str(steps)+', size: '+str(len(H0))

        file_name = gate

        file_num = 0
        while (os.path.exists(os.path.join(data_path,str(file_num).zfill(5) + "_"+ file_name+".h5"))):
            file_num+=1

        file_name = str(file_num).zfill(5) + "_"+ file_name+ ".h5"

        file_path = os.path.join(data_path,file_name)

        with H5File(file_path) as hf:
            hf.add('H0',data=H0)
            hf.add('Hops',data=Hops)
            hf.add('Hnames',data=Hnames)
            hf.add('U',data=U)
            hf.add('total_time', data=total_time)
            hf.add('steps', data=steps)
            hf.add('states_concerned_list', data=states_concerned_list)
            hf.save_dict(convergence,'convergence')
    
    if U0 == None:
        U0 = np.identity(len(H0))
    if convergence == None:
        convergence = {'rate':0.01, 'update_step':100, 'max_iterations':5000,'conv_target':1e-8,'learning_rate_decay':2500}
    
    if penalty_coeffs == None:
        if evolve_only:
            penalty_coeffs = {'envelope' : 0, 'dc':0, 'dwdt':0,'d2wdt2':0, 'forbidden':0}
        else:
            penalty_coeffs = {'envelope' : 0.01, 'dc':0.01, 'dwdt':0.001,'d2wdt2':0.001*0.0001, 'forbidden':100}
        # envelope: to make it close to a Gaussian envelope
        # dc: to limit DC offset of z pulses 
        # dwdt: to limit pulse first derivative
        # d2wdt2: to limit second derivatives
        # forbidden: to penalize forbidden states
       
        
    if maxA == None:
        if initial_guess == None:
            maxAmp = 4*np.ones(len(Hops))
        else:
            maxAmp = 1.5*np.max(np.abs(initial_guess))*np.ones(len(Hops))
    else:
        maxAmp = maxA
    
            
    
    
    sys_para = SystemParametersGeneral(H0,Hops,Hnames,U,U0,total_time,steps,forbidden,states_concerned_list,multi_mode,maxAmp, draw,initial_guess, evolve_only, evolve_error, show_plots, H_time_scales,unitary_error,state_transfer,no_scaling,limit_dc, limit_dc_segment_num, forbid_dressed, save, file_path)
    if use_gpu:
        dev = '/gpu:0'
    else:
        dev = '/cpu:0'
            
    with tf.device(dev):
        tfs = TensorflowState(sys_para,use_gpu) # create tensorflow graph
        graph = tfs.build_graph()
    
    class Convergence(ConvergenceGeneral):
        def __init__(self):
        # paramters
            self.sys_para = sys_para
            self.Modulation = self.sys_para.Modulation
            self.Interpolation = self.sys_para.Interpolation
            self.time_unit = time_unit
          
            if 'rate' in convergence:
                self.rate = convergence['rate']
            else:
                self.rate = 0.01
            
            if 'update_step' in convergence:
                self.update_step = convergence['update_step']
            else:
                self.update_step = 100
                
            if 'conv_target' in convergence:
                self.conv_target = convergence['conv_target']
            else:
                self.conv_target = 1e-8
                
            if 'max_iterations' in convergence:
                self.max_iterations = convergence['max_iterations']
            else:
                self.max_iterations = 5000    
            
            if 'learning_rate_decay' in convergence:
                self.learning_rate_decay = convergence['learning_rate_decay']
            else:
                self.learning_rate_decay = 2500
            
            if 'min_grad' in convergence:
                self.min_grad = convergence['min_grad']
            else:
                self.min_grad = 1e-25 
            


            
            self.reg_alpha_coeff = penalty_coeffs['envelope']

            self.z_reg_alpha_coeff = penalty_coeffs['dc']

            self.dwdt_reg_alpha_coeff = penalty_coeffs['dwdt']
            self.d2wdt2_reg_alpha_coeff = penalty_coeffs['d2wdt2']

            self.inter_reg_alpha_coeff = penalty_coeffs['forbidden']
            

            self.reset_convergence()  
    conv = Convergence()
    
    try:
        SS = run_session(tfs,graph,conv,sys_para,method,switch = switch, show_plots = sys_para.show_plots, name = file_name)
        return SS.uks,SS.Uf
    except KeyboardInterrupt:
        display.clear_output()
    
    
   
