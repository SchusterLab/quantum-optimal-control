import tensorflow as tf
import numpy as np
import scipy.linalg as la
from core.TensorflowState import TensorflowState
from core.SystemParameters import SystemParameters
from core.Convergence import Convergence
from core.run_session import run_session



import random as rd
import time
from IPython import display

from helper_functions.datamanagement import H5File
import os


def Grape(H0,Hops,Hnames,U,total_time,steps,states_concerned_list,convergence = None, U0= None, reg_coeffs = None,dressed_info = None, maxA = None ,use_gpu= True, draw= None, initial_guess = None, evolve_only = False, evolve_error = False,show_plots = True, H_time_scales = None, unitary_error=1e-4, method = 'Adam',state_transfer = False, switch = True,no_scaling = False, freq_unit = 'GHz', gate = None, save = True, data_path = None):
    
    
    if freq_unit == 'GHz':
        time_unit = 'ns'
    elif freq_unit == 'MHz':
        time_unit = 'us'
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
    
    if evolve_only:
        reg_coeffs = dict.fromkeys(reg_coeffs, 0)
       
        
    if maxA == None:
        if initial_guess == None:
            maxAmp = 4*np.ones(len(Hops))
        else:
            maxAmp = 1.5*np.max(np.abs(initial_guess))*np.ones(len(Hops))
    else:
        maxAmp = maxA
    
            
    
    
    sys_para = SystemParameters(H0,Hops,Hnames,U,U0,total_time,steps,states_concerned_list,dressed_info,maxAmp, draw,initial_guess, evolve_only, evolve_error, show_plots, H_time_scales,unitary_error,state_transfer,no_scaling,reg_coeffs, save, file_path)
    if use_gpu:
        dev = '/gpu:0'
    else:
        dev = '/cpu:0'
            
    with tf.device(dev):
        tfs = TensorflowState(sys_para,use_gpu) # create tensorflow graph
        graph = tfs.build_graph()
    
    conv = Convergence(sys_para,time_unit,convergence)
    
    try:
        SS = run_session(tfs,graph,conv,sys_para,method,switch = switch, show_plots = sys_para.show_plots)
        return SS.uks,SS.Uf
    except KeyboardInterrupt:
        display.clear_output()
    
    
   
