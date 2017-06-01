import tensorflow as tf
import numpy as np
import scipy.linalg as la
from quantum_optimal_control.core.tensorflow_state import TensorflowState
from quantum_optimal_control.core.system_parameters import SystemParameters
from quantum_optimal_control.core.convergence import Convergence
from quantum_optimal_control.core.run_session import run_session



import random as rd
import time
from IPython import display

from quantum_optimal_control.helper_functions.data_management import H5File
import os


def Grape(H0,Hops,Hnames,U,total_time,steps,states_concerned_list,convergence = None, U0= None, reg_coeffs = None,dressed_info = None, maxA = None ,use_gpu= True, sparse_H=True,sparse_U=False,sparse_K=False,draw= None, initial_guess = None,show_plots = True, unitary_error=1e-4, method = 'Adam',state_transfer = False,no_scaling = False, freq_unit = 'GHz', file_name = None, save = True, data_path = None, Taylor_terms = None, use_inter_vecs=True):
    
    # start time
    grape_start_time = time.time()
    
    # set timing unit used for plotting
    freq_time_unit_dict = {"GHz": "ns", "MHz": "us","KHz":"ms","Hz":"s"}
    time_unit = freq_time_unit_dict[freq_unit]
    
    # make sparse_{H,U,K} False if use_gpu is True, as GPU Sparse Matmul is not supported yet.
    if use_gpu:
        sparse_H = False
        sparse_U = False
        sparse_K = False
    
    file_path = None
    
    if save:
        # saves all the input values
        if file_name is None:
            raise ValueError('Grape function input: file_name, is not specified.')

        if data_path is None:
            raise ValueError('Grape function input: data_path, is not specified.')


        file_num = 0
        while (os.path.exists(os.path.join(data_path,str(file_num).zfill(5) + "_"+ file_name+".h5"))):
            file_num+=1

        file_name = str(file_num).zfill(5) + "_"+ file_name+ ".h5"

        file_path = os.path.join(data_path,file_name)
        
        print "data saved at: " + str(file_path)

        with H5File(file_path) as hf:
            hf.add('H0',data=H0)
            hf.add('Hops',data=Hops)
            hf.add('Hnames',data=Hnames)
            hf.add('U',data=U)
            hf.add('total_time', data=total_time)
            hf.add('steps', data=steps)
            hf.add('states_concerned_list', data=states_concerned_list)
            hf.add('use_gpu',data=use_gpu)
            hf.add('sparse_H',data=sparse_H)
            hf.add('sparse_U',data=sparse_U)
            hf.add('sparse_K',data=sparse_K)
            
            if not maxA is None:
                hf.add('maxA', data=maxA)
            
            if not initial_guess is None:
                hf.add('initial_guess', data =initial_guess)
            hf.add('method', method)
            
            g1 = hf.create_group('convergence')
            for k, v in convergence.items():
                g1.create_dataset(k, data = v)
            
            if not reg_coeffs is None:
                g2 = hf.create_group('reg_coeffs')
                for k, v in reg_coeffs.items():
                    g2.create_dataset(k, data = v)
                    
            if not dressed_info is None:
                g3 = hf.create_group('dressed_info')
                for k, v in dressed_info.items():
                    g3.create_dataset(k, data = v)        
    
    if U0 is None:
        U0 = np.identity(len(H0))
    if convergence is None:
        convergence = {'rate':0.01, 'update_step':100, 'max_iterations':5000,'conv_target':1e-8,'learning_rate_decay':2500}
       
        
    if maxA is None:
        if initial_guess is None:
            maxAmp = 4*np.ones(len(Hops))
        else:
            maxAmp = 1.5*np.max(np.abs(initial_guess))*np.ones(len(Hops))
    else:
        maxAmp = maxA
    
    # pass in system parameters
    sys_para = SystemParameters(H0,Hops,Hnames,U,U0,total_time,steps,states_concerned_list,dressed_info,maxAmp, draw,initial_guess,  show_plots,unitary_error,state_transfer,no_scaling,reg_coeffs, save, file_path, Taylor_terms, use_gpu, use_inter_vecs,sparse_H,sparse_U,sparse_K)
    
    if use_gpu:
        dev = '/gpu:0'
    else:
        dev = '/cpu:0'
        
        
    with tf.device(dev):
        tfs = TensorflowState(sys_para) # create tensorflow graph
        graph = tfs.build_graph()
    
    conv = Convergence(sys_para,time_unit,convergence)
    
    # run the optimization
    try:
        SS = run_session(tfs,graph,conv,sys_para,method, show_plots = sys_para.show_plots, use_gpu = use_gpu)
        
        # save wall clock time   
        if save:
            wall_clock_time = time.time() - grape_start_time
            with H5File(file_path) as hf:
                hf.add('wall_clock_time',data=np.array(wall_clock_time))
            print "data saved at: " + str(file_path)
        
        return SS.uks,SS.Uf
    except KeyboardInterrupt:
        
        # save wall clock time   
        if save:
            wall_clock_time = time.time() - grape_start_time
            with H5File(file_path) as hf:
                hf.add('wall_clock_time',data=np.array(wall_clock_time))
            print "data saved at: " + str(file_path)
        
        display.clear_output()
    
    
   
