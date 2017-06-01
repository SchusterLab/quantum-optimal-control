import numpy as np
from quantum_optimal_control.helper_functions.grape_functions import sort_ev,get_state_index
import os
import tensorflow as tf

from quantum_optimal_control.helper_functions.data_management import H5File

class Analysis:
    
    def __init__(self, sys_para,tf_final_state, tf_ops_weight, tf_unitary_scale, tf_inter_vecs):
        self.sys_para = sys_para
        self.tf_final_state = tf_final_state
        self.tf_ops_weight = tf_ops_weight
        self.tf_unitary_scale = tf_unitary_scale
        self.tf_inter_vecs = tf_inter_vecs
	self.this_dir = os.path.dirname(__file__)    

    def RtoCMat(self,M):
        # real to complex matrix isomorphism
        state_num = self.sys_para.state_num
        M_real = M[:state_num,:state_num]
        M_imag = M[state_num:2*state_num,:state_num]
        
        return (M_real+1j*M_imag)
        
    def get_final_state(self,save=True):
        # get final evolved unitary state
        M = self.tf_final_state.eval()
        CMat = self.RtoCMat(M)

        if self.sys_para.save and save:
            with H5File(self.sys_para.file_path) as hf:
                hf.append('final_state',np.array(M))

        return CMat
        
    def get_ops_weight(self):
        # get control field
        ops_weight = self.tf_ops_weight.eval()
        
        return ops_weight
    
    
    def get_inter_vecs(self):
        # get propagated states at each time step
        if not self.sys_para.use_inter_vecs:
            return None
        
        state_num = self.sys_para.state_num
        inter_vecs_mag_squared = []
        
        inter_vecs_real = []
        inter_vecs_imag = []
        
        if self.sys_para.is_dressed:
            v_sorted=sort_ev(self.sys_para.v_c,self.sys_para.dressed_id)
            
        ii=0
        
        inter_vecs = tf.stack(self.tf_inter_vecs).eval()
        
        if self.sys_para.save:
            with H5File(self.sys_para.file_path) as hf:
                hf.append('inter_vecs_raw_real',np.array(inter_vecs[:,0:state_num,:]))
                hf.append('inter_vecs_raw_imag',np.array(inter_vecs[:,state_num:2*state_num,:]))
        
        for inter_vec in inter_vecs:
            inter_vec_real = (inter_vec[0:state_num,:])
            inter_vec_imag = (inter_vec[state_num:2*state_num,:])
            inter_vec_c = inter_vec_real+1j*inter_vec_imag

            if self.sys_para.is_dressed:

                dressed_vec_c= np.dot(np.transpose(v_sorted),inter_vec_c)
                
                inter_vec_mag_squared = np.square(np.abs(dressed_vec_c))
                
                inter_vec_real = np.real(dressed_vec_c)
                inter_vec_imag = np.imag(dressed_vec_c)
                
            else:
                inter_vec_mag_squared = np.square(np.abs(inter_vec_c))
                
                inter_vec_real = np.real(inter_vec_c)
                inter_vec_imag = np.imag(inter_vec_c)
                
                
            inter_vecs_mag_squared.append(inter_vec_mag_squared)
            
            inter_vecs_real.append(inter_vec_real)
            inter_vecs_imag.append(inter_vec_imag)
            
            ii+=1
 
        if self.sys_para.save:
            with H5File(self.sys_para.file_path) as hf:
                hf.append('inter_vecs_mag_squared',np.array(inter_vecs_mag_squared))
                hf.append('inter_vecs_real',np.array(inter_vecs_real))
                hf.append('inter_vecs_imag',np.array(inter_vecs_imag))
        
        return inter_vecs_mag_squared
