import numpy as np
from helper_functions.grape_functions import sort_ev,get_state_index
import os

from helper_functions.datamanagement import H5File

class Analysis:
    
    def __init__(self, sys_para,tf_final_state, tf_ops_weight,tf_xy_weight, tf_xy_nocos, tf_unitary_scale, tf_inter_vecs, raw_weight = None, raws = None, iter_num = 0 ):
        self.sys_para = sys_para
        self.tf_final_state = tf_final_state
        self.tf_ops_weight = tf_ops_weight
        self.tf_xy_weight = tf_xy_weight
        self.tf_xy_nocos = tf_xy_nocos
        self.tf_unitary_scale = tf_unitary_scale
        if raw_weight != None:
            self.raw_weight = raw_weight

        self.iter_num = iter_num
        if raws != None:
            self.raws = raws
        self.tf_inter_vecs = tf_inter_vecs
	self.this_dir = os.path.dirname(__file__)    

    def RtoCMat(self,M):
        state_num = self.sys_para.state_num
        M_real = M[:state_num,:state_num]
        M_imag = M[state_num:2*state_num,:state_num]
        
        return (M_real+1j*M_imag)
        
    def get_final_state(self,save=True):
        M = self.tf_final_state.eval()
        CMat = self.RtoCMat(M)

        if self.sys_para.save and save:
            with H5File(self.sys_para.file_path) as hf:
                hf.append('final_state',np.array(M))

        return CMat
        
    def get_ops_weight(self):        
        ops_weight = self.tf_ops_weight.eval()
        
        
        return ops_weight
    def get_raws(self):        
        ops_weight = self.raws.eval()

        if self.sys_para.save:
            with H5File(self.sys_para.file_path) as hf:
                hf.append('ops_weight',np.array(ops_weight))
        
        return ops_weight
    def get_xy_weight(self):        
        xy_weight = self.tf_xy_weight.eval()

        if self.sys_para.save:
            with H5File(self.sys_para.file_path) as hf:
                hf.append('xy_weight',np.array(xy_weight))
                
        return xy_weight
    def get_raw_weight(self): 

        raw_weight =[]
        ops_weight = self.raws.eval()
        for ii in range (len(self.sys_para.Dts)):
            raw_weight.append(np.tanh(self.raw_weight[ii].eval()).flatten())
            if self.sys_para.save:
                with H5File(self.sys_para.file_path) as hf:
                    hf.append('raw_weight',np.array(raw_weight[ii]))

        
        return raw_weight
    def get_nonmodulated_weight(self):        
        xy_nocos = self.tf_xy_nocos.eval()

        if self.sys_para.save:
            with H5File(self.sys_para.file_path) as hf:
                hf.append('xy_nocos',np.array(xy_nocos))
                
        return xy_nocos
    
    
    def get_inter_vecs(self):
        state_num = self.sys_para.state_num
        inter_vecs_mag_squared = []
        if self.sys_para.is_dressed:
            v_sorted=sort_ev(self.sys_para.v_c,self.sys_para.dressed_id)
            
        ii=0
        for tf_inter_vec in self.tf_inter_vecs:
            inter_vec = tf_inter_vec.eval()
            inter_vec_real = (inter_vec[0:state_num,:])
            inter_vec_imag = (inter_vec[state_num:2*state_num,:])
            inter_vec_c = inter_vec_real+1j*inter_vec_imag

            if self.sys_para.is_dressed:
                inter_vec_mag_squared = []
                cplx_vec = []

                dressed_vec_c= np.dot(np.transpose(v_sorted),inter_vec_c)
                
                inter_vec_mag_squared = np.square(np.abs(dressed_vec_c))
                cplx_vec.append(dressed_vec_c)

            else:
                inter_vec_mag_squared = np.square(np.abs(inter_vec_c))
            inter_vecs_mag_squared.append(inter_vec_mag_squared)
            ii+=1
 
        if self.sys_para.save:
            with H5File(self.sys_para.file_path) as hf:
                hf.append('inter_vecs_mag_squared',np.array(inter_vecs_mag_squared))
        
        return inter_vecs_mag_squared
