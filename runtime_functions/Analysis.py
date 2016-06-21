import numpy as np
from math_functions.Sort import Sort_ev
import os

class Analysis:
    
    def __init__(self, sys_para,tf_final_state, tf_ops_weight,tf_xy_weight, tf_xy_nocos, tf_unitary_scale, tf_inter_vecs, raw_weight = None):
        self.sys_para = sys_para
        self.tf_final_state = tf_final_state
        self.tf_ops_weight = tf_ops_weight
        self.tf_xy_weight = tf_xy_weight
        self.tf_xy_nocos = tf_xy_nocos
        self.tf_unitary_scale = tf_unitary_scale
        if raw_weight != None:
            self.raw_weight = raw_weight
        self.tf_inter_vecs = tf_inter_vecs
	self.this_dir = os.path.dirname(__file__)    

    def RtoCMat(self,M):
        state_num = self.sys_para.state_num
        M_real = M[:state_num,:state_num]
        M_imag = M[state_num:2*state_num,:state_num]
        
        return (M_real+1j*M_imag)
        
    def get_final_state(self):
        M = self.tf_final_state.eval()
        CMat = self.RtoCMat(M)
	data_path = os.path.join(self.this_dir,'../data/GRAPE-final-state')
        np.save(data_path, np.array(CMat))
        return CMat
        
    def get_ops_weight(self):        
        ops_weight = self.tf_ops_weight.eval()
	data_path = os.path.join(self.this_dir,'../data/GRAPE-ops-weight')
        np.save(data_path, np.array(ops_weight))
        
        return ops_weight
    
    def get_xy_weight(self):        
        xy_weight = self.tf_xy_weight.eval()
	data_path = os.path.join(self.this_dir,'../data/GRAPE-xy-weight')
        np.save(data_path, np.array(xy_weight))
        return xy_weight
    def get_raw_weight(self): 
	data_path = os.path.join(self.this_dir,'../data/GRAPE-xy-weight')
        raw_weight =[]
        for ii in range (len(self.sys_para.Dts)):
            raw_weight.append(self.raw_weight[ii].eval())
            np.save(data_path, np.array(raw_weight[ii]))

        
        return raw_weight
    def get_nonmodulated_weight(self):        
        xy_nocos = self.tf_xy_nocos.eval()
	data_path = os.path.join(self.this_dir,'../data/GRAPE-nocos-weight')
        np.save(data_path, np.array(xy_nocos))
        return xy_nocos
    
    
    def get_inter_vecs(self):
        state_num = self.sys_para.state_num
        inter_vecs_mag_squared = []
        if self.sys_para.D:
            v_sorted=Sort_ev(self.sys_para.v_c,self.sys_para.dressed)
        for tf_inter_vec in self.tf_inter_vecs:
            inter_vec = tf_inter_vec.eval()
            inter_vec_real = 0.5*(inter_vec[0:state_num,:]+inter_vec[state_num:2*state_num,:])
            inter_vec_imag = 0.5*(inter_vec[state_num:2*state_num,:] - inter_vec[0:state_num,:])
            inter_vec_c = inter_vec_real+1j*inter_vec_imag
            
            #print inter_vec_c[:,1]
            #print np.shape(self.sys_para.v_c)
            #print np.shape (v_sorted)
            if self.sys_para.D:
                dressed_vec_c= np.dot(np.transpose(v_sorted),inter_vec_c)
            #print dressed_vec_c[:,1]
                inter_vec_mag_squared = np.square(np.absolute(dressed_vec_c))
            else:
                inter_vec_mag_squared = np.square(np.absolute(inter_vec_c))
            inter_vecs_mag_squared.append(inter_vec_mag_squared)
        #print np.shape(inter_vecs_mag_squared)
        #print (inter_vecs_mag_squared)
	data_path = os.path.join(self.this_dir,'../data/GRAPE-inter_vecs')    
        np.save(data_path, np.array(inter_vecs_mag_squared))
        return inter_vecs_mag_squared
