import numpy as np

class Analysis:
    
    def __init__(self, sys_para, tf_final_state, tf_Hx, tf_Hz,tf_unitary_scale, tf_inter_vecs):
        self.sys_para = sys_para
        self.tf_final_state = tf_final_state
        self.tf_Hx = tf_Hx
        self.tf_Hz = tf_Hz
        self.tf_unitary_scale = tf_unitary_scale
        self.tf_inter_vecs = tf_inter_vecs
    
    def RtoCMat(self,M):
        state_num = self.sys_para.state_num
        M_real = M[:state_num,:state_num]
        M_imag = M[state_num:2*state_num,:state_num]
        
        return (M_real+1j*M_imag)
        
    def get_final_state(self):
        M = self.tf_final_state.eval()
        CMat = self.RtoCMat(M)
        np.save("./data/GRAPE-M", np.array(CMat))
        return CMat
        
    def get_ops_weight(self):        
        Hx = self.tf_Hx.eval()
        Hz = self.tf_Hz.eval()
        #np.save("./data/GRAPE-control", np.array(ops_weight))
        return [Hx,Hz]
    
    def get_inter_vecs(self):
        state_num = self.sys_para.state_num
        inter_vecs_mag_squared = []
        for tf_inter_vec in self.tf_inter_vecs:
            inter_vec = tf_inter_vec.eval()
            inter_vec_real = 0.5*(inter_vec[0:state_num,:]+inter_vec[state_num:2*state_num,:])
            inter_vec_imag = 0.5*(inter_vec[state_num:2*state_num,:] - inter_vec[0:state_num,:])

            inter_vec_mag_squared = np.square(np.absolute(inter_vec_real+1j*inter_vec_imag))
            inter_vecs_mag_squared.append(inter_vec_mag_squared)
        np.save("./data/GRAPE-inter_vecs", np.array(inter_vecs_mag_squared))
        return inter_vecs_mag_squared
