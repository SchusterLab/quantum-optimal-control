import os

import numpy as np
import tensorflow as tf
import math
from quantum_optimal_control.helper_functions.grape_functions import c_to_r_mat, sort_ev
from regularization_functions import get_reg_loss
from tensorflow.python.framework import function
from tensorflow.python.framework import ops

class TensorflowState:
    
    def __init__(self,sys_para):
        
        self.sys_para = sys_para
        
    
    def init_defined_functions(self):
        # define propagation functions used for evolution
        input_num = len(self.sys_para.Hnames) +1
        taylor_terms = self.sys_para.exp_terms 
        scaling = self.sys_para.scaling
        
        
        def get_matexp(uks,H_all):
            # matrix exponential
            I = H_all[input_num]
            matexp = I
            uks_Hk_list = []
            for ii in range(input_num):
                uks_Hk_list.append((uks[ii]/(2.**scaling))*H_all[ii])
                
            H = tf.add_n(uks_Hk_list)
            H_n = H
            factorial = 1.

            for ii in range(1,taylor_terms+1):      
                factorial = factorial * ii
                matexp = matexp + H_n/factorial
                if not ii == (taylor_terms):
                    H_n = tf.matmul(H,H_n,a_is_sparse=self.sys_para.sparse_H,b_is_sparse=self.sys_para.sparse_U)

            for ii in range(scaling):
                matexp = tf.matmul(matexp,matexp,a_is_sparse=self.sys_para.sparse_U,b_is_sparse=self.sys_para.sparse_U)

            return matexp
            
        
        @function.Defun(tf.float32,tf.float32,tf.float32)
        def matexp_op_grad(uks,H_all, grad):  
            # gradient of matrix exponential
            coeff_grad = []

            coeff_grad.append(tf.constant(0,dtype=tf.float32))
            
            
            ### get output of the function
            matexp = get_matexp(uks,H_all)          
            ###
            
            for ii in range(1,input_num):
                coeff_grad.append(tf.reduce_sum(tf.multiply(grad,
                       tf.matmul(H_all[ii],matexp,a_is_sparse=self.sys_para.sparse_H,b_is_sparse=self.sys_para.sparse_U))))

            return [tf.stack(coeff_grad), tf.zeros(tf.shape(H_all),dtype=tf.float32)]                                         

        global matexp_op
        
        
        @function.Defun(tf.float32,tf.float32, grad_func=matexp_op_grad)                       
        def matexp_op(uks,H_all):
            # matrix exponential defun operator
            matexp = get_matexp(uks,H_all)

            return matexp 
        
        def get_matvecexp(uks,H_all,psi):
            # matrix vector exponential
            I = H_all[input_num]
            matvecexp = psi
            
            uks_Hk_list = []
            
            for ii in range(input_num):
                uks_Hk_list.append(uks[ii]*H_all[ii])

            H = tf.add_n(uks_Hk_list)    
            
            psi_n = psi
            factorial = 1.

            for ii in range(1,taylor_terms):      
                factorial = factorial * ii
                psi_n = tf.matmul(H,psi_n,a_is_sparse=self.sys_para.sparse_H,b_is_sparse=self.sys_para.sparse_K)
                matvecexp = matvecexp + psi_n/factorial

            return matvecexp
            
        
        @function.Defun(tf.float32,tf.float32,tf.float32,tf.float32)
        def matvecexp_op_grad(uks,H_all,psi, grad):  
            # graident of matrix vector exponential
            coeff_grad = []

            coeff_grad.append(tf.constant(0,dtype=tf.float32))
            
            ### get output of the function
            matvecexp = get_matvecexp(uks,H_all,psi)
            #####
            
            
            for ii in range(1,input_num):
                coeff_grad.append(tf.reduce_sum(tf.multiply(grad,
                       tf.matmul(H_all[ii],matvecexp,a_is_sparse=self.sys_para.sparse_H,b_is_sparse=self.sys_para.sparse_K))))
                
             
            
            I = H_all[input_num]
            vec_grad = grad
            uks_Hk_list = []
            for ii in range(input_num):
                uks_Hk_list.append((-uks[ii])*H_all[ii])
                
            H = tf.add_n(uks_Hk_list)
            vec_grad_n = grad
            factorial = 1.

            for ii in range(1,taylor_terms):      
                factorial = factorial * ii
                vec_grad_n = tf.matmul(H,vec_grad_n,a_is_sparse=self.sys_para.sparse_H,b_is_sparse=self.sys_para.sparse_K)
                vec_grad = vec_grad + vec_grad_n/factorial

            return [tf.stack(coeff_grad), tf.zeros(tf.shape(H_all),dtype=tf.float32),vec_grad]                                         
        
        global matvecexp_op
        
        @function.Defun(tf.float32,tf.float32,tf.float32, grad_func=matvecexp_op_grad)                       
        def matvecexp_op(uks,H_all,psi):
            # matrix vector exponential defun operator
            matvecexp = get_matvecexp(uks,H_all,psi)

            return matvecexp

 

    def init_variables(self):
        self.tf_one_minus_gaussian_envelope = tf.constant(self.sys_para.one_minus_gauss,dtype=tf.float32, name = 'Gaussian')
        
        
    def init_tf_vectors(self):

        self.tf_initial_vectors=[]
        for initial_vector in self.sys_para.initial_vectors:
            tf_initial_vector = tf.constant(initial_vector,dtype=tf.float32)
            self.tf_initial_vectors.append(tf_initial_vector)
        self.packed_initial_vectors = tf.transpose(tf.stack(self.tf_initial_vectors))
    
    def init_tf_propagators(self):
        #tf initial and target propagator
        if self.sys_para.state_transfer:
            self.target_vecs = tf.transpose(tf.constant(np.array(self.sys_para.target_vectors),dtype=tf.float32))
        else:
            self.tf_initial_unitary = tf.constant(self.sys_para.initial_unitary,dtype=tf.float32, name = 'U0')
            self.tf_target_state = tf.constant(self.sys_para.target_unitary,dtype=tf.float32)
            self.target_vecs = tf.matmul(self.tf_target_state,self.packed_initial_vectors)
        print "Propagators initialized."
    
    def init_tf_ops_weight(self):
       
        #tf weights of operators
            
        self.H0_weight = tf.Variable(tf.ones([self.sys_para.steps]), trainable=False) #Just a vector of ones needed for the kernel
        self.weights_unpacked=[self.H0_weight] #will collect all weights here
        self.ops_weight_base = tf.Variable(tf.constant(self.sys_para.ops_weight_base, dtype = tf.float32), dtype=tf.float32,name ="weights_base")

        self.ops_weight = tf.sin(self.ops_weight_base,name="weights")
        for ii in range (self.sys_para.ops_len):
            self.weights_unpacked.append(self.sys_para.ops_max_amp[ii]*self.ops_weight[ii,:])

        #print len(self.sys_para.ops_max_amp)
        self.H_weights = tf.stack(self.weights_unpacked,name="packed_weights")
           


        print "Operators weight initialized."
                
    def init_tf_inter_propagators(self):
        #initialize intermediate unitaries
        self.inter_states = []    
        for ii in range(self.sys_para.steps):
            self.inter_states.append(tf.zeros([2*self.sys_para.state_num,2*self.sys_para.state_num],
                                              dtype=tf.float32,name="inter_state_"+str(ii)))
        print "Intermediate propagation variables initialized."
            
    def get_inter_state_op(self,layer):
        # build operator for intermediate state propagation
        # This function determines the nature of propagation
       
        propagator = matexp_op(self.H_weights[:,layer],self.tf_matrix_list)
        
        
        return propagator    
        
    def init_tf_propagator(self):
        self.tf_matrix_list = tf.constant(self.sys_para.matrix_list,dtype=tf.float32)

        # build propagator for all the intermediate states
       
        tf_inter_state_op = []
        for ii in np.arange(0,self.sys_para.steps):
            tf_inter_state_op.append(self.get_inter_state_op(ii))

        #first intermediate propagator
        self.inter_states[0] = tf.matmul(tf_inter_state_op[0],self.tf_initial_unitary,a_is_sparse=self.sys_para.sparse_U,
                                         b_is_sparse=self.sys_para.sparse_K)
        #subsequent operation layers and intermediate propagators
        
        for ii in np.arange(1,self.sys_para.steps):
            self.inter_states[ii] = tf.matmul(tf_inter_state_op[ii],self.inter_states[ii-1],a_is_sparse=self.sys_para.sparse_U,
                                              b_is_sparse=self.sys_para.sparse_K)
            
        
        self.final_state = self.inter_states[self.sys_para.steps-1]
        
        self.unitary_scale = (0.5/self.sys_para.state_num)*tf.reduce_sum(tf.matmul(tf.transpose(self.final_state),self.final_state))
        
        print "Intermediate propagators initialized."
        
    def init_tf_inter_vectors(self):
        # inter vectors for unitary evolution, obtained by multiplying the propagation operator K_j with initial vector
        self.inter_vecs_list =[]
        
        inter_vec = self.packed_initial_vectors
        self.inter_vecs_list.append(inter_vec)
        
        for ii in np.arange(0,self.sys_para.steps):               
            inter_vec = tf.matmul(self.inter_states[ii],self.packed_initial_vectors,name="inter_vec_"+str(ii))
            self.inter_vecs_list.append(inter_vec)
        self.inter_vecs_packed = tf.stack(self.inter_vecs_list, axis=1)
        self.inter_vecs = tf.unstack(self.inter_vecs_packed, axis = 2)
            
        print "Vectors initialized."
        
    def init_tf_inter_vector_state(self): 
        # inter vectors for state transfer, obtained by evolving the initial vector

        tf_matrix_list = tf.constant(self.sys_para.matrix_list,dtype=tf.float32)
        
        self.inter_vecs_list = []
        inter_vec = self.packed_initial_vectors
        self.inter_vecs_list.append(inter_vec)
        
        for ii in np.arange(0,self.sys_para.steps):
            psi = inter_vec               
            inter_vec = matvecexp_op(self.H_weights[:,ii],tf_matrix_list,psi)
            self.inter_vecs_list.append(inter_vec)
        self.inter_vecs_packed = tf.stack(self.inter_vecs_list, axis=1)
        self.inter_vecs = tf.unstack(self.inter_vecs_packed, axis = 2)
        
            
        print "Vectors initialized."
        
    def get_inner_product(self,psi1,psi2):
        #Take 2 states psi1,psi2, calculate their overlap, for single vector
        state_num=self.sys_para.state_num
        
        psi_1_real = (psi1[0:state_num])
        psi_1_imag = (psi1[state_num:2*state_num])
        psi_2_real = (psi2[0:state_num])
        psi_2_imag = (psi2[state_num:2*state_num])
        # psi1 has a+ib, psi2 has c+id, we wanna get Sum ((ac+bd) + i (bc-ad)) magnitude
        with tf.name_scope('inner_product'):
            ac = tf.multiply(psi_1_real,psi_2_real)
            bd = tf.multiply(psi_1_imag,psi_2_imag)
            bc = tf.multiply(psi_1_imag,psi_2_real)
            ad = tf.multiply(psi_1_real,psi_2_imag)
            reals = tf.square(tf.add(tf.reduce_sum(ac),tf.reduce_sum(bd)))
            imags = tf.square(tf.subtract(tf.reduce_sum(bc),tf.reduce_sum(ad)))
            norm = tf.add(reals,imags)
        return norm
        
    def get_inner_product_2D(self,psi1,psi2):
        #Take 2 states psi1,psi2, calculate their overlap, for arbitrary number of vectors
        # psi1 and psi2 are shaped as (2*state_num, number of vectors)
        state_num=self.sys_para.state_num
        
        psi_1_real = (psi1[0:state_num,:])
        psi_1_imag = (psi1[state_num:2*state_num,:])
        psi_2_real = (psi2[0:state_num,:])
        psi_2_imag = (psi2[state_num:2*state_num,:])
        # psi1 has a+ib, psi2 has c+id, we wanna get Sum ((ac+bd) + i (bc-ad)) magnitude
        with tf.name_scope('inner_product'):
            ac = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_real),0)
            bd = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_imag),0)
            bc = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_real),0)
            ad = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_imag),0)
            reals = tf.square(tf.reduce_sum(tf.add(ac,bd))) # first trace inner product of all vectors, then squared
            imags = tf.square(tf.reduce_sum(tf.subtract(bc,ad)))
            norm = (tf.add(reals,imags))/(len(self.sys_para.states_concerned_list)**2)
        return norm
    
    def get_inner_product_3D(self,psi1,psi2):
        #Take 2 states psi1,psi2, calculate their overlap, for arbitrary number of vectors and timesteps
        # psi1 and psi2 are shaped as (2*state_num, time_steps, number of vectors)
        state_num=self.sys_para.state_num
        
        psi_1_real = (psi1[0:state_num,:])
        psi_1_imag = (psi1[state_num:2*state_num,:])
        psi_2_real = (psi2[0:state_num,:])
        psi_2_imag = (psi2[state_num:2*state_num,:])
        # psi1 has a+ib, psi2 has c+id, we wanna get Sum ((ac+bd) + i (bc-ad)) magnitude
        with tf.name_scope('inner_product'):
            ac = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_real),0)
            bd = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_imag),0)
            bc = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_real),0)
            ad = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_imag),0)
            reals = tf.reduce_sum(tf.square(tf.reduce_sum(tf.add(ac,bd),1)))
            # first trace inner product of all vectors, then squared, then sum contribution of all time steps
            imags = tf.reduce_sum(tf.square(tf.reduce_sum(tf.subtract(bc,ad),1)))
            norm = (tf.add(reals,imags))/(len(self.sys_para.states_concerned_list)**2)
        return norm
    
    def init_training_loss(self):
        # Adding all penalties
        if self.sys_para.state_transfer == False:
            
            self.final_vecs = tf.matmul(self.final_state, self.packed_initial_vectors)
            
            self.loss = 1-self.get_inner_product_2D(self.final_vecs,self.target_vecs)
        
        else:
            self.loss = tf.constant(0.0, dtype = tf.float32)
            self.final_state = self.inter_vecs_packed[:,self.sys_para.steps,:]
            self.loss = 1-self.get_inner_product_2D(self.final_state,self.target_vecs)
            self.unitary_scale = self.get_inner_product_2D(self.final_state,self.final_state)
            
    
        self.reg_loss = get_reg_loss(self)
        
        print "Training loss initialized."
            
    def init_optimizer(self):
        # Optimizer. Takes a variable learning rate.
        self.learning_rate = tf.placeholder(tf.float32,shape=[])
        self.opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        
        #Here we extract the gradients of the pulses
        self.grad = self.opt.compute_gradients(self.reg_loss)

        self.grad_pack = tf.stack([g for g, _ in self.grad])
        
        self.grads =[tf.nn.l2_loss(g) for g, _ in self.grad]
        self.grad_squared = tf.reduce_sum(tf.stack(self.grads))
        self.optimizer = self.opt.apply_gradients(self.grad)
        
        print "Optimizer initialized."
    
    def init_utilities(self):
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()
        
        print "Utilities initialized."
        
      
            
    def build_graph(self):
        # graph building for the quantum optimal control
        graph = tf.Graph()
        with graph.as_default():
            
            print "Building graph:"
            
            self.init_defined_functions()
            self.init_variables()
            self.init_tf_vectors()
            self.init_tf_propagators()
            self.init_tf_ops_weight()
            if self.sys_para.state_transfer == False:
                self.init_tf_inter_propagators()
                self.init_tf_propagator()
                if self.sys_para.use_inter_vecs:
                    self.init_tf_inter_vectors()
                else:
                    self.inter_vecs = None
            else:
                self.init_tf_inter_vector_state()
            self.init_training_loss()
            self.init_optimizer()
            self.init_utilities()
         
            
            print "Graph built!"
        
        return graph
