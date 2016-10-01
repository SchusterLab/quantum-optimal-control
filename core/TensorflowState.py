import os

import numpy as np
import tensorflow as tf
import math
from helper_functions.grape_functions import c_to_r_mat, sort_ev
#from custom_kernels.gradients.matexp_grad_vecs_v3 import *
#from custom_kernels.gradients.matexp_grad_v3 import *
from RegularizationFunctions import get_reg_loss
from tensorflow.python.framework import function
from tensorflow.python.framework import ops

class TensorflowState:
    
    def __init__(self,sys_para,use_gpu = True):
        
        self.sys_para = sys_para
        
        input_num = len(self.sys_para.Hnames) +1
        taylor_terms = self.sys_para.exp_terms 
        scaling = self.sys_para.scaling
        
        
        @ops.RegisterGradient("matexp_op")
        def matexp_op_grad(op, grad):  

            coeff_grad = []

            coeff_grad.append(tf.constant(0,dtype=tf.float32))
            for ii in range(1,input_num):
                coeff_grad.append(tf.reduce_sum(tf.mul(grad,
                       tf.matmul(op.inputs[1][ii],op.outputs[0]))))

            return [tf.pack(coeff_grad), tf.zeros(tf.shape(op.inputs[1]),dtype=tf.float32)]                                         

        global matexp_op
        
        @function.Defun(tf.float32,tf.float32, python_grad_func=matexp_op_grad)                       
        def matexp_op(uks,H_all):

            I = H_all[input_num]
            matexp = I
            H = I-I
            for ii in range(input_num):
                H = H + uks[ii]*H_all[ii]/(2.**scaling)
            H_n = H
            factorial = 1.

            for ii in range(1,taylor_terms):      
                factorial = factorial * ii
                matexp = matexp + H_n/factorial
                H_n = tf.matmul(H,H_n)

            for ii in range(scaling):
                matexp = tf.matmul(matexp,matexp)

            return matexp 
        
        @ops.RegisterGradient("matvecexp_op")
        def matvecexp_op_grad(op, grad):  

            coeff_grad = []

            coeff_grad.append(tf.constant(0,dtype=tf.float32))
            for ii in range(1,input_num):
                coeff_grad.append(tf.reduce_sum(tf.mul(grad,
                       tf.matmul(op.inputs[1][ii],op.outputs[0]))))
                
             
            uks = op.inputs[0]
            H_all = op.inputs[1]
            
            I = H_all[input_num]
            vec_grad = grad
            H = I-I
            for ii in range(input_num):
                H = H - uks[ii]*H_all[ii]
            vec_grad_n = grad
            factorial = 1.

            for ii in range(1,taylor_terms):      
                factorial = factorial * ii
                vec_grad_n = tf.matmul(H,vec_grad_n)
                vec_grad = vec_grad + vec_grad_n/factorial

            return [tf.pack(coeff_grad), tf.zeros(tf.shape(op.inputs[1]),dtype=tf.float32),vec_grad]                                         
        global matvecexp_op
        
        @function.Defun(tf.float32,tf.float32,tf.float32, python_grad_func=matvecexp_op_grad)                       
        def matvecexp_op(uks,H_all,psi):
            
            I = H_all[input_num]
            matvecexp = psi
            H = I-I
            for ii in range(input_num):
                H = H + uks[ii]*H_all[ii]
            psi_n = psi
            factorial = 1.

            for ii in range(1,taylor_terms):      
                factorial = factorial * ii
                psi_n = tf.matmul(H,psi_n)
                matvecexp = matvecexp + psi_n/factorial

            return matvecexp

        
        # Setting up our matrix exponential kernel
        #this_dir = os.path.dirname(__file__)
        #user_ops_path = os.path.join(this_dir,'../custom_kernels/build')
    
        #if self.sys_para.state_transfer:
        #    if use_gpu: #choosing matrix_vector kernel
        #        kernel_filename = 'cuda_matexp_vecs_v2.so'
        #        matrix_vec_grad_exp_module = tf.load_op_library(os.path.join(user_ops_path,'cuda_matexp_vecs_grads_v2.so'))
        #        matmul_vec_module = tf.load_op_library(os.path.join(user_ops_path,'cuda_matmul_vec.so'))
        #        import custom_kernels.gradients.matexp_grad_vecs_v3 as mgv
        #        mgv.register_gradient(matrix_vec_grad_exp_module,matmul_vec_module)
        #    else:
        #        kernel_filename = 'matrix_exp_vecs.so'
        #        matrix_vec_grad_exp_module = tf.load_op_library(os.path.join(user_ops_path,'matrix_exp_vecs_grads.so'))
        #        matmul_vec_module = tf.load_op_library(os.path.join(user_ops_path,'matmul_vec.so'))
        #        import custom_kernels.gradients.matexp_grad_vecs_v3 as mgv
        #        mgv.register_gradient(matrix_vec_grad_exp_module,matmul_vec_module)
        #else: #choosing matrix matrix kernel

        #    if use_gpu:
        #        kernel_filename = 'cuda_matexp_v4.so'
        #    else:
        #        kernel_filename = 'matrix_exp_v2.so'

        #with tf.name_scope('kernel'):
        #    self.matrix_exp_module = tf.load_op_library(os.path.join(user_ops_path,kernel_filename))      
        #   

    def init_variables(self):
        self.tf_one_minus_gaussian_envelope = tf.constant(self.sys_para.one_minus_gauss,dtype=tf.float32, name = 'Gaussian')
        
        
    def init_tf_vectors(self):

        self.tf_initial_vectors=[]
        for initial_vector in self.sys_para.initial_vectors:
            tf_initial_vector = tf.constant(initial_vector,dtype=tf.float32)
            self.tf_initial_vectors.append(tf_initial_vector)
        self.packed_initial_vectors = tf.transpose(tf.pack(self.tf_initial_vectors))
    
    def init_tf_propagators(self):
        #tf initial and target propagator
        if self.sys_para.state_transfer:
            self.tf_target_vectors = []
            for target_vector in self.sys_para.target_vectors:
                tf_target_vector = tf.constant(target_vector,dtype=tf.float32)
                self.tf_target_vectors.append(tf_target_vector)
            self.tf_target_state = tf.transpose(tf.pack(self.tf_target_vectors))
        else:
            self.tf_initial_unitary = tf.constant(self.sys_para.initial_unitary,dtype=tf.float32, name = 'U0')
            self.tf_target_state = tf.constant(self.sys_para.target_unitary,dtype=tf.float32)
            self.target_states = tf.matmul(self.tf_target_state,self.packed_initial_vectors)
        print "Propagators initialized."
        
  
        
    def get_j(self,l, Dt): # function for the interpolation
        dt=self.sys_para.dt
        jj=np.floor((l*dt-0.5*Dt)/Dt)
        return jj
    
    
            
    
    def transfer_fn_general(self,w,steps): # Interpolating function, interpolates weights vector w to steps size
        
        indices=[]
        values=[]
        shape=[self.sys_para.steps,steps]
        dt=self.sys_para.dt
        Dt=self.sys_para.total_time/steps
    
    # Cubic Splines
        for ll in range (self.sys_para.steps):
            jj=self.get_j(ll,Dt)
            tao= ll*dt - jj*Dt - 0.5*Dt
            if jj >= 1:
                indices.append([int(ll),int(jj-1)])
                temp= -(tao/(2*Dt))*((tao/Dt)-1)**2
                values.append(temp)
                
            if jj >= 0:
                indices.append([int(ll),int(jj)])
                temp= 1+((3*tao**3)/(2*Dt**3))-((5*tao**2)/(2*Dt**2))
                values.append(temp)
                
            if jj+1 <= steps-1:
                indices.append([int(ll),int(jj+1)])
                temp= ((tao)/(2*Dt))+((4*tao**2)/(2*Dt**2))-((3*tao**3)/(2*Dt**3))
                values.append(temp)
               
            if jj+2 <= steps-1:
                indices.append([int(ll),int(jj+2)])
                temp= ((tao**3)/(2*Dt**3))-((tao**2)/(2*Dt**2))
                values.append(temp)
                
            
        T1=tf.SparseTensor(indices, values, shape)  
        T2=tf.sparse_reorder(T1)
        T=tf.sparse_tensor_to_dense(T2)
        temp1 = tf.matmul(T,tf.reshape(w[0,:],[steps,1]))
        
        return tf.transpose(temp1)
    
    def init_tf_ops_weight(self):
        
        
        self.raw_weight =[]
        self.ops_weight_base = []
        
        #tf weights of operators
        
            
        self.H0_weight = tf.Variable(tf.ones([self.sys_para.steps]), trainable=False) #Just a vector of ones needed for the kernel
        self.weights_unpacked=[self.H0_weight] #will collect all weights here
        self.raw_guesses = tf.constant(self.sys_para.ops_weight_base, dtype = tf.float32)
        self.raws = tf.Variable(self.raw_guesses, dtype=tf.float32,name ="raw_weights")


        if self.sys_para.u0 == []: #No initial guess supplied
            
           
            if self.sys_para.Dts != []: # We have different time scales
                
                

                
                for ii in range (self.sys_para.ops_len -len(self.sys_para.Dts)):
                    self.ops_weight_base.append(self.raws[:,ii*self.sys_para.steps:(ii+1)*self.sys_para.steps])
#start interpolating them
                
                starting_index = (self.sys_para.ops_len -len(self.sys_para.Dts))*self.sys_para.steps 
                

                for ii in range (len(self.sys_para.Dts)):
                    
                    ws = self.raws[:,starting_index:starting_index + self.sys_para.ctrl_steps[ii]]
                    self.ops_weight_base.append(self.transfer_fn_general(ws,self.sys_para.ctrl_steps[ii]))
                    self.raw_weight.append(ws)
                    starting_index = starting_index + self.sys_para.ctrl_steps[ii]
                self.ops_weight_base = tf.reshape(tf.pack(self.ops_weight_base),[self.sys_para.ops_len,self.sys_para.steps])
                    





            else: #No interpolation needed

                self.ops_weight_base = self.raws

            
        else: #initial guess supplied
            self.ops_weight_base = self.raws

        self.ops_weight = tf.tanh(self.ops_weight_base,name="weights")
        for ii in range (self.sys_para.ops_len):
            self.weights_unpacked.append(self.sys_para.ops_max_amp[ii]*self.ops_weight[ii,:])

        
        self.H_weights = tf.pack(self.weights_unpacked,name="packed_weights")
           


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
        
        #propagator = self.matrix_exp_module.matrix_exp(self.H_weights[:,layer],self.tf_matrix_list,size=2*self.sys_para.state_num, input_num = self.sys_para.ops_len+1,exp_num = self.sys_para.exp_terms, div = self.sys_para.scaling)
        
        
        return propagator    
        
    def init_tf_propagator(self):
        self.tf_matrix_list = tf.constant(self.sys_para.matrix_list,dtype=tf.float32)

        # build propagator for all the intermediate states
       
        tf_inter_state_op = []
        for ii in np.arange(0,self.sys_para.steps):
            tf_inter_state_op.append(self.get_inter_state_op(ii))

        #first intermediate propagator
        self.inter_states[0] = tf.matmul(tf_inter_state_op[0],self.tf_initial_unitary)
        #subsequent operation layers and intermediate propagators
        
        for ii in np.arange(1,self.sys_para.steps):
            self.inter_states[ii] = tf.matmul(tf_inter_state_op[ii],self.inter_states[ii-1])
            
        
        self.final_state = self.inter_states[self.sys_para.steps-1]
        
        self.unitary_scale = (0.5/self.sys_para.state_num)*tf.reduce_sum(tf.matmul(tf.transpose(self.final_state),self.final_state))
        
        print "Intermediate propagators initialized."
        
    def init_tf_inter_vectors(self):
        self.inter_vecs=[]
        self.inter_vec =[]
        
        for tf_initial_vector in self.tf_initial_vectors:
        
            inter_vec = tf.reshape(tf_initial_vector,[2*self.sys_para.state_num,1],name="initial_vector")
            self.inter_vec.append(inter_vec)
            for ii in np.arange(0,self.sys_para.steps):
                inter_vec = tf.matmul(self.inter_states[ii],tf.reshape(tf_initial_vector,[2*self.sys_para.state_num,1]),name="inter_vec_"+str(ii))
                self.inter_vec.append(inter_vec)
            self.inter_vec = tf.transpose(tf.reshape(tf.pack(self.inter_vec),[self.sys_para.steps+1,2*self.sys_para.state_num]),name = "vectors_for_one_psi0")
            self.inter_vecs.append(self.inter_vec)
            self.inter_vec=[]
            
        print "Vectors initialized."
        
    def init_tf_inter_vector_state(self): # for state transfer

        tf_matrix_list = tf.constant(self.sys_para.matrix_list,dtype=tf.float32)
        
        self.inter_vecs=[]
        
        for tf_initial_vector in self.tf_initial_vectors:
            self.inter_vec = []
            inter_vec = tf.reshape(tf_initial_vector,[2*self.sys_para.state_num,1],name="initial_vector")
            self.inter_vec.append(inter_vec)

            for ii in np.arange(0,self.sys_para.steps):
                psi = inter_vec               
                inter_vec = matvecexp_op(self.H_weights[:,ii],tf_matrix_list,psi)
                self.inter_vec.append(inter_vec)
            self.inter_vec = tf.transpose(tf.reshape(tf.pack(self.inter_vec),[self.sys_para.steps+1,2*self.sys_para.state_num]),name = "vectors_for_one_psi0")
            self.inter_vecs.append(self.inter_vec)
        print "Vectors initialized."
        
    def get_inner_product(self,psi1,psi2):
        #Take 2 states psi1,psi2, calculate their overlap.
        state_num=self.sys_para.state_num
        
        psi_1_real = (psi1[0:state_num])
        psi_1_imag = (psi1[state_num:2*state_num])
        psi_2_real = (psi2[0:state_num])
        psi_2_imag = (psi2[state_num:2*state_num])
        # psi1 has a+ib, psi2 has c+id, we wanna get Sum ((ac+bd) + i (bc-ad)) magnitude
        with tf.name_scope('inner_product'):
            ac = tf.mul(psi_1_real,psi_2_real)
            bd = tf.mul(psi_1_imag,psi_2_imag)
            bc = tf.mul(psi_1_imag,psi_2_real)
            ad = tf.mul(psi_1_real,psi_2_imag)
            reals = tf.square(tf.add(tf.reduce_sum(ac),tf.reduce_sum(bd)))
            imags = tf.square(tf.sub(tf.reduce_sum(bc),tf.reduce_sum(ad)))
            norm = tf.add(reals,imags)
        return norm
        
    def get_inner_product_gen(self,psi1,psi2):
        #Take 2 states psi1,psi2, calculate their overlap.
        state_num=self.sys_para.state_num
        
        psi_1_real = (psi1[0:state_num,:])
        psi_1_imag = (psi1[state_num:2*state_num,:])
        psi_2_real = (psi2[0:state_num,:])
        psi_2_imag = (psi2[state_num:2*state_num,:])
        # psi1 has a+ib, psi2 has c+id, we wanna get Sum ((ac+bd) + i (bc-ad)) magnitude
        with tf.name_scope('inner_product'):
            ac = tf.reduce_sum(tf.mul(psi_1_real,psi_2_real),0)
            bd = tf.reduce_sum(tf.mul(psi_1_imag,psi_2_imag),0)
            bc = tf.reduce_sum(tf.mul(psi_1_imag,psi_2_real),0)
            ad = tf.reduce_sum(tf.mul(psi_1_real,psi_2_imag),0)
            reals = tf.reduce_sum(tf.square(tf.add(ac,bd)))
            imags = tf.reduce_sum(tf.square(tf.sub(bc,ad)))
            norm = (tf.add(reals,imags))/len(self.sys_para.states_concerned_list)
        return norm
    
    def init_training_loss(self):
        # Adding all penalties
        if self.sys_para.state_transfer == False:
            
            self.final_states = tf.matmul(self.final_state, self.packed_initial_vectors)
            
            self.loss = tf.abs(1-self.get_inner_product_gen(self.final_states,self.target_states))
        
        else:
            self.loss = tf.constant(0.0, dtype = tf.float32)
            self.tf_target_vectors
            for ii in range(len(self.inter_vecs)):
                self.final_state= self.inter_vecs[ii][:,self.sys_para.steps]
                self.inner_product = self.get_inner_product(self.tf_target_vectors[ii],self.final_state)
                self.unitary_scale = self.get_inner_product(self.final_state,self.final_state)
                self.loss = self.loss +  tf.abs(1 - self.inner_product, name ="Fidelity_error")
            
    
        self.reg_loss = get_reg_loss(self)
        
        print "Training loss initialized."
            
    def init_optimizer(self):
        # Optimizer. Takes a variable learning rate.
        self.learning_rate = tf.placeholder(tf.float32,shape=[])
        self.opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        
        #Here we extract the gradients of the xy and z pulses
        self.grad = self.opt.compute_gradients(self.reg_loss)
        if self.sys_para.Dts == [] :
            self.grad_pack = tf.pack([g for g, _ in self.grad])
        
        else:
            self.grad_pack,_ = self.grad[0]
            for ii in range (len(self.grad)-1):
                a,_ = self.grad[ii+1]
                self.grad_pack = tf.concat(1,[self.grad_pack,a])
        
        self.grads =[tf.nn.l2_loss(g) for g, _ in self.grad]
        self.grad_squared = tf.reduce_sum(tf.pack(self.grads))
        self.optimizer = self.opt.apply_gradients(self.grad)
        
        print "Optimizer initialized."
    
    def init_utilities(self):
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()
        
        print "Utilities initialized."
        
      
            
    def build_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            
            print "Building graph:"
            
            self.init_variables()
            self.init_tf_vectors()
            self.init_tf_propagators()
            self.init_tf_ops_weight()
            if self.sys_para.state_transfer == False:
                self.init_tf_inter_propagators()
                self.init_tf_propagator()
                self.init_tf_inter_vectors()
            else:
                self.init_tf_inter_vector_state()
            self.init_training_loss()
            self.init_optimizer()
            self.init_utilities()
         
            
            print "Graph built!"
        
        return graph
