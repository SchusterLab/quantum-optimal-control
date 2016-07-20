import os

import numpy as np
import tensorflow as tf
from helper_functions.grape_functions import c_to_r_mat
from custom_kernels.gradients.matexp_grad_vecs import *
from custom_kernels.gradients.matexp_grad_v3 import *
import os

class TensorflowState:
    
    def __init__(self,sys_para,use_gpu = True):
        
        self.sys_para = sys_para
        # Setting up our matrix exponential kernel
        this_dir = os.path.dirname(__file__)
        user_ops_path = os.path.join(this_dir,'../custom_kernels/build')
    
        if self.sys_para.state_transfer: #choosing matrix_vector kernel
            kernel_filename = 'cuda_matexp_vecs.so'
            matrix_vec_grad_exp_module = tf.load_op_library(os.path.join(user_ops_path,'cuda_matexp_vecs_grads.so'))
            import custom_kernels.gradients.matexp_grad_vecs as mgv
            mgv.register_gradient(matrix_vec_grad_exp_module)
        else: #choosing matrix matrix kernel

            if use_gpu:
                kernel_filename = 'cuda_matexp_v4.so'
            else:
                kernel_filename = 'matrix_exp.so'

        with tf.name_scope('kernel'):
            self.matrix_exp_module = tf.load_op_library(os.path.join(user_ops_path,kernel_filename))      
           

    def init_variables(self):
        self.tf_identity = tf.constant(self.sys_para.identity,dtype=tf.float32)
        self.tf_neg_i = tf.constant(c_to_r_mat(-1j*self.sys_para.identity_c),dtype=tf.float32)
        self.tf_one_minus_gaussian_envelope = tf.constant(self.sys_para.one_minus_gauss,dtype=tf.float32)
        
        
    def init_tf_vectors(self):
        if self.sys_para.state_transfer:
            self.tf_initial_vectors = tf.constant(self.sys_para.initial_vectors[0],dtype=tf.float32)
        else:
            self.tf_initial_vectors=[]
            for initial_vector in self.sys_para.initial_vectors:
                tf_initial_vector = tf.constant(initial_vector,dtype=tf.float32)
                self.tf_initial_vectors.append(tf_initial_vector)
    
    def init_tf_propagators(self):
        #tf initial and target propagator
        self.tf_initial_unitary = tf.constant(self.sys_para.initial_unitary,dtype=tf.float32)
        if self.sys_para.state_transfer:
            self.tf_target_state = tf.constant(self.sys_para.target_vector,dtype=tf.float32)
        else:
            self.tf_target_state = tf.constant(self.sys_para.target_unitary,dtype=tf.float32)
        print "Propagators initialized."
        
        
    def init_tf_ops(self):
        #flat operators for control Hamiltonian 
        i_array = np.eye(2*self.sys_para.state_num)
        op_matrix_I=i_array.tolist()
        self.I_flat = [item for sublist in op_matrix_I  for item in sublist]
        self.H0_flat = [item for sublist in self.sys_para.H0  for item in sublist]
        
        self.flat_ops = []
        for op in self.sys_para.ops:
            flat_op = [item for sublist in op for item in sublist]
            self.flat_ops.append(flat_op)
            
        print "Operators initialized."
        
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
        
        #tf weights of operators
        
            
        self.H0_weight = tf.Variable(tf.ones([self.sys_para.steps]), trainable=False) #Just a vector of ones needed for the kernel
        self.weights_unpacked=[self.H0_weight] #will collect all weights here


        if self.sys_para.u0 == []: #No initial guess supplied
            initial_guess = 0
            index = 0
            

            #initial_xy_stddev = (0.1/np.sqrt(self.sys_para.control_steps))
            initial_stddev = (10./np.sqrt(self.sys_para.steps))
            if self.sys_para.Dts != []: # We have different time scales
                
                if self.sys_para.ops_len - len(self.sys_para.Dts) > 0: # if there exists operators that don't need interpolation
                    weights = tf.truncated_normal([self.sys_para.ops_len - len(self.sys_para.Dts) ,self.sys_para.steps],
                                                               mean= initial_guess ,dtype=tf.float32,
                        stddev=initial_stddev ) #initialize all ops that don't need interpolation together first

                    self.ops_weight_base = weights
                    current = weights[0,:]
                    for ii in range (self.sys_para.ops_len - len(self.sys_para.Dts)-1):
                        current = tf.concat(0,[current,weights[ii+1,:]])
                    self.current = tf.reshape(current,[1, (self.sys_para.ops_len - len(self.sys_para.Dts))*self.sys_para.steps])
                    #self.current holds the concatenated weights
                else: # all operators need interpolation
                    initial_stddev = (10./np.sqrt(self.sys_para.ctrl_steps[0]))
                    weights = tf.truncated_normal([1 ,self.sys_para.ctrl_steps[0]],
                                                                   mean= initial_guess ,dtype=tf.float32,
                            stddev=initial_stddev )
                    index = 1
                    self.current = weights
                    #self.current holds the first weight vector

                for ii in range (len(self.sys_para.Dts)-index): # add all remaining non interpolated wieghts
                    initial_stddev = (1/np.sqrt(self.sys_para.ctrl_steps[ii+index]))
                    weight = tf.truncated_normal([1 ,self.sys_para.ctrl_steps[ii+index]],
                                                                   mean= initial_guess ,dtype=tf.float32,
                            stddev=initial_stddev )

                    self.current = tf.concat(1,[self.current,weight])


                self.raws = tf.Variable(self.current, dtype=tf.float32,name ="weights")
#start interpolating them
                first_steps = self.sys_para.ctrl_steps[0]
                self.ops_weight_base = self.transfer_fn_general(self.raws[:,0: first_steps],first_steps)
                self.raw_weight.append(self.raws[:,(self.sys_para.ops_len -len(self.sys_para.Dts))*self.sys_para.steps:(self.sys_para.ops_len -len(self.sys_para.Dts))*self.sys_para.steps+first_steps])
                starting_index = (self.sys_para.ops_len -len(self.sys_para.Dts))*self.sys_para.steps + (index * first_steps)
                flag = False
                if index == 0:
                    flag = True


                for ii in range (len(self.sys_para.Dts)-index):
                    #R = tf.range(starting_index,starting_index + self.sys_para.ctrl_steps[ii+index],1)
                    #ws = tf.gather(self.raws,R)

                    ws = self.raws[:,starting_index:starting_index + self.sys_para.ctrl_steps[ii+index]]
                    self.ops_weight_base = tf.concat(0,[self.ops_weight_base,self.transfer_fn_general(ws,self.sys_para.ctrl_steps[ii+index])])
                    if flag:
                        flag = False
                    else:
                        self.raw_weight.append(ws)


                    starting_index = starting_index + self.sys_para.ctrl_steps[ii+index]
                    





            else: #No interpolation needed

                self.ops_weight_base = tf.Variable(tf.truncated_normal([self.sys_para.ops_len,self.sys_para.steps],
                                                               mean= initial_guess ,dtype=tf.float32,
                        stddev=initial_stddev ),name="weight_bases")
                self.raws = self.ops_weight_base

            
        else: #initial guess supplied
            self.op_weight = tf.constant(self.sys_para.u0[0],dtype=tf.float32)
            for ii in range (self.sys_para.ops_len-1):

                self.op_weight = tf.concat(0,[self.op_weight,self.sys_para.u0[ii+1]])
            self.op_weight = tf.reshape(self.op_weight, [self.sys_para.ops_len,self.sys_para.steps])
            self.ops_weight_base = tf.Variable(self.op_weight,dtype=tf.float32,name="weights")
            self.raws = self.ops_weight_base

        self.ops_weight = tf.tanh(self.ops_weight_base,name="weights")
        for ii in range (self.sys_para.ops_len):
            self.weights_unpacked.append(self.sys_para.ops_max_amp[ii]*self.ops_weight[ii,:])


        self.H_weights = tf.pack(self.weights_unpacked,name="packed_weights")
            
        #self.ops_weight = tf.tanh(self.ops_weight_base)
        
        


        print "Operators weight initialized."
                
    def init_tf_inter_propagators(self):
        #initialize intermediate unitaries
        self.inter_states = []    
        for ii in range(self.sys_para.steps):
            self.inter_states.append(tf.zeros([2*self.sys_para.state_num,2*self.sys_para.state_num],
                                              dtype=tf.float32,name="inter_state_"+str(ii)))
        print "Intermediate propagators initialized."
            
    def get_inter_state_op(self,layer):
        # build operator for intermediate state propagation
        # This function determines the nature of propagation
        matrix_list = self.H0_flat
        for ii in range(self.sys_para.ops_len):
            matrix_list = matrix_list + self.flat_ops[ii]
        matrix_list = matrix_list + self.I_flat
        
        propagator = self.matrix_exp_module.matrix_exp(self.H_weights[:,layer],size=2*self.sys_para.state_num, input_num = self.sys_para.ops_len+1,
                                      exp_num = self.sys_para.exp_terms, div = self.sys_para.scaling
                                      ,matrix=matrix_list)
        
        
        return propagator    
        
    def init_tf_propagator(self):
        # build propagator for all the intermediate states
        
        #first intermediate propagator
        self.inter_states[0] = tf.matmul(self.get_inter_state_op(0),self.tf_initial_unitary)
        #subsequent operation layers and intermediate propagators
        for ii in np.arange(1,self.sys_para.steps):
            self.inter_states[ii] = tf.matmul(self.get_inter_state_op(ii),self.inter_states[ii-1])
            
        
        self.final_state = self.inter_states[self.sys_para.steps-1]
        
        self.unitary_scale = (0.5/self.sys_para.state_num)*tf.reduce_sum(tf.matmul(tf.transpose(self.final_state),self.final_state))
        
        print "Intermediate propagators correlations initialized."
        
    def init_tf_inter_vectors(self):
        self.inter_vecs=[]
        
        for tf_initial_vector in self.tf_initial_vectors:
        
            inter_vec = tf.reshape(tf_initial_vector,[2*self.sys_para.state_num,1],name="initial_vector")
            for ii in np.arange(0,self.sys_para.steps):
                inter_vec_temp = tf.matmul(self.inter_states[ii],tf.reshape(tf_initial_vector,[2*self.sys_para.state_num,1]),name="inter_vec_"+str(ii))
                inter_vec = tf.concat(1,[inter_vec,inter_vec_temp],name = "all_vectors")
                
            self.inter_vecs.append(inter_vec)
            
        print "Vectors initialized."
        
    def init_tf_inter_vector_state(self): # for state transfer
        self.inter_vecs=[]
       
        matrix_list = self.H0_flat
        for ii in range(self.sys_para.ops_len):
            matrix_list = matrix_list + self.flat_ops[ii]
        matrix_list = matrix_list + self.I_flat
        inter_vec = tf.reshape(self.tf_initial_vectors,[2*self.sys_para.state_num,1],name="initial_vector")
        for ii in np.arange(0,self.sys_para.steps):
            if ii==0:
                psi = self.tf_initial_vectors
            else:
                psi = inter_vec_temp
            inter_vec_temp = tf.reshape(self.matrix_exp_module.matrix_exp_vecs(self.H_weights[:,ii],psi,size=2*self.sys_para.state_num, input_num = self.sys_para.ops_len+1,
                                      exp_num = self.sys_para.exp_terms, vecs_num = 1
                                      ,matrix=matrix_list),[2*self.sys_para.state_num,1],name="inter_vec_"+str(ii))
            inter_vec = tf.concat(1,[inter_vec,inter_vec_temp],name = "all_vectors")
            
        self.inter_vecs.append(inter_vec)
        self.unitary_scale = 0
            
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
    
    def init_training_loss(self):
        # Adding all penalties
        if self.sys_para.state_transfer == False:

            inner_product = tf.matmul(tf.transpose(self.tf_target_state),self.final_state)
            inner_product_trace_real = tf.reduce_sum(tf.pack([inner_product[ii,ii] for ii in self.sys_para.states_concerned_list]))\
            /float(len(self.sys_para.states_concerned_list))
            inner_product_trace_imag = tf.reduce_sum(tf.pack([inner_product[self.sys_para.state_num+ii,ii] for ii in self.sys_para.states_concerned_list]))\
            /float(len(self.sys_para.states_concerned_list))

            inner_product_trace_mag_squared = tf.square(inner_product_trace_real) + tf.square(inner_product_trace_imag)

            self.loss = tf.abs(1 - inner_product_trace_mag_squared)
        
        else:
            for inter_vec in self.inter_vecs:
                self.final_state= inter_vec[:,self.sys_para.steps]
            inner_product = self.get_inner_product(self.tf_target_state,self.final_state)
            self.loss = tf.abs(1 - inner_product, name ="Fidelity_error")
    
    
        # Regulizer
        with tf.name_scope('reg_errors'):
            self.reg_loss = self.loss
            self.reg_alpha_coeff = tf.placeholder(tf.float32,shape=[])
            reg_alpha = self.reg_alpha_coeff/float(self.sys_para.steps)
            self.reg_loss = self.reg_loss + reg_alpha * tf.nn.l2_loss(tf.mul(self.tf_one_minus_gaussian_envelope,self.ops_weight))

            # Constrain Z to have no dc value
            self.z_reg_alpha_coeff = tf.placeholder(tf.float32,shape=[])
            z_reg_alpha = self.z_reg_alpha_coeff/float(self.sys_para.steps)
            for state in self.sys_para.limit_dc:
            
                self.reg_loss = self.reg_loss + z_reg_alpha*tf.square(tf.reduce_sum(self.ops_weight[state,:]))
            
            
            

            # Limiting the dwdt of control pulse
            zeros_for_training = tf.zeros([self.sys_para.ops_len, 2])
            new_weights = tf.concat(1, [self.ops_weight, zeros_for_training])
            new_weights = tf.concat(1, [zeros_for_training,new_weights])
            self.dwdt_reg_alpha_coeff = tf.placeholder(tf.float32,shape=[])
            dwdt_reg_alpha = self.dwdt_reg_alpha_coeff/float(self.sys_para.steps)
            self.reg_loss = self.reg_loss + dwdt_reg_alpha*tf.nn.l2_loss((new_weights[:,1:]-new_weights[:,:self.sys_para.steps+3])/self.sys_para.dt) 
            #+ dwdt_reg_alpha*tf.nn.l2_loss((self.ops_weight[:,0] + self.ops_weight[:,self.sys_para.steps])/self.sys_para.dt)



            # Limiting the d2wdt2 of control pulse
            self.d2wdt2_reg_alpha_coeff = tf.placeholder(tf.float32,shape=[])
            d2wdt2_reg_alpha = self.d2wdt2_reg_alpha_coeff/float(self.sys_para.steps)
            self.reg_loss = self.reg_loss + d2wdt2_reg_alpha*tf.nn.l2_loss((new_weights[:,2:] -\
                            2*new_weights[:,1:self.sys_para.steps+3] +new_weights[:,:self.sys_para.steps+2])/(self.sys_para.dt**2))
            #+d2wdt2_reg_alpha*tf.nn.l2_loss((self.ops_weight[:,1]-self.ops_weight[:,0] - 2*self.ops_weight[:,self.sys_para.steps] + self.ops_weight[:,self.sys_para.steps-1] )/(self.sys_para.dt**2))

            # Limiting the access to forbidden states
            self.inter_reg_alpha_coeff = tf.placeholder(tf.float32,shape=[])
            inter_reg_alpha = self.inter_reg_alpha_coeff/float(self.sys_para.steps)

            for inter_vec in self.inter_vecs:
                for state in self.sys_para.states_forbidden_list:
                    forbidden_state_pop = tf.square(inter_vec[state,:]) +\
                                        tf.square(inter_vec[self.sys_para.state_num+state,:])
                    self.reg_loss = self.reg_loss + inter_reg_alpha * tf.nn.l2_loss(forbidden_state_pop)

            ends = 1/float(self.sys_para.steps)
            end_steps = int(self.sys_para.steps *0.05)

            #self.reg_loss = self.reg_loss + ends*(tf.add(tf.nn.l2_loss(self.ops_weight[:,:end_steps]),tf.nn.l2_loss(self.ops_weight[:,self.sys_para.steps-end_steps:])))
           
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
            self.init_tf_ops()
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
            #tf.train.SummaryWriter.add_graph(graph)
            #tf.train.SummaryWriter('./tmp/graph',graph)
        
        return graph
