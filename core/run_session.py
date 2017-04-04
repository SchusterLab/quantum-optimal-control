import numpy as np
import tensorflow as tf
from Analysis import Analysis
import os
import time
from scipy.optimize import minimize

from helper_functions.datamanagement import H5File


class run_session:
    def __init__(self, tfs,graph,conv,sys_para,method,show_plots=True,single_simulation = False,use_gpu =True):
        self.tfs=tfs
        self.graph = graph
        self.conv = conv
        self.sys_para = sys_para
        self.update_step = conv.update_step
        self.iterations = 0
        self.method = method.upper()
        self.show_plots = show_plots
        self.target = False
        if not use_gpu:
            config = tf.ConfigProto(device_count = {'GPU': 0})
        else:
            config = None
        
        with tf.Session(graph=graph, config = config) as self.session:
            
            tf.global_variables_initializer().run()

            print "Initialized"
            
            if self.method == 'EVOLVE':
                self.start_time = time.time()
                x0 = self.sys_para.ops_weight_base
                self.l,self.rl,self.grads,self.metric,self.g_squared=self.get_error(x0)
                self.get_end_results()
                
            else:
                if self.method != 'ADAM': #Any BFGS scheme
                    self.bfgs_optimize(method=self.method)

                if self.method =='ADAM':
                    self.start_adam_optimizer()    
                
                  
    def create_tf_vectors(self,vec_traj):
        tf_initial_vectors = []
        jj = 0
        for initial_vector in self.sys_para.initial_vectors:
            
            tf_initial_vector = np.array(initial_vector)
            for ii in range (vec_traj[jj]):
                tf_initial_vectors.append(tf_initial_vector)
            jj = jj + 1
        #tf_initial_vectors = np.transpose((tf_initial_vectors))
        
        return tf_initial_vectors
    
    def create_target_vectors(self,vec_traj):
        target_vectors = []
        jj = 0
        for target_vector in self.sys_para.target_vectors:
            
            tf_target_vector = np.array(target_vector)
            for ii in range (vec_traj[jj]):
                target_vectors.append(tf_target_vector)
            jj = jj + 1
        #target_vectors = np.transpose((target_vectors))
        
        return target_vectors
    
    def start_adam_optimizer(self):
        # adam optimizer  
        self.start_time = time.time()
        self.end = False
        while True:
            
            learning_rate = float(self.conv.rate) * np.exp(-float(self.iterations) / self.conv.learning_rate_decay)
            if self.sys_para.traj:
                self.traj_num = self.sys_para.trajectories
                
                num_psi0 = len(self.sys_para.initial_vectors)
                needed_traj = np.zeros([num_psi0])
                start = (np.zeros([num_psi0]))
                end  = (np.zeros([num_psi0]))
                vec_trajs = np.ones([num_psi0])
                
                for kk in range (num_psi0):
                    vec_trajs = np.zeros([num_psi0])
                    vec_trajs[kk] = 1
                    self.feed_dict = {self.tfs.learning_rate: 0, self.tfs.start: start, self.tfs.end: end,self.tfs.num_trajs:vec_trajs}
                    self.grad_pack, self.norms = self.session.run(
                [self.tfs.grad_pack,self.tfs.norms], feed_dict=self.feed_dict)
                    #print self.norms
                
                    needed_traj[kk] = self.traj_num-np.rint(self.norms* self.traj_num) 
                    start[kk] = self.norms
                    end[kk] = 1.0
                    print "Generating "+str(needed_traj[kk])+" jump trajectories for state "+str(kk)+". Last norm is "+str(start[kk])
                self.feed_dict = {self.tfs.learning_rate: 0, self.tfs.start: start, self.tfs.end: end,self.tfs.num_trajs:needed_traj}
                self.g_squared, self.l, self.rl, self.metric,self.norms,self.r,self.a,self.vecs,self.tar,self.final = self.session.run(
                [self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss, self.tfs.unitary_scale,self.tfs.norms,self.tfs.r,self.tfs.a,self.tfs.vecs,self.tfs.target_vecs,self.tfs.final_state], feed_dict=self.feed_dict)
                
                #print self.r
                #print self.norms
            
            
            else: 
                self.feed_dict = {self.tfs.learning_rate: learning_rate}

                self.g_squared, self.l, self.rl, self.metric = self.session.run(
                [self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss, self.tfs.unitary_scale], feed_dict=self.feed_dict)
                
            if (self.l < self.conv.conv_target) or (self.g_squared < self.conv.min_grad) \
                    or (self.iterations >= self.conv.max_iterations):
                self.end = True

            self.update_and_save()
                
            if self.end:
                self.get_end_results()
                break
                
            
            

            _ = self.session.run([self.tfs.optimizer], feed_dict=self.feed_dict)

                
                

            
    def update_and_save(self):
        
        if not self.end:

            if (self.iterations % self.conv.update_step == 0):
                self.anly = Analysis(self.sys_para, self.tfs.final_state, self.tfs.ops_weight, self.tfs.unitary_scale,
                                     self.tfs.inter_vecs,self.feed_dict)
                
                self.save_data()
                self.display()
            if (self.iterations % self.conv.evol_save_step == 0):
                if not (self.sys_para.show_plots == True and (self.iterations % self.conv.update_step == 0)):
                    self.anly = Analysis(self.sys_para, self.tfs.final_state, self.tfs.ops_weight, self.tfs.unitary_scale,
                                         self.tfs.inter_vecs,self.feed_dict)
                    if not (self.iterations % self.conv.update_step == 0):
                        self.save_data()
                    self.conv.save_evol(self.anly)

            self.iterations += 1
    
    def get_end_results(self):
        # get optimized pulse and propagation
        
        # get and save inter vects
        
        self.anly = Analysis(self.sys_para, self.tfs.final_state, self.tfs.ops_weight, self.tfs.unitary_scale,
                                     self.tfs.inter_vecs,self.feed_dict)
        self.save_data()
        self.display()
        if not self.show_plots:  
            self.conv.save_evol(self.anly)
        
        self.uks = self.Get_uks()
        if not self.sys_para.state_transfer:
            self.Uf = self.anly.get_final_state()
        else:
            self.Uf = []
    
    def Get_uks(self): 
        # to get the pulse amplitudes
        uks = self.anly.get_ops_weight()
        for ii in range (len(uks)):
            uks[ii] = self.sys_para.ops_max_amp[ii]*uks[ii]
        return uks    

    def get_error(self,uks):
        #get error and gradient for scipy bfgs:
        self.session.run(self.tfs.ops_weight_base.assign(uks))

        g,l,rl,metric,g_squared = self.session.run([self.tfs.grad_pack, self.tfs.loss, self.tfs.reg_loss, self.tfs.unitary_scale, self.tfs.grad_squared])
        
        final_g = np.transpose(np.reshape(g,(len(self.sys_para.ops_c)*self.sys_para.steps)))

        return l,rl,final_g,metric, g_squared
    
    def save_data(self):
        if self.sys_para.save:
            self.elapsed = time.time() - self.start_time
            with H5File(self.sys_para.file_path) as hf:
                hf.append('error', np.array(self.l))
                hf.append('reg_error', np.array(self.rl))
                hf.append('uks', np.array(self.Get_uks()))
                hf.append('iteration', np.array(self.iterations))
                hf.append('run_time', np.array(self.elapsed))
                hf.append('unitary_scale', np.array(self.metric))
    
    
    def display(self):
        # display of simulation results

        if self.show_plots:
            self.conv.update_plot_summary(self.l, self.rl, self.anly)
        else:
            print 'Error = :%1.2e; Runtime: %.1fs; Iterations = %d, grads =  %10.3e, unitary_metric = %.5f' % (
            self.l, self.elapsed, self.iterations, self.g_squared, self.metric)
    
    
    def minimize_opt_fun(self,x):
        # minimization function called by scipy in each iteration
        self.l,self.rl,self.grads,self.metric,self.g_squared=self.get_error(np.reshape(x,(len(self.sys_para.ops_c),len(x)/len(self.sys_para.ops_c))))
        
        if self.l <self.conv.conv_target :
            self.conv_time = time.time()-self.start_time
            self.conv_iter = self.iterations
            self.end = True
            print 'Target fidelity reached'
            self.grads= 0*self.grads # set zero grads to terminate the scipy optimization
        
        self.update_and_save()
        
        if self.method == 'L-BFGS-B':
            return np.float64(self.rl),np.float64(np.transpose(self.grads))
        else:
            return self.rl,np.reshape(np.transpose(self.grads),[len(np.transpose(self.grads))])

    
    def bfgs_optimize(self, method='L-BFGS-B',jac = True, options=None):
        # scipy optimizer
        self.conv.reset_convergence()
        self.first=True
        self.conv_time = 0.
        self.conv_iter=0
        self.end=False
        print "Starting " + self.method + " Optimization"
        self.start_time = time.time()
        
        x0 = self.sys_para.ops_weight_base
        options={'maxfun' : self.conv.max_iterations,'gtol': self.conv.min_grad, 'disp':False,'maxls': 40}
        
        res = minimize(self.minimize_opt_fun,x0,method=method,jac=jac,options=options)

        uks=np.reshape(res['x'],(len(self.sys_para.ops_c),len(res['x'])/len(self.sys_para.ops_c)))

        print self.method + ' optimization done'
        
        g, l,rl = self.session.run([self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss])
            
        if self.sys_para.show_plots == False:
            print res.message
            print("Error = %1.2e" %l)
            print ("Total time is " + str(time.time() - self.start_time))
            
        self.get_end_results()          

    
        
