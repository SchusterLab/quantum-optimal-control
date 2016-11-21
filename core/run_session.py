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
        self.method = method
        self.show_plots = show_plots
        self.BFGS_time =0
        self.target = False
        if not use_gpu:
            config = tf.ConfigProto(device_count = {'GPU': 0})
        else:
            config = None
        
        with tf.Session(graph=graph, config = config) as self.session:
            
            tf.initialize_all_variables().run()


            print "Initialized"
            iterations = 0
            max_iterations = conv.max_iterations
 
            if self.method != 'Adam': #Any BFGS scheme
                learning_rate=0
                self.feed_dict = {tfs.learning_rate : learning_rate}
                
                g,l,rl,uks = self.session.run([tfs.grad_pack, tfs.loss, tfs.reg_loss,tfs.ops_weight_base], feed_dict=self.feed_dict)
                
                myfactr = 1e-20
                ftol = myfactr * np.finfo(float).eps
                res=self.optimize(uks, method=self.method,jac = True, options={'maxfun' : self.conv.max_iterations,'gtol': self.conv.min_grad, 'disp':False,'ftol':ftol, 'maxls': 40, 'factr':-10.0})
                
            if self.method =='Adam':
                
                
                start_time = time.time() - self.BFGS_time
                while True:
                    learning_rate = float(self.conv.rate) * np.exp(-float(self.iterations)/conv.learning_rate_decay)
                    self.feed_dict = {tfs.learning_rate : learning_rate}
                    
                    g,_, l,rl, metric= self.session.run([tfs.grad_squared, tfs.optimizer, tfs.loss, tfs.reg_loss, tfs.unitary_scale], feed_dict=self.feed_dict)
                    
            
                    
                    if (self.iterations % self.conv.update_step == 0) or (l < self.conv.conv_target) or (g < self.conv.min_grad):    
                        elapsed = time.time() - start_time
                        if self.sys_para.save:
                            iter_num = self.iterations
                
                            self.anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.unitary_scale,self.tfs.inter_vecs)
                            with H5File(self.sys_para.file_path) as hf:
                                hf.append('error',np.array(l))
                                hf.append('reg_error',np.array(rl))
                                hf.append('uks',np.array(self.Get_uks()))
                                hf.append('iteration',np.array(self.iterations))
                                hf.append('run_time',np.array(elapsed))
                                hf.append('unitary_scale',np.array(metric))
                            
                        if self.show_plots and (not self.sys_para.save):
                            self.anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.unitary_scale,self.tfs.inter_vecs)
                            
                        if self.show_plots:
                        # Plot convergence
                            
                            
                            self.conv.update_convergence(l,rl,self.anly,self.show_plots)

                            if (self.iterations >= max_iterations) or (l < self.conv.conv_target): 
                                
                                self.uks= self.Get_uks()
                                if not self.sys_para.state_transfer:
                                    self.Uf = self.anly.get_final_state()
                                else:
                                    self.Uf=[]
                            
                                break
                        else:
                            
                            print 'Error = :%1.2e; Runtime: %.1fs; Iterations = %d, grads =  %10.3e, unitary_metric = %.5f'%(l,elapsed,self.iterations,g, metric)
                            if (self.iterations >= max_iterations) or (l < self.conv.conv_target) or (g < self.conv.min_grad): 
                                
                                self.anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.unitary_scale,self.tfs.inter_vecs)
                                
                                self.conv.update_convergence(l,rl,self.anly,True)
                                print 'Error = :%1.2e; Runtime: %.1fs; grads =  %10.3e, unitary_metric = %.5f'%(l,elapsed,g,metric)
                                self.uks= self.Get_uks()
                                if not self.sys_para.state_transfer:
                                    self.Uf = self.anly.get_final_state(save=False)
                                else:
                                    self.Uf=[]
                           
                                break


                    self.iterations+=1
                    
                      
    def Sort_back(self,uks): # To restore the order of uks (pulse amplitudes) as in the input
        uks_original = []
        for op in self.sys_para.Hnames_original:
            index = self.sys_para.Hnames.tolist().index(op)
            uks_original.append(uks[index])
        return uks_original
        
    def Get_uks(self): # to get the pulse amplitudes in any scenario (including different time scales) 
        uks = self.anly.get_ops_weight()
        for ii in range (len(uks)):
            uks[ii] = self.sys_para.ops_max_amp[ii]*uks[ii]
        return uks    

    
#BFGS functions:
    def get_error(self,uks):
        
        self.session.run(self.tfs.ops_weight_base.assign(uks))

        g,l,rl = self.session.run([self.tfs.grad_pack, self.tfs.loss, self.tfs.reg_loss], feed_dict=self.feed_dict)
        
        final_g = np.transpose(np.reshape(g,(len(self.sys_para.ops_c)*self.sys_para.steps)))

        return l,rl,final_g
    

    
    def minimize_opt_fun(self,x):
        l,rl,grads=self.get_error(np.reshape(x,(len(self.sys_para.ops_c),len(x)/len(self.sys_para.ops_c))))
        
        #print l,self.iterations
        if l <self.conv.conv_target :
            self.conv_time = time.time()-self.start_time
            self.conv_iter = self.iterations
            self.target = True
            print 'Target fidelity reached'
            grads= 0*grads
        if self.iterations % self.update_step == 0 or self.target :
            g, l,rl,metric = self.session.run([self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss, self.tfs.unitary_scale], feed_dict=self.feed_dict)
            elapsed = time.time() - self.start_time
            if self.sys_para.save:
                iter_num = self.iterations
                
                self.anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.unitary_scale,self.tfs.inter_vecs)

                
                with H5File(self.sys_para.file_path) as hf:
                    hf.append('error',np.array(l))
                    hf.append('reg_error',np.array(rl))
                    hf.append('uks',np.array(self.Get_uks()))
                    hf.append('iteration',np.array(self.iterations))
                    hf.append('run_time',np.array(elapsed))
                    hf.append('unitary_scale',np.array(metric))
                
            if self.iterations ==0:
                self.start_time = time.time()
            if self.show_plots and (not self.sys_para.save):
                self.anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.unitary_scale,self.tfs.inter_vecs)
            if self.show_plots:
                
                self.conv.update_convergence(l,rl,self.anly,True)
            else:
                
                print 'Error = :%1.2e; Runtime: %.1fs; Iterations = %d, grads =  %10.3e, unitary_metric = %.5f'%(l,elapsed,self.iterations,g,metric)
        
        self.iterations+=1
        
        if self.method == 'L-BFGS-B':
            return np.float64(rl),np.float64(np.transpose(grads))
        else:
            return rl,np.reshape(np.transpose(grads),[len(np.transpose(grads))])

    def optimize(self,x0, method='L-BFGS-B',jac = False, options=None):
        
        self.conv.reset_convergence()
        self.first=True
        self.conv_time = 0.
        self.conv_iter=0
        #print np.shape(x0)
        print "Starting " + self.method + " Optimization"
        self.start_time = time.time()
        res = minimize(self.minimize_opt_fun,x0,method=method,jac=jac,options=options)

        uks=np.reshape(res['x'],(len(self.sys_para.ops_c),len(res['x'])/len(self.sys_para.ops_c)))

        print self.method + ' optimization done'
        
        g, l,rl = self.session.run([self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss], feed_dict=self.feed_dict)
            
        if self.sys_para.show_plots == False:
            self.anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.unitary_scale,self.tfs.inter_vecs)
            self.conv.update_convergence(l,rl,self.anly,True)
        self.uks= self.Get_uks()
            
        if not self.sys_para.state_transfer:
            self.Uf = self.anly.get_final_state(save=False)
        else:
            self.Uf=[]
        if self.show_plots == False:
            print res.message
            print("Error = %1.2e" %l)
            print ("Total time is " + str(time.time() - self.start_time))
                

        
            
        return res, uks
    
        
