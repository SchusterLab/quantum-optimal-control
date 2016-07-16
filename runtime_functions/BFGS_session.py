import numpy as np
import tensorflow as tf
from runtime_functions.Analysis import Analysis
import os
import time
import scipy
from scipy.optimize import minimize

class BFGS_Session:
    def __init__(self, tfs,graph,conv,sys_para,method,show_plots=True,single_simulation = False):
        self.tfs=tfs
        self.graph = graph
        self.conv = conv
        self.sys_para = sys_para
        self.update_step = conv.update_step
        self.iterations = 0
        self.method = method
        print method
        
        with tf.Session(graph=graph) as self.session:
            tf.initialize_all_variables().run()


            print "Initialized"
            iterations = 0
            start_time = time.time()
            
            if (single_simulation == False):
                max_iterations = conv.max_iterations
            else:
                max_iterations = 0
            
            if self.method == 'Adam':
                learning_rate = float(conv.rate) * np.exp(-float(iterations)/conv.learning_rate_decay)
            elif self.method == 'BFGS':
                learning_rate = 0

            self.feed_dict = {tfs.learning_rate : learning_rate, tfs.z_reg_alpha_coeff: conv.z_reg_alpha_coeff,
                            tfs.reg_alpha_coeff: conv.reg_alpha_coeff, 
                             tfs.dwdt_reg_alpha_coeff: conv.dwdt_reg_alpha_coeff,
                             tfs.d2wdt2_reg_alpha_coeff: conv.d2wdt2_reg_alpha_coeff,
                             tfs.inter_reg_alpha_coeff:conv.inter_reg_alpha_coeff}
            if self.method == 'BFGS':
                
                g,l,rl,uks = self.session.run([tfs.grad_pack, tfs.loss, tfs.reg_loss,tfs.ops_weight], feed_dict=self.feed_dict)
                self.start_time = time.time()
                res=self.optimize(uks, method='BFGS',jac = True, options={'maxiter' : self.conv.max_iterations,'gtol': 1e-15, 'disp':True},show_plots=True)
            elif self.method =='Adam':
                
                
                while True:
                    g,_, l,rl = self.session.run([tfs.grad_squared, tfs.optimizer, tfs.loss, tfs.reg_loss], feed_dict=self.feed_dict)
            
                    
                    if (self.iterations % self.conv.update_step == 0) or (l < self.conv.conv_target) or (g < self.conv.min_grad):    
                        if self.sys_para.show_plots:
                        # Plot convergence
                            if self.sys_para.multi:
                                anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.xy_weight, self.tfs.xy_nocos, self.tfs.unitary_scale,self.tfs.inter_vecs)
                            else:
                                anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.ops_weight, self.tfs.ops_weight, self.tfs.unitary_scale,self.tfs.inter_vecs, raw_weight =self.tfs.raw_weight)
                            self.conv.update_convergence(l,rl,anly,show_plots)

                        # Save the variables to disk.
                            this_dir = os.path.dirname(__file__)
                            tmp_path = os.path.join(this_dir,'../tmp/grape.ckpt')
                            save_path = tfs.saver.save(self.session, tmp_path)
                            if (self.iterations >= max_iterations) or (l < self.conv.conv_target): #(l<conv.conv_target) or (iterations>=conv.max_iterations):
                                anly.get_ops_weight()
                            #anly.get_xy_weight()
                            #if sys_para.Modulation:
                                #anly.get_nonmodulated_weight() 
                                break
                        else:
                            elapsed = time.time() - start_time
                            print 'Error = %.9f; Runtime: %.1fs; Iterations = %d, grads =  %10.3e'%(l,elapsed,self.iterations,g)
                            if (self.iterations >= max_iterations) or (l < self.conv.conv_target) or (g < self.conv.min_grad): #(l<conv.conv_target) or (iterations>=conv.max_iterations):
                                if self.sys_para.multi:
                                    anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.xy_weight, self.tfs.xy_nocos, self.tfs.unitary_scale,self.tfs.inter_vecs)
                                else:
                                    anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.ops_weight, self.tfs.ops_weight, self.tfs.unitary_scale,self.tfs.inter_vecs, raw_weight =self.tfs.raw_weight)
                                self.conv.update_convergence(l,rl,anly,show_plots)
                                print 'Error = %.9f; Runtime: %.1fs; Iterations = %d, grads =  %10.3e'%(l,elapsed,iterations,g)
                                anly.get_ops_weight()
                            #anly.get_xy_weight()
                            #if sys_para.Modulation:
                                #anly.get_nonmodulated_weight() 
                                break


                    iterations+=1


            
           

           
        

    def evolve(self,uks):
        
        self.session.run(self.tfs.ops_weight.assign(uks))
        g,l,rl = self.session.run([self.tfs.grad_pack, self.tfs.loss, self.tfs.reg_loss], feed_dict=self.feed_dict)
        
        return l,rl,np.transpose(np.reshape(g,(len(self.sys_para.ops_c)*self.sys_para.steps)))
    

    
    def minimize_opt_fun(self,x):
        l,rl,grads=self.evolve(np.reshape(x,(len(self.sys_para.ops_c),len(x)/len(self.sys_para.ops_c))))
        #l,rl,grads=self.evolve(x)
        
        if self.iterations % self.update_step == 0:
            g, l,rl = self.session.run([self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss], feed_dict=self.feed_dict)
            
            if self.sys_para.show_plots:
                anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.ops_weight, self.tfs.ops_weight, self.tfs.unitary_scale,self.tfs.inter_vecs, raw_weight =self.tfs.raw_weight)
                self.conv.update_convergence(l,rl,anly,True)
            else:
                elapsed = time.time() - self.start_time
                print 'Error = %.9f; Runtime: %.1fs; Iterations = %d, grads =  %10.3e'%(l,elapsed,self.iterations,g)
        self.iterations=self.iterations+1
        #print np.shape(rl)
        #print np.shape(grads)
        return rl,np.transpose(grads)

    def optimize(self,x0, method='Nelder-Mead',jac = False, options=None,show_plots=False):
        self.conv.reset_convergence()
        #print np.shape(x0)

        res = minimize(self.minimize_opt_fun,x0,method=method,jac=jac,options=options)
        uks=np.reshape(res['x'],(len(self.sys_para.ops_c),len(res['x'])/len(self.sys_para.ops_c)))
        if show_plots: 
            g,_, l,rl = self.session.run([self.tfs.grad_squared, self.tfs.optimizer, self.tfs.loss, self.tfs.reg_loss], feed_dict=self.feed_dict)
            anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.ops_weight, self.tfs.ops_weight, self.tfs.unitary_scale,self.tfs.inter_vecs, raw_weight =self.tfs.raw_weight)
            self.conv.update_convergence(l,rl,anly,True)
        print l
        print res.message
            
        return res, uks

