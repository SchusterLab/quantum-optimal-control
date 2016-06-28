import numpy as np
import tensorflow as tf
from runtime_functions.Analysis import Analysis
import os
import time
import scipy
from scipy.optimize import minimize

class run_session:
    def __init__(self, tfs,graph,conv,sys_para,method,show_plots=True,single_simulation = False,switch = True):
        self.tfs=tfs
        self.graph = graph
        self.conv = conv
        self.sys_para = sys_para
        self.update_step = conv.update_step
        self.iterations = 0
        self.method = method
        self.switch = switch
        self.show_plots = show_plots
        
        
        with tf.Session(graph=graph) as self.session:
            tf.initialize_all_variables().run()


            print "Initialized"
            iterations = 0
            start_time = time.time()
            
            if (single_simulation == False):
                max_iterations = conv.max_iterations
            else:
                max_iterations = 0
            
            
                
            
            if self.method != 'Adam':
                learning_rate=0
                self.feed_dict = {tfs.learning_rate : learning_rate, tfs.z_reg_alpha_coeff: conv.z_reg_alpha_coeff,
                            tfs.reg_alpha_coeff: conv.reg_alpha_coeff, 
                             tfs.dwdt_reg_alpha_coeff: conv.dwdt_reg_alpha_coeff,
                             tfs.d2wdt2_reg_alpha_coeff: conv.d2wdt2_reg_alpha_coeff,
                             tfs.inter_reg_alpha_coeff:conv.inter_reg_alpha_coeff}
                if self.sys_para.Dts==[]:
                    g,l,rl,uks = self.session.run([tfs.grad_pack, tfs.loss, tfs.reg_loss,tfs.ops_weight_base], feed_dict=self.feed_dict)
                else:
                    g,l,rl,uks = self.session.run([tfs.grad_pack, tfs.loss, tfs.reg_loss,tfs.raws], feed_dict=self.feed_dict)
                
                self.start_time = time.time()
                myfactr = 1e-15
                ftol = myfactr * np.finfo(float).eps
                res=self.optimize(uks, method=self.method,jac = True, options={'maxfun' : self.conv.max_iterations,'gtol': self.conv.min_grad, 'disp':False,'ftol':ftol})
                
            if self.method =='Adam':
                
                
                
                while True:
                    learning_rate = float(self.conv.rate) * np.exp(-float(self.iterations)/conv.learning_rate_decay)
                    self.feed_dict = {tfs.learning_rate : learning_rate, tfs.z_reg_alpha_coeff: conv.z_reg_alpha_coeff,
                            tfs.reg_alpha_coeff: conv.reg_alpha_coeff, 
                             tfs.dwdt_reg_alpha_coeff: conv.dwdt_reg_alpha_coeff,
                             tfs.d2wdt2_reg_alpha_coeff: conv.d2wdt2_reg_alpha_coeff,
                             tfs.inter_reg_alpha_coeff:conv.inter_reg_alpha_coeff}
                    g,_, l,rl = self.session.run([tfs.grad_squared, tfs.optimizer, tfs.loss, tfs.reg_loss], feed_dict=self.feed_dict)
                    
            
                    
                    if (self.iterations % self.conv.update_step == 0) or (l < self.conv.conv_target) or (g < self.conv.min_grad):    
                        
                        if self.show_plots:
                        # Plot convergence
                            if self.sys_para.multi:
                                self.anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.xy_weight, self.tfs.xy_nocos, self.tfs.unitary_scale,self.tfs.inter_vecs)
                            else:
                                self.anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.ops_weight, self.tfs.ops_weight, self.tfs.unitary_scale,self.tfs.inter_vecs, raw_weight =self.tfs.raw_weight, raws = self.tfs.raws)
                            self.conv.update_convergence(l,rl,self.anly,self.show_plots)

                        # Save the variables to disk.
                            this_dir = os.path.dirname(__file__)
                            tmp_path = os.path.join(this_dir,'../tmp/grape.ckpt')
                            save_path = tfs.saver.save(self.session, tmp_path)
                            if (self.iterations >= max_iterations) or (l < self.conv.conv_target): #(l<conv.conv_target) or (iterations>=conv.max_iterations):
                                
                                self.uks= self.Get_uks()
                                self.Uf = self.anly.get_final_state()
                            #anly.get_xy_weight()
                            #if sys_para.Modulation:
                                #anly.get_nonmodulated_weight() 
                                break
                        else:
                            elapsed = time.time() - start_time
                            print 'Error = %.9f; Runtime: %.1fs; Iterations = %d, grads =  %10.3e'%(l,elapsed,self.iterations,g)
                            if (self.iterations >= max_iterations) or (l < self.conv.conv_target) or (g < self.conv.min_grad): #(l<conv.conv_target) or (iterations>=conv.max_iterations):
                                if self.sys_para.multi:
                                    self.anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.xy_weight, self.tfs.xy_nocos, self.tfs.unitary_scale,self.tfs.inter_vecs)
                                else:
                                    self.anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.ops_weight, self.tfs.ops_weight, self.tfs.unitary_scale,self.tfs.inter_vecs, raw_weight =self.tfs.raw_weight, raws = self.tfs.raws)
                                
                                self.conv.update_convergence(l,rl,self.anly,True)
                                print 'Error = %.9f; Runtime: %.1fs; grads =  %10.3e'%(l,elapsed,g)
                                self.uks= self.Get_uks()
                                self.Uf = self.anly.get_final_state()
                            #anly.get_xy_weight()
                            #if sys_para.Modulation:
                                #anly.get_nonmodulated_weight() 
                                break


                    self.iterations+=1
                    
                      
    def Sort_back(self,uks):
        uks_original = []
        for op in self.sys_para.Hnames_original:
            index = self.sys_para.Hnames.tolist().index(op)
            uks_original.append(uks[index])
        return uks_original
        
    def Get_uks(self):
        if self.sys_para.Dts==[]:
            uks = self.anly.get_ops_weight()
            for ii in range (len(uks)):
                uks[ii] = self.sys_para.ops_max_amp[ii]*uks[ii]
        else:
            uks = []
            end = 0
            raws = self.anly.get_raws()
            for ii in range (self.sys_para.ops_len - len(self.sys_para.Dts)):
                start = ii*self.sys_para.steps
                end = (ii+1)*self.sys_para.steps
                new_uk = self.sys_para.ops_max_amp[ii]*raws[:,start:end]
                
                uks.append(np.reshape(new_uk,[len(np.transpose(new_uk))]))
            for jj in range (len(self.sys_para.Dts)):
                start = end
                end = start + self.sys_para.ctrl_steps[jj]
                new_uk = self.sys_para.ops_max_amp[jj +self.sys_para.ops_len - len(self.sys_para.Dts)]*raws[:,start:end]
                uks.append(np.reshape(new_uk,[len(np.transpose(new_uk))]))
                
            uks = self.Sort_back(uks)
        return uks    

    def evolve(self,uks):
        
        if self.sys_para.Dts==[]:
            self.session.run(self.tfs.ops_weight_base.assign(uks))
        else:
            self.session.run(self.tfs.raws.assign(uks))
        g,l,rl = self.session.run([self.tfs.grad_pack, self.tfs.loss, self.tfs.reg_loss], feed_dict=self.feed_dict)
        if self.sys_para.Dts==[]:
            final_g = np.transpose(np.reshape(g,(len(self.sys_para.ops_c)*self.sys_para.steps)))
        else:
            final_g = np.reshape(g,uks.shape)
        return l,rl,final_g
    

    
    def minimize_opt_fun(self,x):
        if self.sys_para.Dts==[]:
            l,rl,grads=self.evolve(np.reshape(x,(len(self.sys_para.ops_c),len(x)/len(self.sys_para.ops_c))))
        else:
            l,rl,grads=self.evolve(np.reshape(x,[1,len(x)]))
        #l,rl,grads=self.evolve(x)
        
        if self.iterations % self.update_step == 0:
            g, l,rl = self.session.run([self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss], feed_dict=self.feed_dict)
            
            if self.show_plots:
                self.anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.ops_weight, self.tfs.ops_weight, self.tfs.unitary_scale,self.tfs.inter_vecs, raw_weight =self.tfs.raw_weight, raws = self.tfs.raws)
                self.conv.update_convergence(l,rl,self.anly,True)
            else:
                elapsed = time.time() - self.start_time
                print 'Error = %.9f; Runtime: %.1fs; Iterations = %d, grads =  %10.3e'%(l,elapsed,self.iterations,g)
        self.iterations=self.iterations+1
        
        
            
        #print np.shape(rl)
        #print np.shape(grads)
        if self.method == 'L-BFGS-B':
            return np.float64(rl),np.float64(np.transpose(grads))
        else:
            return rl,np.reshape(np.transpose(grads),[len(np.transpose(grads))])

    def optimize(self,x0, method='L-BFGS-B',jac = False, options=None):
        self.conv.reset_convergence()
        #print np.shape(x0)

        res = minimize(self.minimize_opt_fun,x0,method=method,jac=jac,options=options)
        if self.sys_para.Dts==[]:
            uks=np.reshape(res['x'],(len(self.sys_para.ops_c),len(res['x'])/len(self.sys_para.ops_c)))
        else:
            uks = res['x']
        print self.method + ' optimization done'
        
        g, l,rl = self.session.run([self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss], feed_dict=self.feed_dict)
        
        
        if l > self.conv.conv_target and self.switch == True and self.iterations < self.conv.max_iterations:
            self.method ='Adam'
            print 'Switching to Adam optimizer'
        else:
            if self.sys_para.show_plots == False:
                self.anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.ops_weight, self.tfs.ops_weight, self.tfs.unitary_scale,self.tfs.inter_vecs, raw_weight =self.tfs.raw_weight, raws = self.tfs.raws)
                self.conv.update_convergence(l,rl,self.anly,True)
            self.uks= self.Get_uks()
            
            self.Uf = self.anly.get_final_state()
            if self.show_plots == False:
                print l
            #print res.message
            
        return res, uks
    
        
