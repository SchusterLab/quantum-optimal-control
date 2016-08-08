import numpy as np
import tensorflow as tf
from runtime_functions.Analysis import Analysis
import os
import time
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
        self.BFGS_time =0
        self.target = False
        
        with tf.Session(graph=graph) as self.session:
            
            tf.initialize_all_variables().run()


            print "Initialized"
            iterations = 0
            max_iterations = conv.max_iterations
 
            if self.method != 'Adam': #Any BFGS scheme
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
                    
               
                myfactr = 1e-20
                ftol = myfactr * np.finfo(float).eps
                res=self.optimize(uks, method=self.method,jac = True, options={'maxfun' : self.conv.max_iterations,'gtol': self.conv.min_grad, 'disp':False,'ftol':ftol, 'maxls': 40})
                
            if self.method =='Adam':
                
                
                start_time = time.time() - self.BFGS_time
                while True:
                    learning_rate = float(self.conv.rate) * np.exp(-float(self.iterations)/conv.learning_rate_decay)
                    self.feed_dict = {tfs.learning_rate : learning_rate, tfs.z_reg_alpha_coeff: conv.z_reg_alpha_coeff,
                            tfs.reg_alpha_coeff: conv.reg_alpha_coeff, 
                             tfs.dwdt_reg_alpha_coeff: conv.dwdt_reg_alpha_coeff,
                             tfs.d2wdt2_reg_alpha_coeff: conv.d2wdt2_reg_alpha_coeff,
                             tfs.inter_reg_alpha_coeff:conv.inter_reg_alpha_coeff}
                    
                    g,_, l,rl= self.session.run([tfs.grad_squared, tfs.optimizer, tfs.loss, tfs.reg_loss], feed_dict=self.feed_dict)
                    
            
                    
                    if (self.iterations % self.conv.update_step == 0) or (l < self.conv.conv_target) or (g < self.conv.min_grad):    
                        if self.show_plots:
                        # Plot convergence
                            
                            self.anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.ops_weight, self.tfs.ops_weight, self.tfs.unitary_scale,self.tfs.inter_vecs, raw_weight =self.tfs.raw_weight, raws = self.tfs.raws)
                            self.conv.update_convergence(l,rl,self.anly,self.show_plots)

                        # Save the variables to disk.
                            this_dir = os.path.dirname(__file__)
                            tmp_path = os.path.join(this_dir,'../Examples/tmp/grape.ckpt')
                            save_path = tfs.saver.save(self.session, tmp_path)
                            if (self.iterations >= max_iterations) or (l < self.conv.conv_target): 
                                
                                self.uks= self.Get_uks()
                                if not self.sys_para.state_transfer:
                                    self.Uf = self.anly.get_final_state()
                                else:
                                    self.Uf=[]
                            
                                break
                        else:
                            elapsed = time.time() - start_time
                            print 'Error = :%1.2e; Runtime: %.1fs; Iterations = %d, grads =  %10.3e'%(l,elapsed,self.iterations,g)
                            if (self.iterations >= max_iterations) or (l < self.conv.conv_target) or (g < self.conv.min_grad): 
                                
                                self.anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.ops_weight, self.tfs.ops_weight, self.tfs.unitary_scale,self.tfs.inter_vecs, raw_weight =self.tfs.raw_weight, raws = self.tfs.raws)
                                
                                self.conv.update_convergence(l,rl,self.anly,True)
                                print 'Error = :%1.2e; Runtime: %.1fs; grads =  %10.3e'%(l,elapsed,g)
                                self.uks= self.Get_uks()
                                if not self.sys_para.state_transfer:
                                    self.Uf = self.anly.get_final_state()
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

    
#BFGS functions:
    def get_error(self,uks):
        
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
            l,rl,grads=self.get_error(np.reshape(x,(len(self.sys_para.ops_c),len(x)/len(self.sys_para.ops_c))))
        else:
            l,rl,grads=self.get_error(np.reshape(x,[1,len(x)]))
        
        #print l,self.iterations
        if l <self.conv.conv_target :
            self.conv_time = time.time()-self.start_time
            self.conv_iter = self.iterations
            self.target = True
            print 'Target fidelity reached'
            grads= 0*grads
            print self.tfs.final_state.eval()
        if self.iterations % self.update_step == 0 or self.target :
            g, l,rl,metric = self.session.run([self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss, self.tfs.unitary_scale], feed_dict=self.feed_dict)
            #np.save('uks_QFT_7',x)
            if self.show_plots:
                self.anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.ops_weight, self.tfs.ops_weight, self.tfs.unitary_scale,self.tfs.inter_vecs, raw_weight =self.tfs.raw_weight, raws = self.tfs.raws)
                self.conv.update_convergence(l,rl,self.anly,True)
            else:
                elapsed = time.time() - self.start_time
                print 'Error = :%1.2e; Runtime: %.1fs; Iterations = %d, grads =  %10.3e, unitary_metric = %.5f'%(l,elapsed,self.iterations,g,metric)
        self.iterations=self.iterations+1
        
        
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
        if self.sys_para.Dts==[]:
            uks=np.reshape(res['x'],(len(self.sys_para.ops_c),len(res['x'])/len(self.sys_para.ops_c)))
        else:
            uks = res['x']
        print self.method + ' optimization done'
        
        g, l,rl = self.session.run([self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss], feed_dict=self.feed_dict)
        
        
        if l > self.conv.conv_target and self.switch == True and self.iterations < self.conv.max_iterations:
            self.method ='Adam'
            self.BFGS_time = time.time()-self.start_time
            print 'Switching to Adam optimizer'
        else:
            
            if self.sys_para.show_plots == False:
                self.anly = Analysis(self.sys_para,self.tfs.final_state,self.tfs.ops_weight,self.tfs.ops_weight, self.tfs.ops_weight, self.tfs.unitary_scale,self.tfs.inter_vecs, raw_weight =self.tfs.raw_weight, raws = self.tfs.raws)
                self.conv.update_convergence(l,rl,self.anly,True)
            self.uks= self.Get_uks()
            
            if not self.sys_para.state_transfer:
                self.Uf = self.anly.get_final_state()
            else:
                self.Uf=[]
            if self.show_plots == False:
                print res.message
                print("Error = %1.2e" %l)
                print ("Total time is " + str(time.time() - self.start_time))
                np.save('final',self.tfs.final_states.eval())
                np.save('target',self.tfs.target_states.eval())
                np.save('lol',self.anly.get_inter_vecs()[0])

        
            
        return res, uks
    
        
