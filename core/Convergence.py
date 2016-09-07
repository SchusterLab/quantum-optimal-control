import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import display
from helper_functions.grape_functions import sort_ev


class Convergence:
    
    def __init__(self,sys_para,time_unit,convergence):
        # paramters
        self.sys_para = sys_para
        self.Modulation = self.sys_para.Modulation
        self.Interpolation = self.sys_para.Interpolation
        self.time_unit = time_unit

        if 'rate' in convergence:
            self.rate = convergence['rate']
        else:
            self.rate = 0.01

        if 'update_step' in convergence:
            self.update_step = convergence['update_step']
        else:
            self.update_step = 100

        if 'conv_target' in convergence:
            self.conv_target = convergence['conv_target']
        else:
            self.conv_target = 1e-8

        if 'max_iterations' in convergence:
            self.max_iterations = convergence['max_iterations']
        else:
            self.max_iterations = 5000

        if 'learning_rate_decay' in convergence:
            self.learning_rate_decay = convergence['learning_rate_decay']
        else:
            self.learning_rate_decay = 2500

        if 'min_grad' in convergence:
            self.min_grad = convergence['min_grad']
        else:
            self.min_grad = 1e-25


        self.reset_convergence()
        plt.figure()
    
    def reset_convergence(self):
        self.costs=[]
        self.reg_costs = []
        self.iterations=[]
        self.learning_rate=[]
        self.last_iter = 0
        self.accumulate_rate = 1.00

    def update_convergence(self,last_cost, last_reg_cost, anly,show_plots=True):
        self.concerned = self.sys_para.states_concerned_list
        self.last_cost = last_cost
        self.last_reg_cost = last_reg_cost
          
        self.anly = anly

	if show_plots:
        	self.plot_summary()
	else:
		print '###### last cost: ' + str(last_cost) + ' ######'
        	if self.sys_para.state_transfer == False:
            		self.anly.get_final_state()
        	self.anly.get_ops_weight()
        	self.anly.get_inter_vecs()			
    
    def get_convergence(self):
        self.costs.append(self.last_cost)
        self.reg_costs.append(self.last_reg_cost)
        self.iterations.append(self.last_iter)
        self.last_iter+=self.update_step
 
    
    def plot_inter_vecs_general(self,pop_inter_vecs,start):
        
        if self.sys_para.draw_list !=[]:
            for kk in range(len(self.sys_para.draw_list)):
                plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps+1)]),np.array(pop_inter_vecs[self.sys_para.draw_list[kk],:]),label=self.sys_para.draw_names[kk])
                
        
        else:
            
            if start  > 4:
                plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps+1)]),np.array(pop_inter_vecs[start,:]),label='Starting level '+str(start))
                
            for jj in range(4):
                
                plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps+1)]),np.array(pop_inter_vecs[jj,:]),label='level '+str(jj))
            
        
        forbidden =np.zeros(self.sys_para.steps+1)
        if 'states_forbidden_list' in self.sys_para.reg_coeffs:
            for forbid in self.sys_para.reg_coeffs['states_forbidden_list']:
                if self.sys_para.dressed_info is None or ('forbid_dressed' in self.sys_para.reg_coeffs and self.sys_para.reg_coeffs['forbid_dressed']) :
                    forbidden = forbidden +np.array(pop_inter_vecs[forbid,:])
                else:
                    v_sorted=sort_ev(self.sys_para.v_c,self.sys_para.dressed_id)
                    dressed_vec= np.dot(v_sorted,np.sqrt(pop_inter_vecs))
                    forbidden = forbidden +np.array(np.square(np.abs(dressed_vec[forbid,:])))
                    
            plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps+1)]), forbidden,'c',label='forbidden')
        
        plt.ylabel('Population')
        plt.ylim(-0.1,1.1)
        plt.xlabel('Time ('+ self.time_unit+')')
        plt.legend(ncol=7)
  
    def plot_summary(self):
        

        if not self.last_iter == 0:
            self.runtime = time.time() - self.start_time
            self.estimated_runtime = float(self.runtime * (self.max_iterations-self.last_iter) / self.last_iter)/(60*60)
        else:
            self.start_time = time.time()
            self.runtime = 0
            self.estimated_runtime = 0

        
        self.get_convergence()
        i1=0
        i2=0
        if self.Modulation:
            i1=1
        if self.Interpolation or self.sys_para.dts!= []:
            i2=1
        if self.sys_para.state_transfer:
            i2 = i2-1
        if self.sys_para.evolve:
            gs = gridspec.GridSpec(2+i1+i2+len(self.concerned), 2)
        else:
            gs = gridspec.GridSpec(3+i1+i2+len(self.concerned), 2)
        
        index = 0
        ## cost
        if self.sys_para.evolve == False and self.sys_para.show_plots == True:
            
            
          
            plt.subplot(gs[index, :],title='Error = %1.2e; Other errors = %1.2e; Unitary Metric: %.5f; Runtime: %.1fs; Estimated Remaining Runtime: %.1fh' % (self.last_cost, self.last_reg_cost-self.last_cost,
                                                                                                   self.anly.tf_unitary_scale.eval(),
                                                                                                 
                                                                                                  self.runtime,
                                                                                                  self.estimated_runtime))
            
            index +=1
            plt.plot(np.array(self.iterations),np.array(self.costs),'bx-',label='Fidelity Error')
            plt.plot(np.array(self.iterations),np.array(self.reg_costs),'go-',label='All Penalties')
            plt.ylabel('Error')
            plt.xlabel('Iteration')
            plt.yscale('log')
            plt.legend()
        else:
            print "Error = %.9f"%self.last_cost
        ## unitary evolution
        if not self.sys_para.state_transfer:
            M = self.anly.get_final_state()
            plt.subplot(gs[index, 0],title="operator: real")
            plt.imshow(M.real,interpolation='none')
            plt.clim(-1,1)
            plt.colorbar()
            plt.subplot(gs[index, 1],title="operator: imaginary")
            plt.imshow(M.imag,interpolation='none')
            plt.clim(-1,1)
            plt.colorbar()
            index +=1
        
        ## operators
        plt.subplot(gs[index, :],title="Simulation Weights")
        ops_weight = self.anly.get_ops_weight()
            
        for jj in range (self.sys_para.ops_len):
           
            plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(self.sys_para.ops_max_amp[jj]*ops_weight[jj,:]),label='u'+self.sys_para.Hnames[jj])

        if self.sys_para.evolve:
            plt.title('Pulse')
        else:
            plt.title('Optimized pulse')
            
        plt.ylabel('Amplitude')
        plt.xlabel('Time ('+ self.time_unit+')')
        plt.legend()
        
        index+=1
        ## Control Fields
        if ( self.sys_para.Dts !=[]):
            plt.subplot(gs[index, :],title="Non-Interpolated Control Fields")
            index+=1
            
            
            raw_weight = self.anly.get_raw_weight()


            for kk in range (len(self.sys_para.Dts)):

                plt.plot(np.array([self.sys_para.Dts[kk]* ii for ii in range(self.sys_para.ctrl_steps[kk])]),np.array(self.sys_para.ops_max_amp[self.sys_para.ops_len -len(self.sys_para.Dts) +kk]*np.transpose(raw_weight[kk])),label=self.sys_para.Hnames[self.sys_para.ops_len -len(self.sys_para.Dts)+kk])
            
            plt.title('Optimized Non interpolated pulses')
            plt.ylabel('Amplitude')
            plt.xlabel('Time ('+ self.time_unit+')')
            plt.legend()
     
        ## state evolution
        inter_vecs = self.anly.get_inter_vecs()
        
        inter_vecs_array = np.array(inter_vecs)
        
        for ii in range(len(self.concerned)):
            plt.subplot(gs[index+ii, :],title="Evolution")

            pop_inter_vecs = inter_vecs[ii]
            self.plot_inter_vecs_general(pop_inter_vecs,self.concerned[ii])
                
                
        
        fig = plt.gcf()
        if self.sys_para.state_transfer:
            plots = 2
        else:
            plots = 3
        if self.sys_para.Dts !=[]:
            plots= plots+1
        
        
        fig.set_size_inches(15, int (plots+len(self.concerned)*18))
	
        display.display(plt.gcf())
        display.clear_output(wait=True)
