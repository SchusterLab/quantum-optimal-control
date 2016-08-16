import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import display
from helper_functions.grape_functions import sort_ev


class ConvergenceGeneral:
    

    
    def reset_convergence(self):
        self.costs=[]
        self.reg_costs = []
        self.iterations=[]
        self.learning_rate=[]
        self.last_iter = 0
        self.accumulate_rate = 1.00

    def update_convergence(self,last_cost, last_reg_cost, anly,show_plots=True):
        if len(self.sys_para.states_concerned_list) > 8:
            self.concerned = [0,1,2,3,4,5,6,7]
        else:
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
		#self.anly.get_xy_weight()
		#self.anly.get_nonmodulated_weight()
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
        if self.sys_para.states_forbidden_list!= []:
            for forbid in self.sys_para.states_forbidden_list:
                if self.sys_para.forbid_dressed:
                    forbidden = forbidden +np.array(pop_inter_vecs[forbid,:])
                else:
                    v_sorted=sort_ev(self.sys_para.v_c,self.sys_para.dressed)
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
            if self.sys_para.evolve_error:
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
    #plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(self.sys_para.ops_max_amp[0]*ops_weight[1,:]),'c',label='y')
        #plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),(self.sys_para.qm_g1/(2*np.pi))\
         #    *np.array(self.sys_para.ops_max_amp[1]*ops_weight[2,:]),'g',label='(g/2pi)z')
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
                #plt.plot(np.array([self.sys_para.Dts[kk]* ii for ii in range(self.sys_para.ctrl_steps[kk])]),np.array(3*np.transpose(raw_weight[kk])),label=self.sys_para.Hnames[self.sys_para.ops_len -len(self.sys_para.Dts)+kk])

                plt.plot(np.array([self.sys_para.Dts[kk]* ii for ii in range(self.sys_para.ctrl_steps[kk])]),np.array(self.sys_para.ops_max_amp[self.sys_para.ops_len -len(self.sys_para.Dts) +kk]*np.transpose(raw_weight[kk])),label=self.sys_para.Hnames[self.sys_para.ops_len -len(self.sys_para.Dts)+kk])
            
            plt.title('Optimized Non interpolated pulses')
            plt.ylabel('Amplitude')
            plt.xlabel('Time ('+ self.time_unit+')')
            plt.legend()
     
        ## state evolution
        inter_vecs = self.anly.get_inter_vecs()
        
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
        if self.sys_para.evolve_error:
            print "Error = %.9f"%self.last_cost
	        

    def __init__(self):
	plt.figure()
