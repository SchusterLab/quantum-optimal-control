import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import display

class ConvergenceGeneral:

    
    def reset_convergence(self):
        self.costs=[]
        self.reg_costs = []
        self.iterations=[]
        self.learning_rate=[]
        self.last_iter = 0
        self.accumulate_rate = 1.00

    def update_convergence(self,last_cost, last_reg_cost, anly,show_plots=True):
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
        
    def plot_inter_vecs(self,pop_inter_vecs):
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(pop_inter_vecs[0,1:]),label='g00')
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(pop_inter_vecs[1,1:]),label='g01')
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(pop_inter_vecs[self.sys_para.mode_state_num,1:]),label='g10')
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(pop_inter_vecs[self.sys_para.mode_state_num+1,1:]),label='g11')
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)])
    ,np.array(pop_inter_vecs[self.sys_para.mode_state_num**2:2*self.sys_para.mode_state_num**2,1:].sum(axis=0))
             ,label='e(012)(012)')
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)])
    ,np.array(pop_inter_vecs[2*self.sys_para.mode_state_num**2:3*self.sys_para.mode_state_num**2,1:].sum(axis=0))
             ,label='f(012)(012)') 
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)])
             ,np.array(pop_inter_vecs[3*self.sys_para.mode_state_num**2:4*self.sys_para.mode_state_num**2,1:].sum(axis=0)) +\
             np.array(pop_inter_vecs[2,1:])+\
             np.array(pop_inter_vecs[self.sys_para.mode_state_num**2+2,1:])+\
             np.array(pop_inter_vecs[2*self.sys_para.mode_state_num**2+2,1:])+\
             np.array(pop_inter_vecs[2*self.sys_para.mode_state_num,1:])+\
             np.array(pop_inter_vecs[self.sys_para.mode_state_num**2+2*self.sys_para.mode_state_num,1:])+\
             np.array(pop_inter_vecs[2*self.sys_para.mode_state_num**2+2*self.sys_para.mode_state_num,1:])
             ,label='forbidden')
        
        plt.ylabel('Population')
        plt.ylim(0,1)
        plt.xlabel('Time ('+ self.time_unit+')')
        plt.legend(loc=6)
    
    
    def plot_inter_vecs_general(self,pop_inter_vecs,start):
        
        if self.sys_para.draw_list !=[]:
            for kk in range(len(self.sys_para.draw_list)):
                plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(pop_inter_vecs[self.sys_para.draw_list[kk],1:]),label=self.sys_para.draw_names[kk])
                
        
        else:
            
            if start  > 4:
                plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(pop_inter_vecs[start,1:]),label='Starting level '+str(start))
                
            for jj in range(4):
                
                plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(pop_inter_vecs[jj,1:]),label='level '+str(jj))
            
        
        forbidden =np.zeros(self.sys_para.steps)
        if self.sys_para.states_forbidden_list!= []:
            for forbid in self.sys_para.states_forbidden_list:
                forbidden = forbidden +np.array(pop_inter_vecs[forbid,1:])
            plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]), forbidden,label='forbidden')
        
        plt.ylabel('Population')
        plt.ylim(-0.1,1.1)
        plt.xlabel('Time ('+ self.time_unit+')')
        plt.legend(loc=6)
    
    def plot_inter_vecs_v2(self,pop_inter_vecs):
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(pop_inter_vecs[0,1:]),label='g00')
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(pop_inter_vecs[1,1:]),label='g01')
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(pop_inter_vecs[self.sys_para.mode_state_num,1:]),label='g10')
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(pop_inter_vecs[self.sys_para.mode_state_num+1,1:]),label='g11')
        
        h_state= range(self.sys_para.mode_state_num**2,2*self.sys_para.mode_state_num**2)
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)])
    ,np.array(pop_inter_vecs[h_state,1:].sum(axis=0))
             ,label='e(012)(012)')
        
        
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)])
    ,np.array(pop_inter_vecs[range(2*self.sys_para.mode_state_num**2,3*self.sys_para.mode_state_num**2),1:].sum(axis=0))
             ,label='f(012)(012)') 
        
       
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)])
             ,np.array(pop_inter_vecs[range(3*self.sys_para.mode_state_num**2,4*self.sys_para.mode_state_num**2),1:].sum(axis=0)) +\
             np.array(pop_inter_vecs[2,1:])+\
             np.array(pop_inter_vecs[self.sys_para.mode_state_num**2+2,1:])+\
             np.array(pop_inter_vecs[2*self.sys_para.mode_state_num**2+2,1:])+\
             np.array(pop_inter_vecs[2*self.sys_para.mode_state_num,1:])+\
             np.array(pop_inter_vecs[self.sys_para.mode_state_num**2+2*self.sys_para.mode_state_num,1:])+\
             np.array(pop_inter_vecs[2*self.sys_para.mode_state_num**2+2*self.sys_para.mode_state_num,1:])
             ,label='forbidden')
        
        plt.ylabel('Population')
        plt.ylim(0,1)
        plt.xlabel('Time ('+ self.time_unit+')')
        plt.legend(loc=6)
    def plot_inter_vecs_v3(self,pop_inter_vecs):
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(pop_inter_vecs[0,1:]),label='g00')
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(pop_inter_vecs[self.sys_para.mode_state_num**2,1:]),label='e00')
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(pop_inter_vecs[self.sys_para.mode_state_num,1:]),label='g10')
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)])
             ,np.array(pop_inter_vecs[self.sys_para.mode_state_num+self.sys_para.mode_state_num**2,1:]),label='e10')
#        plot(array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)])
#    ,np.array(pop_inter_vecs[self.sys_para.mode_state_num**2:2*self.sys_para.mode_state_num**2,1:].sum(axis=0))
#             ,label='e(012)(012)')
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)])
    ,np.array(pop_inter_vecs[2*self.sys_para.mode_state_num**2:3*self.sys_para.mode_state_num**2,1:].sum(axis=0))
             ,label='f(012)(012)') 
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)])
             ,np.array(pop_inter_vecs[3*self.sys_para.mode_state_num**2:4*self.sys_para.mode_state_num**2,1:].sum(axis=0)) +\
             np.array(pop_inter_vecs[2,1:])+\
             np.array(pop_inter_vecs[self.sys_para.mode_state_num**2+2,1:])+\
             np.array(pop_inter_vecs[2*self.sys_para.mode_state_num**2+2,1:])+\
             np.array(pop_inter_vecs[2*self.sys_para.mode_state_num,1:])+\
             np.array(pop_inter_vecs[self.sys_para.mode_state_num**2+2*self.sys_para.mode_state_num,1:])+\
             np.array(pop_inter_vecs[2*self.sys_para.mode_state_num**2+2*self.sys_para.mode_state_num,1:])
             ,label='forbidden')
        
        plt.ylabel('Population')
        plt.ylim(0,1)
        plt.xlabel('Time ('+ self.time_unit+')')
        plt.legend(loc=6)
        
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
            gs = gridspec.GridSpec(2+i1+i2+len(self.sys_para.states_concerned_list), 2)
        else:
            gs = gridspec.GridSpec(3+i1+i2+len(self.sys_para.states_concerned_list), 2)
        
        index = 0
        ## cost
        if self.sys_para.evolve == False and self.sys_para.show_plots == True:
            
            if self.sys_para.state_transfer:
                plt.subplot(gs[index, :],title='Error = %.9f; Runtime: %.1fs; Estimated Remaining Runtime: %.1fh' % (self.last_cost,
                                                                                                   
                                                                                                  self.runtime,
                                                                                                  self.estimated_runtime))
            else:
                plt.subplot(gs[index, :],title='Error = %.9f; Unitary Metric: %.5f; Runtime: %.1fs; Estimated Remaining Runtime: %.1fh' % (self.last_cost,
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
            np.save("./data/GRAPE-costs", np.array(self.costs))
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
        if ( self.sys_para.dts !=[]):
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
        
        for ii in range(len(self.sys_para.states_concerned_list)):
            plt.subplot(gs[index+ii, :],title="Evolution")

            pop_inter_vecs = inter_vecs[ii]
            if self.sys_para.multi:
                self.plot_inter_vecs_general(pop_inter_vecs,self.sys_para.states_concerned_list[ii])
            else:
                self.plot_inter_vecs_general(pop_inter_vecs,self.sys_para.states_concerned_list[ii])        
        
	fig = plt.gcf()
	fig.set_size_inches(15, int (200/4+len(self.sys_para.states_concerned_list)))
	
        display.display(plt.gcf())
        display.clear_output(wait=True)
        if self.sys_para.evolve_error:
            print "Error = %.9f"%self.last_cost
	        

    def __init__(self):
	plt.figure()
