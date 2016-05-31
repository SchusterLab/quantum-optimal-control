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

    def update_convergence(self,last_cost, last_reg_cost, anly):
        self.last_cost = last_cost
        self.last_reg_cost = last_reg_cost
          
        self.anly = anly
        self.plot_summary()
    
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
        plt.xlabel('Time (ns)')
        plt.legend(loc=6)
        
    def plot_inter_vecs_v2(self,pop_inter_vecs):
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(pop_inter_vecs[0,1:])+\
             np.array(pop_inter_vecs[self.sys_para.mode_state_num**2,1:])+\
             np.array(pop_inter_vecs[2*self.sys_para.mode_state_num**2,1:]),label='00')
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(pop_inter_vecs[1,1:])+\
             np.array(pop_inter_vecs[self.sys_para.mode_state_num**2+1,1:])+\
             np.array(pop_inter_vecs[2*self.sys_para.mode_state_num**2+1,1:]),label='01')
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(pop_inter_vecs[self.sys_para.mode_state_num,1:])+\
             np.array(pop_inter_vecs[self.sys_para.mode_state_num**2+self.sys_para.mode_state_num,1:])+\
             np.array(pop_inter_vecs[2*self.sys_para.mode_state_num**2+self.sys_para.mode_state_num,1:]),label='10')
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(pop_inter_vecs[self.sys_para.mode_state_num+1,1:])+\
             np.array(pop_inter_vecs[self.sys_para.mode_state_num**2+self.sys_para.mode_state_num+1,1:])+\
             np.array(pop_inter_vecs[2*self.sys_para.mode_state_num**2+self.sys_para.mode_state_num+1,1:]),label='11')
         
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
        plt.xlabel('Time (ns)')
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
        plt.xlabel('Time (ns)')
        plt.legend(loc=6)
        
    def plot_summary(self):
        
	plt.figure(figsize=(15,50))

        if not self.last_iter == 0:
            self.runtime = time.time() - self.start_time
            self.estimated_runtime = float(self.runtime * (self.max_iterations-self.last_iter) / self.last_iter)/(60*60)
        else:
            self.start_time = time.time()
            self.runtime = 0
            self.estimated_runtime = 0

        
        self.get_convergence()
        
        gs = gridspec.GridSpec(3+len(self.sys_para.states_concerned_list), 2)
        
        ## cost
        plt.subplot(gs[0, :],title='Error = %.9f; Unitary Metric: %.5f; Runtime: %.1fs; Estimated Remaining Runtime: %.1fh' % (self.last_cost,
                                                                                                   self.anly.tf_unitary_scale.eval(),
                                                                                                 
                                                                                                  self.runtime,
                                                                                                  self.estimated_runtime))
        plt.plot(np.array(self.iterations),np.array(self.costs),'bx-',label='loss')
        plt.plot(np.array(self.iterations),np.array(self.reg_costs),'go-',label='reg loss')
        plt.ylabel('Error')
        plt.xlabel('Iteration')
        plt.yscale('log')
        plt.legend()
        np.save("./data/GRAPE-costs", np.array(self.costs))
    
        ## unitary evolution
        M = self.anly.get_final_state()
        plt.subplot(gs[1, 0],title="operator: real")
        plt.imshow(M.real,interpolation='none')
        plt.clim(-1,1)
        plt.colorbar()
        plt.subplot(gs[1, 1],title="operator: imaginary")
        plt.imshow(M.imag,interpolation='none')
        plt.clim(-1,1)
        plt.colorbar()
        
        ## operators
        plt.subplot(gs[2, :],title="Operators")
        ops_weight = self.anly.get_ops_weight()
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),np.array(self.sys_para.ops_max_amp[0]*ops_weight[0]),'r',label='x')
        plt.plot(np.array([self.sys_para.dt* ii for ii in range(self.sys_para.steps)]),(self.sys_para.qm_g1/(2*np.pi))\
             *np.array(self.sys_para.ops_max_amp[1]*ops_weight[1]),'c',label='(g/2pi)z')
        plt.title('Optimized pulse')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (ns)')
        plt.legend()
        
        ## state evolution
        inter_vecs = self.anly.get_inter_vecs()
        
        for ii in range(len(self.sys_para.states_concerned_list)):
            plt.subplot(gs[3+ii, :],title="Evolution")

            pop_inter_vecs = inter_vecs[ii]
            self.plot_inter_vecs_v3(pop_inter_vecs)        
        
        display.display(plt.gcf())
        display.clear_output(wait=True)
        
    def __init__(self):
	pass
