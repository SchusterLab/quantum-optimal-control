# GRAPE-Tensorflow
This is the packaged function:  
**Grape (H0, Hops, Hnames, U, U0, total_time, steps,psi0, convergence, reg_coeffs = None, multi_mode = None, maxA = None, use_gpu = True, draw= None, forbidden = None, initial_guess = None, show_plots = True)**

#Mandatory Arguments:
**H0:** Drift Hamiltonian (n by n)   
**Hops:** A list of Control Hamiltonians  (k hamiltonians, each is n by n)  
**Hnames:** A list of Control Hamiltonian names, with k string elements  
**U:** Target Unitary (n by n)  
**U0:** Initial Unitary (n by n)  
**total_time:** Total Time (float)  
**Steps:** Number of time steps (int)  
**psi0:** Initial States (list of integers specifying the indices of those states)  
**convergence:** A dictionary (can be empty) that might include the following parameters with default values as shown:
               convergence = {'rate':0.01, 'update_step':100, 'max_iterations':5000,
               'conv_target':1e-8,'learning_rate_decay':2500}   

#Optional Arguments:  
**Initial_guess:** A list of k elements, each of them is a steps size array, defining the initial pulses for all operators. If not provided, a default value of a gaussian random distribution will be used.  
**reg_coeffs:** A dictionary of regulaization coeffecients with default values: reg_coeffs = {'alpha' : 0.01, 'z':0.01, 'dwdt':0.01,'d2wdt2':0.001*0.0001, 'inter':100}   
where alpha: imposes a Gaussian envelope    
z: limits z dc value  
dwdt: limits the first derivative of the pulses  
d2wdt2: limits the second derivative of the pulses  
inter: limits forbidden states occupation  
  
**multi_mode  :** A dictionary including the details of our specific system (a qubit with multimodes) to treat it differently (having more options for it like modulation and dressed state treatment)  
Ex: multi_mode = {'dressed':dressed, 'vectors':v_c, 'qnum':qubit_state_num, 'mnum': mode_state_num,\
              'f':freq_ge, 'es':w_c, 'g1':qm_g1, 'D':D, 'Interpolation':True, 'Modulation':True}  
  

**maxA:** a list of the maximum amplitudes of the control pulses (default value is 4)  
  
**use_gpu:** a boolean switching gpu and cpu, default is True  
  
**draw:** a list including the indices and names for the states to include in drawing state occupation. Ex: states_draw_list = [0,1,mode_state_num,mode_state_num+1,mode_state_num**2]
states_draw_names = ['g00','g01','g10','g11','e00'] and  draw = [states_draw_list,states_draw_names]  
default value is to draw states with indices 0-3  

**forbidden:** a list of integer indices indicating the states to penalize  

**show_plots:** a boolean (default is True) toggling between progress bar and graphs



