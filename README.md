# GRAPE-Tensorflow

 This is the packaged function:  
 **uks, U_final = Grape (H0, Hops, Hnames, U, total_time, steps,psi0, convergence = None, U0 = None, reg_coeffs = None, multi_mode = None, maxA = None, use_gpu = True, draw= None, forbidden = None, initial_guess = None, show_plots = True, H_time_scales = None, Unitary_error = 1e-4, state_transfer = False, method = 'Adam', switch = True, no_scaling = False, freq_unit= 'GHz', limit_dc = None)**
 
# Returns:  
 **uks:** The optimized control pulses  ( a list of list of floats, each of them has length  = ctrl_steps(ctrl_op) ) same order as the input  
 **U_final:** The final Unitary (n by n)  
 
# Mandatory Arguments:  
 **H0:** Drift Hamiltonian (n by n)   
 **Hops:** A list of Control Hamiltonians  (k hamiltonians, each is n by n)  
 **Hnames:** A list of Control Hamiltonian names, with k string elements  
 **U:** Target Unitary (n by n)  if state_transfer = False. a vector (n by 1) if state_transfer = True  
 **total_time:** Total Time (float)  
 **Steps:** Number of time steps (int)  
 **psi0:** Initial States (list of integers specifying the indices of those states)  
 
 -#Optional Arguments:  
 **U0:** Initial Unitary (n by n), default is identity  
 **convergence:** A dictionary (can be empty) that might include the following parameters with default values as shown:
                convergence = {'rate':0.01, 'update_step':100, 'max_iterations':5000,
                'conv_target':1e-8,'learning_rate_decay':2500, 'min_grad': 1e-25}   
 
 **Initial_guess:** A list of k elements, each of them is a steps size array, defining the initial pulses for all operators. If not provided, a default value of a gaussian random distribution will be used.  
 **reg_coeffs:** A dictionary of regulaization coeffecients with default values: reg_coeffs = {'envelope' : 0.01, 'dc':0.01, 'dwdt':0.01,'d2wdt2':0.001*0.0001, 'forbidden':100}   
 where envelope: imposes a Gaussian envelope    
 dc: limits dc value  
 dwdt: limits the first derivative of the pulses  
 d2wdt2: limits the second derivative of the pulses  
 forbidden: limits forbidden states occupation  
   
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
 
 **H_time_scales:** a dictionary whose keys are the indices of the control ops to be interpolated, and the values are the dt to use for each key.    
 
 **Unitary_error:** a float indicating the desired maximum error of the Taylor expansion of the exponential to choose a proper number of expansion terms, default is 1e-4  
 
 **state_transfer:** a boolean (default is False) if True, targetting state transfer. If false, targetting unitary evolution. If True, the U is expected to be a vector, not a matrix.    
 **method:** 'Adam', 'BFGS'   or 'L-BFGS-B'. Default is Adam  
 **switch:** a boolean (default is True) to switch from BFGS/L-BFGS-B to Adam if a precision loss happens  
 **no_scaling**:  a boolean (default is False)) to stop scaling and squaring  
 **freq_unit**: a string with default 'GHz'. Can be 'MHz', 'kHz' or 'Hz'  
 **limit_dc**: a list of control indices that we want to penalize their dc offset  
 
 
