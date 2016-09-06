# GRAPE-Tensorflow

 This is the packaged function:  
 **uks, U_final = Grape(H0,Hops,Hnames,U,total_time,steps,states_concerned_list,convergence, U0, reg_coeffs,dressed_info, maxA ,use_gpu, draw, initial_guess, evolve_only,show_plots, H_time_scales, unitary_error, method,state_transfer, switch,no_scaling, freq_unit, file_name, save, data_path)**
 
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
 
# Optional Arguments:  
 **U0:** Initial Unitary (n by n), default is identity  
 **convergence:** A dictionary (can be empty) that might include the following parameters with default values as shown:
                convergence = {'rate':0.01, 'update_step':100, 'max_iterations':5000,
                'conv_target':1e-8,'learning_rate_decay':2500, 'min_grad': 1e-25}   
 
 **Initial_guess:** A list of k elements, each of them is a steps size array, defining the initial pulses for all operators. If not provided, a default value of a gaussian random distribution will be used.  
 **reg_coeffs:** A dictionary of regulaization coeffecients

 **dressed_info :** A dictionary including the eigenvalues and eigenstates of dressed states
 
 **maxA:** a list of the maximum amplitudes of the control pulses (default value is 4)  
   
 **use_gpu:** a boolean switching gpu and cpu, default is True  
   
 **draw:** a list including the indices and names for the states to include in drawing state occupation. Ex: states_draw_list = [0,1,mode_state_num,mode_state_num+1,mode_state_num**2]
 states_draw_names = ['g00','g01','g10','g11','e00'] and  draw = [states_draw_list,states_draw_names]  
 default value is to draw states with indices 0-3  
 
 **show_plots:** a boolean (default is True) toggling between progress bar and graphs  
 
 **H_time_scales:** a dictionary whose keys are the indices of the control ops to be interpolated, and the values are the dt to use for each key.    
 
 **Unitary_error:** a float indicating the desired maximum error of the Taylor expansion of the exponential to choose a proper number of expansion terms, default is 1e-4  
 
 **state_transfer:** a boolean (default is False) if True, targetting state transfer. If false, targetting unitary evolution. If True, the U is expected to be a vector, not a matrix.    
 **method:** 'Adam', 'BFGS'   or 'L-BFGS-B'. Default is L-BFGS-B  
 **switch:** a boolean (default is True) to switch from BFGS/L-BFGS-B to Adam if a precision loss happens  
 **no_scaling**:  a boolean (default is False)) to stop scaling and squaring  
 **freq_unit**: a string with default 'GHz'. Can be 'MHz', 'kHz' or 'Hz'  

 **forbid_dressed**: A boolean (default is True) to forbid dressed (hamiltonian's eigen vectors) vs bare states in coupled systems 
 
 **file_name**: file name for saving the simulation  
 **save**: A boolean (default is True) to save the control ops, intermediate vectors, final unitary every update step  
 **data_path**: path for saving the simulation  
 
 
 
