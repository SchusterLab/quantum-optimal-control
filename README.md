This work has been followed up [here](https://github.com/SchusterLab/rbqoc).


# GRAPE-Tensorflow

This is the code repository of our recent publication "Speedup for quantum optimal control from automatic differentiation based on graphics processing units" https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.042318

This is a software package that performs quantum optimal control using the automatic differentiation capabilities of [Tensorflow](https://www.tensorflow.org/) and has full GPU support. Its main goal is to produce a set of optimal pulses to apply in a given period of time that will drive a quantum system to achieve a certain unitary gate or to reach a certain final quantum state with a fidelity as close as possible to unity. In addition, the user can add any penalties (cost functions) on either the control pulses or the quantum intermediate states and the code will automatically include this constraint in the optimization process without having to write down an analytical form for the gradient of the new cost function.    

As an example of what the package produces, here is its output in the example of a qubit pi pulse:  


![Qubit Pi Pulse Example](http://i.imgur.com/OfqFqZ6.png)

# Setup  
You will just need to setup Tensorflow, Please follow the instructions [here] (https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html)  

Currently only supports linux system and Python 2.7.

# Currently Implemented Cost Functions  
Refer to the [Regularization functions file](https://github.com/SchusterLab/GRAPE-Tensorflow/blob/master/core/RegularizationFunctions.py) for details or to add a new cost function  
 **1) The fidelity cost function:** The overlap between the target unitary/final state and the achieved unitary/final state. In the code, it's referred to as tfs.loss.  
 **2) The gaussian envelope cost function:** A penalty if the control pulses do not have a gaussian envelope. The user supplies a coeffecient called **'envelope'** in the reg_coeffs input. A value of 0.01 is found to be a good statring value empirically.    
 **3) The first derivative cost function:** To make the control pulses smooth. The user supplies a coeffecient called **'dwdt'** in the reg_coeffs input. A value of 0.001 is found to be a good statring value empirically.  
 **4) The second derivative cost function:** To make the control pulses smooth. The user supplies a coeffecient called **'d2wdt2'** in the reg_coeffs input. A value of 0.000001 is found to be a good statring value empirically.  
 **5) The bandpass cost function:** To filter the control pulses frequency **'bandpass'** (start around 0.1) to supress control pulses frequency outside the defined band **'band'**. This cost function requires GPU, since TensorFlow QFT is only implemented in GPU.  
 **6) The forbidden state cost function:** A cost function to forbid the quantum occupation of certain levels through out the time of the control. The user supplies a coeffecinet called **'forbidden'** (start around 100 empirically) and a list called **'states_forbidden_list'** to specify the indices of the levels to forbid.  **forbid_dressed**: A boolean (default is True) to forbid dressed (hamiltonian's eigen vectors) vs bare states in coupled systems  
 **7) The time optimal cost function:** If the user wants to speed up the gate, he should provide a coeffecient called **'speed_up'** (start around 100) to award the occupation of the target state at all intermediate states, hence, making the gate as fast as possible.   


 **To add a new cost function:**  
Just follow the same logic we used and add new code [here](https://github.com/SchusterLab/GRAPE-Tensorflow/blob/master/core/RegularizationFunctions.py) penalizing properties of:  
1) The control fields: held in **tfs.ops_weight**  
and/or  
2) The intermediate states: held in **tfs.inter_vecs**    
 

# Use   
 You should call this function:  
```python
uks, U_final = Grape(H0,Hops,Hnames,U,total_time,steps,states_concerned_list,convergence, U0, 
reg_coeffs,dressed_info, maxA ,use_gpu, draw, initial_guess, show_plots, H_time_scales, 
unitary_error, method,state_transfer, no_scaling, freq_unit, file_name, save, data_path) 
```
 
 You can follow the [examples](https://github.com/SchusterLab/GRAPE-Tensorflow-Examples/tree/master) we are providing for details on defining the quantum system and then calling the function. We suggest starting with a simple example (e.g. spin Pi).
 
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
 **states_concerned_list:** Initial States (list of integers specifying the indices of those states)  
 
# Optional Arguments:  
 **U0:** Initial Unitary (n by n), default is identity  
 **convergence:** A dictionary (can be empty) that might include the following parameters with default values as shown:
                convergence = {'rate':0.01, 'update_step':100, 'max_iterations':5000,
                'conv_target':1e-8,'learning_rate_decay':2500, 'min_grad': 1e-25}   
 **Initial_guess:** A list of k elements, each of them is a steps size array, defining the initial pulses for all operators. If not provided, a default value of a gaussian random distribution will be used.  
 **reg_coeffs:** A dictionary of regularization coeffecients  
 **dressed_info :** A dictionary including the eigenvalues and eigenstates of dressed states  
 **maxA:** a list of the maximum amplitudes of the control pulses (default value is 4)   
 **use_gpu:** a boolean switching gpu and cpu, default is True   
 **sparse_H, sparse_U, sparse_K:** booleans specifying whether (Hamiltonian, Unitary Operator, Unitary Evolution) is sparse. Speedup is expected if the corresponding sparsity is satisfied. (only available in CPU)  
 **use_inter_vecs:** a boolean enable/disable the involvement of state evolution in graph building  
 **draw:** a list including the indices and names for the states to include in drawing state occupation. Ex: states_draw_list = [0,1]
 states_draw_names = ['g00','g01','g10','g11','e00'] and  draw = [states_draw_list,states_draw_names]  
 default value is to draw states with indices 0-3  
 **show_plots:** a boolean (default is True) toggling between progress bar and graphs    
 **state_transfer:** a boolean (default is False) if True, targetting state transfer. If false, targetting unitary evolution. If True, the U is expected to be a vector, not a matrix.    
 **method:** 'ADAM', 'BFGS', 'L-BFGS-B' or 'EVOLVE'. Defining the optimizer. Default is ADAM. EVOLVE only simulate the propagation without optimizing.  
 **Unitary_error:** a float indicating the desired maximum error of the Taylor expansion of the exponential to choose a proper number of expansion terms, default is 1e-4  
 **no_scaling**:  a boolean (default is False)) to disable scaling and squaring  
 **Taylor_terms**: a list [expansion terms, scaling and squaring terms], manually choose the Taylor terms for matrix exponentials.  
 **freq_unit**: a string with default 'GHz'. Can be 'MHz', 'kHz' or 'Hz'  
 **file_name**: file name for saving the simulation  
 **save**: A boolean (default is True) to save the control ops, intermediate vectors, final unitary every update step  
 **data_path**: path for saving the simulation  
 
# More examples:
We applied the optimizer to generate photonic Schrodinger cat states for a circuit quantum electrodynamics system:  
![photonic Schrodinger cat states](http://i.imgur.com/ponY2R9.png)
 

# Questions
If you have any questions, please reach either of the developers of the package: Nelson Leung (nelsonleung@uchicago.edu), Mohamed Abdelhafez (abdelhafez@uchicago.edu) or David Schuster (david.schuster@uchicago.edu)
