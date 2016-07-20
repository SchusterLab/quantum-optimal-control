import numpy as np
from helper_functions.grape_functions import c_to_r_mat
from helper_functions.grape_functions import c_to_r_vec
from helper_functions.grape_functions import get_state_index
import scipy.linalg as la
from scipy.special import factorial

class SystemParametersGeneral:

    def __init__(self,H0,Hops,Hnames,U,U0,total_time,steps,states_forbidden_list,states_concerned_list,multi_mode,maxA, draw,initial_guess,evolve, evolve_error, show_plots, H_time_scales,Unitary_error,state_transfer,no_scaling,limit_dc):
        # Input variable
        self.state_transfer = state_transfer
        self.no_scaling = no_scaling
        self.H0_c = H0
        self.ops_c = Hops
        self.ops_max_amp = maxA
        self.Hnames = Hnames
        self.Hnames_original = Hnames #because we might rearrange them later if we have different timescales 
        self.multi = False #are we using a multimode system?
        self.total_time = total_time
        self.steps = steps
        self.show_plots = show_plots
        self.Unitary_error= Unitary_error
        if states_forbidden_list!= None:
            self.states_forbidden_list = states_forbidden_list
        else:
            self.states_forbidden_list =[]
        
        if limit_dc == None:
            self.limit_dc = []
        else:
            self.limit_dc = limit_dc
        if initial_guess!= None:
            self.u0 = initial_guess
            for ii in range (len(self.u0)):
                self.u0[ii]= self.u0[ii]/self.ops_max_amp[ii]
            self.u0 = np.arctanh(self.u0) #because we take the tanh of weights later
            
        else:
            self.u0 =[]
        self.states_concerned_list = states_concerned_list
        if H_time_scales!= None:
            self.dts = H_time_scales
        else:
            self.dts =[]
        self.Modulation = False
        self.Interpolation = False
        self.D = False
        self.U0_c = U0
        self.initial_unitary = c_to_r_mat(U0) #CtoRMat is converting complex matrices to their equivalent real (double the size) matrices
        if self.state_transfer == False:
            self.target_unitary = c_to_r_mat(U)
        else:
            self.target_vector = c_to_r_vec(U)
        
        if draw != None:
            self.draw_list = draw[0]
            self.draw_names = draw[1]
        else:
            self.draw_list = []
            self.draw_names = []
        if multi_mode !=None:
            self.multi = True
            self.v_c = multi_mode['vectors']
            self.dressed = multi_mode['dressed']
            self.mode_state_num = multi_mode['mnum']
            self.qubit_state_num = multi_mode['qnum']
            self.freq_ge= multi_mode['f']
            self.w_c = multi_mode['es']
            self.qm_g1 = multi_mode['g1']
            self.D = multi_mode['D']
            self.Interpolation = multi_mode['Interpolation']
            self.Modulation = multi_mode['Modulation']
            self.H0_diag=np.diag(self.w_c)
        self.evolve = evolve
        self.evolve_error = evolve_error
        self.init_system()
        self.init_vectors()
        self.init_operators()
        self.init_one_minus_gaussian_envelope()
        self.init_pulse_operator()

    def approx_expm(self,M,exp_t, scaling_terms): #approximate the exp at the beginning to estimate the number of taylor terms and scaling and squaring needed
        U=np.identity(len(M),dtype=M.dtype)
        Mt=np.identity(len(M),dtype=M.dtype)
        factorial=1.0 #for factorials
        
        for ii in xrange(1,exp_t):
            factorial*=ii
            Mt=np.dot(Mt,M)
            U+=Mt/((2.**float(ii*scaling_terms))*factorial) #scaling by 2**scaling_terms

        
        for ii in xrange(scaling_terms):
            U=np.dot(U,U) #squaring scaling times
        
        return U
    
    def Choose_exp_terms(self, d): #given our hamiltonians and a number of scaling/squaring, we determine the number of Taylor terms
        exp_t = 20 #maximum
        
        H=self.H0_c
        U_f = self.U0_c
        for ii in range (len(self.ops_c)):
            H = H + self.ops_max_amp[ii]*self.ops_c[ii]
        if d == 0:
            self.scaling = max(int(2*np.log2(np.max(np.abs(-(0+1j) * self.dt*H)))),0) 
         
        else:
            self.scaling += d
        
        if self.state_transfer or self.no_scaling:
            self.scaling =0
        while True:
            
            for ii in range (self.steps):
                U_f = np.dot(U_f,self.approx_expm((0-1j)*self.dt*H, exp_t, self.scaling))
            Metric = np.abs(np.trace(np.dot(np.conjugate(np.transpose(U_f)), U_f)))/(self.state_num)
            
            if exp_t == 3:
                break
            if np.abs(Metric - 1.0) < self.Unitary_error:
                exp_t = exp_t-1
            else:
                break
        
        return exp_t



        
    def init_system(self):
        self.dt = self.total_time/self.steps
        
        self.Dts = []
        self.Dts_indices = []
        self.ctrl_steps = []
        idx = []
        if self.dts != []: #generate Hops and Hnames rearranged such that the controls without a special dt come first
            for key in self.dts:
                Dt= self.dts[key]
                if Dt > self.dt:
                    self.Dts.append(Dt)
                    self.Dts_indices.append(int(key))
                    self.ctrl_steps.append(int(self.total_time/Dt)+1)

            for ii in range (len(self.ops_c)):
                if ii not in self.Dts_indices:
                    idx.append(ii)
            for jj in range (len(self.Dts_indices)):
                idx.append(self.Dts_indices[jj])

            
            self.new_limit_dc = []
            for ii in self.limit_dc:
                self.new_limit_dc.append( idx.index(ii))
            self.limit_dc = self.new_limit_dc
            self.ops_c = np.asarray(self.ops_c)[idx]
            self.ops_max_amp = np.asarray(self.ops_max_amp)[idx]
            self.Hnames = np.asarray(self.Hnames)[idx]
            self.Dt = self.dt
            self.control_steps = self.steps
        
            print self.ctrl_steps
        
        self.state_num= len(self.H0_c)
        
        
    def init_vectors(self):
        self.initial_vectors=[]

        for state in self.states_concerned_list:
            if self.D:
                self.initial_vector_c= self.v_c[:,get_state_index(state,self.dressed)]
            else:
                self.initial_vector_c=np.zeros(self.state_num)
                self.initial_vector_c[state]=1
            self.initial_vector = c_to_r_vec(self.initial_vector_c)

            self.initial_vectors.append(self.initial_vector)


    def init_operators(self):
        # Create operator matrix in numpy array


        
        self.ops=[]
        for op_c in self.ops_c:
            op = c_to_r_mat(-1j*self.dt*op_c)
            self.ops.append(op)
        #y_op = CtoRMat(YI)
        
        self.ops_len = len(self.ops)
        


        self.H0 = c_to_r_mat(-1j*self.dt*self.H0_c)
        
        
            
        self.identity_c = np.identity(self.state_num)
        
        
        self.identity = c_to_r_mat(self.identity_c)
        
        self.exps =[]
        self.scalings = []
        if self.state_transfer or self.no_scaling:
            comparisons = 1
        else:
            comparisons = 5
        d = 0
        while comparisons >0:
            
            self.exp_terms = self.Choose_exp_terms(d)
            self.exps.append(self.exp_terms)
            self.scalings.append(self.scaling)
            comparisons = comparisons -1
            d = d+1
        self.complexities = np.add(self.exps,self.scalings)
        a = np.argmin(self.complexities)
        
        self.exp_terms = self.exps[a]
        self.scaling = self.scalings[a]
        print "Using "+ str(self.exp_terms) + " Taylor terms and "+ str(self.scaling)+" Scaling & Squaring terms"
        
        
    def init_one_minus_gaussian_envelope(self):
        # Generating the Gaussian envelope that pulses should obey
        one_minus_gauss = []
        offset = 0.0
        overall_offset = 0.01
        opsnum=self.ops_len
        for ii in range(opsnum):
            constraint_shape = np.ones(self.steps)- self.gaussian(np.linspace(-2,2,self.steps)) - offset
            constraint_shape = constraint_shape * (constraint_shape>0)
            constraint_shape = constraint_shape + overall_offset* np.ones(self.steps)
            one_minus_gauss.append(constraint_shape)


        self.one_minus_gauss = np.array(one_minus_gauss)


    def gaussian(self,x, mu = 0. , sig = 1. ):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def init_pulse_operator(self):

        #functions
        def sin(t, a, f):
            return a*np.sin(2*np.pi*f*t)

        def cos(t, a, f):
            return a*np.cos(2*np.pi*f*t)

        # gaussian envelope
        gaussian_envelope = self.gaussian(np.linspace(-2,2,self.steps))

        # This is to generate a manual pulse
        manual_pulse = []

        a=0.00

        #manual_pulse.append(gaussian_envelope * cos(np.linspace(0,self.total_time,self.steps),a,self.freq_ge))
        #manual_pulse.append(gaussian_envelope * sin(np.linspace(0,self.total_time,self.steps),a,self.freq_ge))
        manual_pulse.append(np.zeros(self.steps))


        self.manual_pulse = np.array(manual_pulse)
