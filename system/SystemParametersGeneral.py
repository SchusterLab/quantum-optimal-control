import numpy as np
from math_functions.c_to_r_mat import CtoRMat
from math_functions.c_to_r_mat import CtoRVec
import scipy.linalg as la
from math_functions.Get_state_index import Get_State_index
from scipy.special import factorial

class SystemParametersGeneral:

    def __init__(self,H0,Hops,Hnames,U,U0,total_time,steps,states_forbidden_list,states_concerned_list,multi_mode,maxA, draw,initial_guess,evolve, evolve_error, show_plots, H_time_scales,Unitary_error,opti_traj=False):
        # Input variable
        self.opti_traj=opti_traj
        self.H0_c = H0
        self.ops_c = Hops
        self.ops_max_amp = maxA
        self.Hnames = Hnames
        self.Hnames_original = Hnames
        self.multi = False
        self.total_time = total_time
        self.steps = steps
        self.show_plots = show_plots
        self.Unitary_error= Unitary_error
        if states_forbidden_list!= None:
            self.states_forbidden_list = states_forbidden_list
        else:
            self.states_forbidden_list =[]
        
        if initial_guess!= None:
            self.u0 = initial_guess
            for ii in range (len(self.u0)):
                self.u0[ii]= self.u0[ii]/self.ops_max_amp[ii]
            self.u0 = np.arctanh(self.u0)
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
        self.initial_state = CtoRMat(U0)
        self.target_state = CtoRMat(U)
        
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
        self.init_one_minus_gaussian_envelop()
        self.init_pulse_operator()
        self.prev_ops_weight()

    def approx_expm(self,M,exp_t, div):
        U=np.identity(len(M),dtype=M.dtype)
        Mt=np.identity(len(M),dtype=M.dtype)
        iif=1.0
        
        for ii in xrange(1,exp_t):
            iif*=ii
            #print "ii = %d, iif = %f, 2**(ii*div) = %f" % (ii,iif, 2**(ii*div))
            #print Mt
            Mt=np.dot(Mt,M)
            U+=Mt/((2.**float(ii*div))*iif)

        
        for ii in xrange(div):
            U=np.dot(U,U)
        
        return U
    def Choose_exp_terms(self, d):
        exp_t = 20
        
        H=self.H0_c
        U_f = self.U0_c
        for ii in range (len(self.ops_c)):
            H = H + self.ops_max_amp[ii]*self.ops_c[ii]
        if d == 0:
            self.div = max(int(2*np.log2(np.max(np.abs(-(0+1j) * self.dt*H)))),0) 
         
        else:
            self.div = d
            
        while True:
            for ii in range (self.steps):
                U_f = np.dot(U_f,self.approx_expm((0-1j)*self.dt*H, exp_t, self.div))
            Metric = np.abs(np.trace(np.dot(np.conjugate(np.transpose(U_f)), U_f)))/(self.state_num)
            if exp_t == 3:
                break
            if np.abs(Metric - 1.0) < self.Unitary_error:
                exp_t = exp_t-1
            else:
                break
        
        return exp_t



        
    def init_system(self):
        self.initial_pulse = False
        self.prev_pulse = False

        
        self.subpixels = 50
        

        self.dt = self.total_time/self.steps
        if self.Interpolation:
            self.Dt = self.dt*self.subpixels
            self.control_steps = int(self.total_time/self.Dt)+1
        else:
            self.Dts = []
            self.Dts_indices = []
            self.ctrl_steps = []
            idx = []
            if self.dts != []:
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
                    
                
                self.ops_c = np.asarray(self.ops_c)[idx]
                self.ops_max_amp = np.asarray(self.ops_max_amp)[idx]
                self.Hnames = np.asarray(self.Hnames)[idx]
            self.Dt = self.dt
            self.control_steps = self.steps
        
        
        self.state_num= len(self.H0_c)
        
        
    def init_vectors(self):
        self.initial_vectors=[]

        for state in self.states_concerned_list:
            if self.D:
                self.initial_vector_c= self.v_c[:,Get_State_index(state,self.dressed)]
            else:
                self.initial_vector_c=np.zeros(self.state_num)
                self.initial_vector_c[state]=1
            self.initial_vector = CtoRVec(self.initial_vector_c)

            self.initial_vectors.append(self.initial_vector)


    def init_operators(self):
        # Create operator matrix in numpy array


        
        self.ops=[]
        for op_c in self.ops_c:
            op = CtoRMat(-1j*self.dt*op_c)
            self.ops.append(op)
        #y_op = CtoRMat(YI)
        
        self.ops_len = len(self.ops)
        


        self.H0 = CtoRMat(-1j*self.dt*self.H0_c)
        
        
            
        self.identity_c = np.identity(self.state_num)
        
        
        self.identity = CtoRMat(self.identity_c)
        
        self.exps =[]
        self.divs = []
        comparisons = 5
        d = 0
        while comparisons >0:
            self.exp_terms = self.Choose_exp_terms(d)
            self.exps.append(self.exp_terms)
            self.divs.append(self.div)
            comparisons = comparisons -1
            d = d+1
        self.complexities = np.add(self.exps,self.divs)
        a = np.argmin(self.complexities)
        
        self.exp_terms = self.exps[a]
        self.div = self.divs[a]
        print "Using "+ str(self.exp_terms) + " Taylor terms and "+ str(self.div)+" Scaling & Squaring terms"
        
        
    def init_one_minus_gaussian_envelop(self):
        # This is used for weighting the weight so the final pulse can have more or less gaussian like
        one_minus_gauss = []
        offset = 0.0
        overall_offset = 0.01
        if self.multi:
            opsnum=self.ops_len+1
        else:
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

        # gaussian envelop
        gaussian_envelop = self.gaussian(np.linspace(-2,2,self.steps))

        # This is to generate a manual pulse
        manual_pulse = []

        a=0.00

        #manual_pulse.append(gaussian_envelop * cos(np.linspace(0,self.total_time,self.steps),a,self.freq_ge))
        #manual_pulse.append(gaussian_envelop * sin(np.linspace(0,self.total_time,self.steps),a,self.freq_ge))
        manual_pulse.append(np.zeros(self.steps))


        self.manual_pulse = np.array(manual_pulse)

    def prev_ops_weight(self):
        if self.initial_pulse and self.prev_pulse:
            prev_ops_weight = np.load("/home/nelson/Simulations/GRAPE-GPU/data/g00-g11/GRAPE-control.npy")
            prev_ops_weight_base = np.arctanh(prev_ops_weight)
            temp_ops_weight_base = np.zeros([self.ops_len,self.steps])
            temp_ops_weight_base[:,:len(prev_ops_weight_base[0])] +=prev_ops_weight_base
            self.prev_ops_weight_base = temp_ops_weight_base
            

