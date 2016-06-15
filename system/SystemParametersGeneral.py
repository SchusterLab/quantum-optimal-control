import numpy as np
from math_functions.c_to_r_mat import CtoRMat
import scipy.linalg as la
from math_functions.Get_state_index import Get_State_index

class SystemParametersGeneral:

    def __init__(self,H0,Hops,U,U0,total_time,steps,states_forbidden_list,states_concerned_list,D,Modulation,Interpolation,multi_mode,maxA):
        # Input variable
        
        self.H0_c = H0
        self.ops_c = Hops
        self.ops_max_amp = maxA
        
        self.total_time = total_time
        self.steps = steps+1
        self.states_forbidden_list = states_forbidden_list
        self.states_concerned_list = states_concerned_list
        self.Modulation = Modulation
        self.Interpolation = Interpolation
        self.D = D
        self.initial_state = CtoRMat(U0)
        self.target_state = CtoRMat(U)
        self.v_c = multi_mode['vectors']
        self.dressed = multi_mode['dressed']
        self.mode_state_num = multi_mode['mnum']
        self.qubit_state_num = multi_mode['qnum']
        self.freq_ge= multi_mode['f']
        self.w_c = multi_mode['es']
        self.qm_g1 = multi_mode['g1']
        
        self.init_system()
        self.init_vectors()
        self.init_operators()
        self.init_one_minus_gaussian_envelop()
        self.init_pulse_operator()
        self.prev_ops_weight()

    def init_system(self):
        self.initial_pulse = False
        self.prev_pulse = False

        self.exp_terms = 20
        self.subpixels = 50
        

        self.dt = self.total_time/self.steps
        if self.Interpolation:
            self.Dt = self.dt*self.subpixels
        else:
            self.Dt = self.dt
        
        self.control_steps = int(self.total_time/self.Dt)+1
        self.state_num= len(self.H0_c)
        
        
    def init_vectors(self):
        self.initial_vectors=[]

        for state in self.states_concerned_list:
            if self.D:
                self.initial_vector_c= self.v_c[:,Get_State_index(state,self.dressed)]
            else:
                self.initial_vector_c=np.zeros(self.state_num)
                self.initial_vector_c[state]=1
            self.initial_vector = np.append(self.initial_vector_c,self.initial_vector_c)

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
        
        
        self.H0_diag=np.diag(self.w_c)
        self.identity_c = np.identity(self.state_num)
        
        self.identity = CtoRMat(self.identity_c)
        
        
    def init_one_minus_gaussian_envelop(self):
        # This is used for weighting the weight so the final pulse can have more or less gaussian like
        one_minus_gauss = []
        offset = 0.0
        overall_offset = 0.01
        for ii in range(self.ops_len+1):
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

        manual_pulse.append(gaussian_envelop * cos(np.linspace(0,self.total_time,self.steps),a,self.freq_ge))
        manual_pulse.append(gaussian_envelop * sin(np.linspace(0,self.total_time,self.steps),a,self.freq_ge))
        manual_pulse.append(np.zeros(self.steps))


        self.manual_pulse = np.array(manual_pulse)

    def prev_ops_weight(self):
        if self.initial_pulse and self.prev_pulse:
            prev_ops_weight = np.load("/home/nelson/Simulations/GRAPE-GPU/data/g00-g11/GRAPE-control.npy")
            prev_ops_weight_base = np.arctanh(prev_ops_weight)
            temp_ops_weight_base = np.zeros([self.ops_len,self.steps])
            temp_ops_weight_base[:,:len(prev_ops_weight_base[0])] +=prev_ops_weight_base
            self.prev_ops_weight_base = temp_ops_weight_base
            

