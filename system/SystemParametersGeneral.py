import numpy as np

class SystemParametersGeneral:

    def __init__(self,total_time):
        # Input variable
        self.total_time = total_time
        
        self.init_system()
        self.init_operators()
        self.init_one_minus_gaussian_envelop()
        self.init_pulse_operator()
        self.prev_ops_weight()
        self.Bessel_func()

    def init_system(self):
        self.initial_pulse = False
        self.prev_pulse = False

        self.qubit_state_num = 4
        self.alpha = 0.224574
        self.freq_ge = 3.9225#GHz
        self.ens = np.array([ 2*np.pi*ii*(self.freq_ge - 0.5*(ii-1)*self.alpha) for ii in arange(self.qubit_state_num)])

        self.mode_state_num = 3

        self.qm_g1 = 2*np.pi*0.1 #GHz
        self.mode_freq1 = 6.0 #GHz
        self.mode_ens1 = np.array([ 2*np.pi*ii*(self.mode_freq1) for ii in np.arange(self.mode_state_num)])


        self.qm_g2 = 2*np.pi*0.1 #GHz
        self.mode_freq2 = 6.5 #GHz
        self.mode_ens2 = np.array([ 2*np.pi*ii*(self.mode_freq2) for ii in np.arange(self.mode_state_num)])

        self.state_num = self.qubit_state_num * (self.mode_state_num**2)

        
        states_h = range(3*self.mode_state_num**2,4*self.mode_state_num**2)
        states_gef02 = [2,self.mode_state_num**2+2,2*self.mode_state_num**2+2]
        states_gef20 = [2*self.mode_state_num,self.mode_state_num**2+2*self.mode_state_num,2*self.mode_state_num**2+2*self.mode_state_num]

        self.states_forbidden_list = states_h + states_gef02 + states_gef20


        self.pts_per_period = 10
        self.exp_terms = 20

        self.dt = (1./self.mode_freq2)/self.pts_per_period
        self.steps = int(self.total_time/self.dt)
        
        
    def init_vectors(self):
        self.initial_vectors=[]

        for state in self.states_concerned_list:
            self.initial_vector_c=np.zeros(self.state_num)
            self.initial_vector_c[state]=1
            self.initial_vector = np.append(self.initial_vector_c,self.initial_vector_c)

            self.initial_vectors.append(self.initial_vector)


    def init_operators(self):
        # Create operator matrix in numpy array

        H_q = np.diag(self.ens)
        H_m1 = np.diag(self.mode_ens1)
        H_m2 = np.diag(self.mode_ens2)

        Q_x   = np.diag(np.sqrt(np.arange(1,self.qubit_state_num)),1)+np.diag(np.sqrt(np.arange(1,self.qubit_state_num)),-1)
        #Q_y   = (0+1j) *(np.diag(np.sqrt(np.arange(1,self.qubit_state_num)),1)-np.diag(np.sqrt(np.arange(1,self.qubit_state_num)),-1))
        Q_z   = np.diag(np.arange(0,self.qubit_state_num))

        M_x = np.diag(np.sqrt(np.arange(1,self.mode_state_num)),1)+np.diag(np.sqrt(np.arange(1,self.mode_state_num)),-1)

        self.I_q = np.identity(self.qubit_state_num)
        self.I_m = np.identity(self.mode_state_num)

        XI = np.kron(Q_x,np.kron(self.I_m,self.I_m))
        #YI = np.kron(Q_y,np.kron(self.I_m,self.I_m))
        ZI = np.kron(Q_z,np.kron(self.I_m,self.I_m))

        self.ops = [XI,ZI]

        x_op = CtoRMat(-1j*self.dt*XI)
        #y_op = CtoRMat(YI)
        z_op = CtoRMat(-1j*self.dt*ZI)

        self.ops = [x_op,z_op]
        self.ops_max_amp = [4.0,2*np.pi*2.0]

        self.ops_len = len(self.ops)

        self.H0_c = np.kron(H_q,np.kron(self.I_m,self.I_m)) + np.kron(self.I_q,np.kron(H_m1,self.I_m)) +\
        np.kron(self.I_q,np.kron(self.I_m,H_m2)) + self.qm_g1*np.kron(Q_x,np.kron(M_x,self.I_m)) +\
        self.qm_g2*np.kron(Q_x,np.kron(self.I_m,M_x))

        self.H0 = CtoRMat(-1j*self.dt*self.H0_c)
        
        self.q_identity_c = np.identity(self.qubit_state_num)
        self.m_identity_c = np.identity(self.mode_state_num)

        self.identity_c = np.kron(self.q_identity_c,np.kron(self.m_identity_c,self.m_identity_c))
        self.identity = CtoRMat(self.identity_c)
        
    def init_one_minus_gaussian_envelop(self):
        # This is used for weighting the weight so the final pulse can have more or less gaussian like
        one_minus_gauss = []
        offset = 0.0
        overall_offset = 0.01
        for ii in range(self.ops_len):
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
            
    def Bessel_func(self):
        self.J0_i = sp.special.jn(0,0+1j)
        self.Jk_neg_i = []

        for ii in range (0,self.exp_terms):
            self.Jk_neg_i.append(sp.special.jn(ii,0-1j))
