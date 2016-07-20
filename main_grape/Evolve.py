
from main_grape.Grape import Grape

def Evolve(H0,Hops,total_time,steps,psi0,initial_guess,U0 = None, U=None,draw=None,Hnames = None):
    
    
    if U0 == None:
        U0 = np.identity(len(H0))
    flag = True
    if Hnames == None:
        for ii in range (len(Hops)):
            Hnames.append(str(ii))
    
    if U == None:
        flag = False
        U = U0
    convergence = {'rate':0, 'update_step':1, 'max_iterations':0,\
               'conv_target':1e-8,'learning_rate_decay':1}
    
    Grape(H0,Hops,Hnames,U,total_time,steps,psi0,convergence=convergence, draw= draw, initial_guess = initial_guess, U0 = U0, evolve_only = True, evolve_error = flag,Unitary_error = 1e-20)
