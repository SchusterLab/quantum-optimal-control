import numpy as np
import h5py
import qutip as qt

def qutip_verification(datafile,atol):
    
    
    with h5py.File(datafile,'r') as hf:
    
        gate_time = np.array(hf.get('total_time'))
        gate_steps = np.array(hf.get('steps'))
        H0 = np.array(hf.get('H0'))
        Hops = np.array(hf.get('Hops'))
        initial_vectors_c = np.array(hf.get('initial_vectors_c'))
        uks = np.array(hf.get('uks'))[-1]

        inter_vecs_raw_real = np.array(hf.get('inter_vecs_raw_real'))[-1]
        inter_vecs_raw_imag = np.array(hf.get('inter_vecs_raw_imag'))[-1]

        inter_vecs_raw = inter_vecs_raw_real + 1j*inter_vecs_raw_imag
    
    
    max_abs_diff_list = []
    all_close_list = []
    
    for init_vector_id in range(len(initial_vectors_c)):
        
        psi0 = qt.Qobj(initial_vectors_c[init_vector_id])
        H0_qobj = qt.Qobj(H0)
        Hops_qobj = []

        for Hop in Hops:
            Hops_qobj.append(qt.Qobj(Hop))
            
            
        tlist = np.linspace(0,gate_time,gate_steps+1)
        
        uks_t0 = np.zeros((uks.shape[0],1))

        uks = np.hstack([uks,uks_t0])
        
        dt = gate_time/gate_steps
        
        def make_get_uks_func(id):
            def _function(t,args=None):
                time_id = int(t/dt)
                return uks[id][time_id]
            return _function
        
        Ht_list = []
        Ht_list.append(H0_qobj)

        for ii in range(len(Hops)):
            Ht_list.append([Hops_qobj[ii],make_get_uks_func(ii)])
            
        output = qt.sesolve(Ht_list, psi0, tlist, [])
        
        state_tlist = []
        for state in output.states:
            state_tlist.append(state.full())

        state_tlist = np.array(state_tlist)[:,:,0]
        state_tlist = np.transpose(state_tlist)
        
        abs_diff = np.abs(state_tlist) - np.abs(inter_vecs_raw[init_vector_id])
        
        max_abs_diff_list.append(np.max(abs_diff))
        
        all_close = np.allclose(state_tlist,inter_vecs_raw[init_vector_id],atol=atol)
        
        all_close_list.append(all_close)
    
    print "QuTiP simulation verification result for each initial state"
    print "================================================"
    print "max abs diff: " + str(max_abs_diff_list)
    print "all close: " + str(all_close_list)
    print "================================================"