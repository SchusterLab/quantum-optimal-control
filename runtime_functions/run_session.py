import numpy as np
import tensorflow as tf
from runtime_functions.Analysis import Analysis
import os
import time

def run_session(tfs,graph,conv,sys_para,show_plots=True,single_simulation = False):
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        
        
        print "Initialized"
        iterations = 0
        start_time = time.time()
        while True:
            if (single_simulation == False):
                max_iterations = conv.max_iterations
            else:
                max_iterations = 0
            learning_rate = float(conv.rate) * np.exp(-float(iterations)/conv.learning_rate_decay)
            
            feed_dict = {tfs.learning_rate : learning_rate, tfs.z_reg_alpha_coeff: conv.z_reg_alpha_coeff,
                        tfs.reg_alpha_coeff: conv.reg_alpha_coeff, 
                         tfs.dwdt_reg_alpha_coeff: conv.dwdt_reg_alpha_coeff,
                         tfs.d2wdt2_reg_alpha_coeff: conv.d2wdt2_reg_alpha_coeff,
                         tfs.inter_reg_alpha_coeff:conv.inter_reg_alpha_coeff}
            _, l,rl = session.run([tfs.optimizer, tfs.loss, tfs.reg_loss], feed_dict=feed_dict)
            
                
            if (iterations % conv.update_step == 0) or (l < conv.conv_target):    
                if sys_para.show_plots:
                # Plot convergence
                    if sys_para.multi:
                        anly = Analysis(sys_para,tfs.final_state,tfs.ops_weight,tfs.xy_weight, tfs.xy_nocos, tfs.unitary_scale,tfs.inter_vecs)
                    else:
                        anly = Analysis(sys_para,tfs.final_state,tfs.ops_weight,tfs.ops_weight, tfs.ops_weight, tfs.unitary_scale,tfs.inter_vecs)
                    conv.update_convergence(l,rl,anly,show_plots)
                
                # Save the variables to disk.
                    this_dir = os.path.dirname(__file__)
                    tmp_path = os.path.join(this_dir,'../tmp/grape.ckpt')
                    save_path = tfs.saver.save(session, tmp_path)
                    if (iterations >= max_iterations) or (l < conv.conv_target): #(l<conv.conv_target) or (iterations>=conv.max_iterations):
                        anly.get_ops_weight()
                    #anly.get_xy_weight()
                    #if sys_para.Modulation:
                        #anly.get_nonmodulated_weight() 
                        break
                else:
                    elapsed = time.time() - start_time
                    print 'Error = %.9f; Runtime: %.1fs; Iterations = %d'%(l,elapsed,iterations)
                    if (iterations >= max_iterations) or (l < conv.conv_target): #(l<conv.conv_target) or (iterations>=conv.max_iterations):
                        if sys_para.multi:
                            anly = Analysis(sys_para,tfs.final_state,tfs.ops_weight,tfs.xy_weight, tfs.xy_nocos, tfs.unitary_scale,tfs.inter_vecs)
                        else:
                            anly = Analysis(sys_para,tfs.final_state,tfs.ops_weight,tfs.ops_weight, tfs.ops_weight, tfs.unitary_scale,tfs.inter_vecs)
                        conv.update_convergence(l,rl,anly,show_plots)
                        print 'Error = %.9f; Runtime: %.1fs; Iterations = %d'%(l,elapsed,iterations)
                        anly.get_ops_weight()
                    #anly.get_xy_weight()
                    #if sys_para.Modulation:
                        #anly.get_nonmodulated_weight() 
                        break
                    
                
            iterations+=1
