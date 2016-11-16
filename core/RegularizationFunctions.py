import tensorflow as tf
import numpy as np
import math

from helper_functions.grape_functions import c_to_r_mat, sort_ev

def get_reg_loss(tfs):
    
    # Regulizer
    with tf.name_scope('reg_errors'):
        
        tfs.reg_loss = tfs.loss
        
        # envelope: to make it close to a Gaussian envelope
        # dc: to limit DC offset of z pulses 
        # dwdt: to limit pulse first derivative
        # d2wdt2: to limit second derivatives
        # forbidden: to penalize forbidden states
        
        # amplitude
        if 'envelope' in tfs.sys_para.reg_coeffs:
            tfs.reg_alpha_coeff = tfs.sys_para.reg_coeffs['envelope']
            reg_alpha = tfs.reg_alpha_coeff / float(tfs.sys_para.steps)
            tfs.reg_loss = tfs.reg_loss + reg_alpha * tf.nn.l2_loss(
                tf.mul(tfs.tf_one_minus_gaussian_envelope, tfs.ops_weight))

        # Constrain Z to have no dc value
        if 'dc' in tfs.sys_para.reg_coeffs:
            tfs.z_reg_alpha_coeff = tfs.sys_para.reg_coeffs['dc']
            z_reg_alpha = tfs.z_reg_alpha_coeff / float(tfs.sys_para.steps)
            for state in tfs.sys_para.reg_coeffs['dc_id']:
                segment_num = tfs.sys_para.reg_coeffs['dc_seg_num']
                segment = int(math.ceil(tfs.sys_para.steps / segment_num))
                for kk in range(segment_num - 1):
                    tfs.reg_loss = tfs.reg_loss + z_reg_alpha * tf.square(
                        tf.reduce_sum(tfs.ops_weight[state, kk * segment:(kk + 1) * segment]))
                tfs.reg_loss = tfs.reg_loss + z_reg_alpha * tf.square(
                    tf.reduce_sum(tfs.ops_weight[state, (segment_num - 1) * segment:]))

        # Limiting the dwdt of control pulse
        if 'dwdt' in tfs.sys_para.reg_coeffs:
            zeros_for_training = tf.zeros([tfs.sys_para.ops_len, 2])
            new_weights = tf.concat(1, [tfs.ops_weight, zeros_for_training])
            new_weights = tf.concat(1, [zeros_for_training, new_weights])
            tfs.dwdt_reg_alpha_coeff = tfs.sys_para.reg_coeffs['dwdt']
            dwdt_reg_alpha = tfs.dwdt_reg_alpha_coeff / float(tfs.sys_para.steps)
            tfs.reg_loss = tfs.reg_loss + dwdt_reg_alpha * tf.nn.l2_loss(
                (new_weights[:, 1:] - new_weights[:, :tfs.sys_para.steps + 3]) / tfs.sys_para.dt)

        # Limiting the d2wdt2 of control pulse
        if 'd2wdt2' in tfs.sys_para.reg_coeffs:
            tfs.d2wdt2_reg_alpha_coeff = tfs.sys_para.reg_coeffs['d2wdt2']
            d2wdt2_reg_alpha = tfs.d2wdt2_reg_alpha_coeff / float(tfs.sys_para.steps)
            tfs.reg_loss = tfs.reg_loss + d2wdt2_reg_alpha * tf.nn.l2_loss((new_weights[:, 2:] - \
                                                                              2 * new_weights[:,
                                                                                  1:tfs.sys_para.steps + 3] + new_weights[:,
                                                                                                               :tfs.sys_para.steps + 2]) / (
                                                                             tfs.sys_para.dt ** 2))
        # bandpass filter on the control    
        if 'bandpass' in tfs.sys_para.reg_coeffs:
            tfs.bandpass_reg_alpha_coeff = tfs.sys_para.reg_coeffs['bandpass']
            bandpass_reg_alpha = tfs.bandpass_reg_alpha_coeff/ float(tfs.sys_para.steps)
            
            tf_u = tf.cast(tfs.ops_weight,dtype=tf.complex64)
           
            tf_fft = tf.complex_abs(tf.fft(tf_u))
            
            band = np.array(tfs.sys_para.reg_coeffs['band'])

            band_id = (band*tfs.sys_para.total_time).astype(int)
            half_id = int(tfs.sys_para.steps/2)
            
            
            fft_loss = bandpass_reg_alpha*(tf.reduce_sum(tf_fft[:,0:band_id[0]]) + tf.reduce_sum(tf_fft[:,band_id[1]:half_id]))
            
            tfs.reg_loss = tfs.reg_loss + fft_loss
        

        # Limiting the access to forbidden states
        if 'forbidden' in tfs.sys_para.reg_coeffs:
            tfs.inter_reg_alpha_coeff = tfs.sys_para.reg_coeffs['forbidden']
            inter_reg_alpha = tfs.inter_reg_alpha_coeff / float(tfs.sys_para.steps)
            if tfs.sys_para.is_dressed:
                v_sorted = tf.constant(c_to_r_mat(np.reshape(sort_ev(tfs.sys_para.v_c, tfs.sys_para.dressed_id),
                                                             [len(tfs.sys_para.dressed_id), len(tfs.sys_para.dressed_id)])),
                                       dtype=tf.float32)

            for inter_vec in tfs.inter_vecs:
                if tfs.sys_para.is_dressed and ('forbid_dressed' in tfs.sys_para.reg_coeffs and tfs.sys_para.reg_coeffs['forbid_dressed']):
                    inter_vec = tf.matmul(tf.transpose(v_sorted), inter_vec)
                for state in tfs.sys_para.reg_coeffs['states_forbidden_list']:
                    forbidden_state_pop = tf.square(inter_vec[state, :]) + \
                                          tf.square(inter_vec[tfs.sys_para.state_num + state, :])
                    tfs.reg_loss = tfs.reg_loss + inter_reg_alpha * tf.nn.l2_loss(forbidden_state_pop)
                    
        # Speeding up the gate time
        if 'speed_up' in tfs.sys_para.reg_coeffs:
            tfs.speed_up_reg_alpha_coeff = - tfs.sys_para.reg_coeffs['speed_up']
            speed_up_reg_alpha = tfs.speed_up_reg_alpha_coeff / float(tfs.sys_para.steps)
            
            ####
            
            for ii in range(len(tfs.inter_vecs)):
                
                inter_vec =  tfs.inter_vecs[ii]
                
                if tfs.sys_para.state_transfer == False:
                    target_state = tfs.target_states[:,ii]
                else:
                    target_state = tfs.tf_target_vectors[ii]
                
                target_state_all_timestep = tf.tile(tf.reshape(target_state,[2*tfs.sys_para.state_num,1]) , [1, tfs.sys_para.steps+1])
                
                target_state_pop = tfs.get_inner_product_gen(target_state_all_timestep, inter_vec)
                
                tfs.reg_loss = tfs.reg_loss + speed_up_reg_alpha * tf.nn.l2_loss(target_state_pop)
            
            ####
            

        return tfs.reg_loss
                    
