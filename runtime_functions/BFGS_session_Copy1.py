if self.sys_para.ops_len != len(self.sys_para.Dts):
                        self.ops_weight_base = tf.truncated_normal([self.sys_para.ops_len - len(self.sys_para.Dts) ,self.sys_para.steps],
                                                                   mean= initial_guess ,dtype=tf.float32,
                            stddev=initial_stddev )
                        self.raws = self.ops_weight_base
                       
                    else:
                        initial_stddev = (0.1/np.sqrt(self.sys_para.ctrl_steps[0]))
                        weight = tf.truncated_normal([1 ,self.sys_para.ctrl_steps[0]],
                                                                       mean= initial_guess ,dtype=tf.float32,
                                stddev=initial_stddev )
                        
                        self.raw_weight.append(weight)
                        interpolated_weight = self.transfer_fn_general(weight,self.sys_para.ctrl_steps[0])
                        self.ops_weight_base = interpolated_weight
                        self.raws = weight
                        index =1
                    
                    for ii in range (len(self.sys_para.Dts)-index):
                        
                        initial_stddev = (0.1/np.sqrt(self.sys_para.ctrl_steps[ii+index]))
                        weight = tf.truncated_normal([1 ,self.sys_para.ctrl_steps[ii+index]],
                                                                       mean= initial_guess ,dtype=tf.float32,
                                stddev=initial_stddev )
                        self.raw_weight.append(weight)
                        interpolated_weight = self.transfer_fn_general(weight,self.sys_para.ctrl_steps[ii+index])
                        self.ops_weight_base = tf.concat(0,[self.ops_weight_base,interpolated_weight])
                        self.raws = tf.concat(1,[self.raws,weight])
                    #self.raw_weight = tf.pack(self.raw_weight)
                    self.raw_weights = tf.Variable(self.raws, name = "weights")
                