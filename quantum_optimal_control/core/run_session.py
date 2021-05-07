"""
run_session.py - This module contains the main logic for running a grape session.
"""

import os
import time
from datetime import datetime

import numpy as np
from scipy.optimize import minimize
import tensorflow as tf

from .analysis import Analysis
from qoc.helper_functions.data_management import H5File


# TODO: Fields documentation is incomplete.
class run_session:
    """A class to encapsulate a grape session.
    Fields:
    """
    def __init__(self, tfs, graph, conv, sys_para, method,
                 show_plots=True, single_simulation=False, use_gpu=True):
        self.tfs = tfs
        self.graph = graph
        self.conv = conv
        self.sys_para = sys_para
        self.update_step = conv.update_step
        self.iterations = 0
        self.method = method.upper()
        self.show_plots = show_plots
        self.target = False
        self.drive_op = None
        if self.sys_para.drive_squared:
            # formula determined from how the tfs.ops_weight are defined (divided by max operator amplitudes)
            # extra multiplied factor carries over from definition of drive operator (drive_scale)
            self.drive_op = tf.assign(self.tfs.ops_weight_base[-1],
                                      tf.asin((tf.square(tf.sin(self.tfs.ops_weight_base[-2])) +
                                               tf.square(tf.sin(self.tfs.ops_weight_base[-3]))) *
                                              self.sys_para.ops_max_amp[-2] ** 2 /
                                              self.sys_para.ops_max_amp[-1] * 1.0))
        elif self.sys_para.integral_zero:
            with self.graph.as_default():
                self.drive_op = tf.assign(self.tfs.ops_weight_base[0], tf.asin(tf.clip_by_value(tf.sin(self.tfs.ops_weight_base[0])
                                                                               - tf.reduce_mean(tf.sin(self.tfs.ops_weight_base[0])),
                                                                                        clip_value_min=-1.0,
                                                                                        clip_value_max=1.0)))

        if not use_gpu:
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            config = None

        with tf.Session(graph=graph, config=config) as self.session:
            tf.global_variables_initializer().run()
            print("Initialized", flush=True)
            
            # Run the desired optimization based on the specified method.
            if self.method == 'EVOLVE':
                self.start_time = time.time()
                x0 = self.sys_para.ops_weight_base
                self.l, self.rl, self.grads, self.metric, self.g_squared = self.get_error(
                    x0)
                self.get_end_results()
            elif self.method == 'ADAM':
                self.start_adam_optimizer()
            else:
                self.bfgs_optimize(method=self.method)


    def start_adam_optimizer(self):
        # adam optimizer
        self.start_time = time.time()
        self.end = False

        if self.sys_para.LRF:
            print("Learning Rate Finder Test")
            growth = 8.0
            self.conv.max_iterations = int(growth * 8)
            self.conv.max_iterations = 100
            with open("C:/Users/hek/Desktop/Learning Rate Tests/LRF_" + datetime.now().strftime('%Y-%m-%d %H-%M-%S') +
                      ".txt", "w") as f:
                while True:
                    self.g_squared, self.l, self.rl, self.metric = self.session.run(
                        [self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss, self.tfs.unitary_scale])
                    if (self.l < self.conv.conv_target) or (self.g_squared < self.conv.min_grad) \
                            or (self.iterations >= self.conv.max_iterations):
                        self.end = True

                    self.update_and_save()

                    if self.end:
                        self.get_end_results()
                        break

                    # a min start, max end
                    # learning_rate = 0.002 + (0.1 - 0.002)/self.conv.max_iterations * self.iterations
                    learning_rate = 10**(-6 + self.iterations/growth)
                    learning_rate = 10**(-4) + (0.1 - 10**(-4)) / self.iterations
                    print("learning rate: ", learning_rate)
                    f.write(str(learning_rate) + "\t" + str(self.l) + "\t" + str(self.rl) + "\t" + str(self.g_squared)
                            + "\t" + str(self.iterations) + "\n")
                    f.flush()  # write to file immediately instead of sitting in buffer

                    self.feed_dict = {self.tfs.learning_rate: learning_rate}

                    _ = self.session.run([self.tfs.optimizer],
                                         feed_dict=self.feed_dict)
                    if self.drive_op is not None:
                        self.session.run([self.drive_op])
                f.close()
        else:
            while True:
                self.g_squared, self.l, self.rl, self.metric = self.session.run(
                    [self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss, self.tfs.unitary_scale])

                if (self.l < self.conv.conv_target) or (self.g_squared < self.conv.min_grad) \
                        or (self.iterations >= self.conv.max_iterations):
                    self.end = True

                self.update_and_save()

                if self.end:
                    self.get_end_results()
                    break

                # exponentially decaying learning rate
                learning_rate = float(self.conv.rate) * np.exp(-float(self.iterations) / self.conv.learning_rate_decay)

                # Stochastic Gradient Descent with Warm Restarts learning rate (Loshchilov, Hutter)
                # period 400, min_rate 0.001, max_rate self.conv.rate
                #self.learning_rate = 0.001 + 0.5 * (self.conv.rate - 0.001) * (1 + np.cos(
                #    np.pi / self.conv.learning_rate_decay * (self.iterations % int(self.conv.learning_rate_decay))))

                self.feed_dict = {self.tfs.learning_rate: learning_rate}

                self.session.run([self.tfs.optimizer], feed_dict=self.feed_dict)
                if self.drive_op is not None:
                    self.session.run([self.drive_op])


    def update_and_save(self):
        if not self.end:
            if (self.iterations % self.conv.update_step == 0):
                self.anly = Analysis(self.sys_para, self.tfs.final_state, self.tfs.ops_weight, self.tfs.unitary_scale,
                                     self.tfs.inter_vecs)
                self.save_data()
                self.display()
            if (self.iterations % self.conv.evol_save_step == 0):
                if not (self.sys_para.show_plots == True and (self.iterations % self.conv.update_step == 0)):
                    self.anly = Analysis(self.sys_para, self.tfs.final_state,
                                         self.tfs.ops_weight, self.tfs.unitary_scale,
                                         self.tfs.inter_vecs)
                    if not (self.iterations % self.conv.update_step == 0):
                        self.save_data()
                    self.conv.save_evol(self.anly)

            self.iterations += 1


    def get_end_results(self):
        # get optimized pulse and propagation

        # get and save inter vects

        self.anly = Analysis(self.sys_para, self.tfs.final_state, self.tfs.ops_weight, self.tfs.unitary_scale,
                             self.tfs.inter_vecs)
        self.save_data()
        self.display()
        if not self.show_plots:
            self.conv.save_evol(self.anly)

        self.uks = self.Get_uks()
        if not self.sys_para.state_transfer:
            self.Uf = self.anly.get_final_state()
        else:
            self.Uf = []


    def Get_uks(self):
        # to get the pulse amplitudes
        uks = self.anly.get_ops_weight()
        for ii in range(len(uks)):
            uks[ii] = self.sys_para.ops_max_amp[ii]*uks[ii]
        return uks


    def get_error(self, uks):
        # get error and gradient for scipy bfgs:
        self.session.run(self.tfs.ops_weight_base.assign(uks))

        g, l, rl, metric, g_squared = self.session.run(
            [self.tfs.grad_pack, self.tfs.loss, self.tfs.reg_loss, self.tfs.unitary_scale, self.tfs.grad_squared])

        final_g = np.transpose(np.reshape(
            g, (len(self.sys_para.ops_c)*self.sys_para.steps)))

        return l, rl, final_g, metric, g_squared


    def save_data(self):
        if self.sys_para.save:
            self.elapsed = time.time() - self.start_time
            with H5File(self.sys_para.file_path) as hf:
                hf.append('error', np.array(self.l))
                hf.append('reg_error', np.array(self.rl))
                hf.append('uks', np.array(self.Get_uks()))
                hf.append('iteration', np.array(self.iterations))
                hf.append('run_time', np.array(self.elapsed))
                hf.append('unitary_scale', np.array(self.metric))


    def display(self):
        # display of simulation results

        if self.show_plots:
            self.conv.update_plot_summary(self.l, self.rl, self.anly)
        else:
            self.elapsed = time.time() - self.start_time
            print('Error = :%1.2e; Runtime: %.1fs; Iterations = %d, grads =  %10.3e, unitary_metric = %.5f' % (self.l, self.elapsed, self.iterations, self.g_squared, self.metric), flush=True)


    def minimize_opt_fun(self, x):
        # minimization function called by scipy in each iteration
        self.l, self.rl, self.grads, self.metric, self.g_squared = self.get_error(
            np.reshape(x, (len(self.sys_para.ops_c), int(len(x)/len(self.sys_para.ops_c)))))

        if self.l < self.conv.conv_target:
            self.conv_time = time.time()-self.start_time
            self.conv_iter = self.iterations
            self.end = True
            print('Target fidelity reached', flush=True)
            self.grads = 0*self.grads  # set zero grads to terminate the scipy optimization

        self.update_and_save()

        if self.method == 'L-BFGS-B':
            return np.float64(self.rl), np.float64(np.transpose(self.grads))
        else:
            return self.rl, np.reshape(np.transpose(self.grads), [len(np.transpose(self.grads))])


    def bfgs_optimize(self, method='L-BFGS-B', jac=True, options=None):
        # scipy optimizer
        self.conv.reset_convergence()
        self.first = True
        self.conv_time = 0.
        self.conv_iter = 0
        self.end = False
        print("Starting " + self.method + " Optimization", flush=True)
        self.start_time = time.time()

        x0 = self.sys_para.ops_weight_base
        options = {'maxfun': self.conv.max_iterations, 'ftol': 1e-12, 'maxiter': 1e6,
                   'gtol': self.conv.min_grad, 'disp': False, 'maxls': 40}

        res = minimize(self.minimize_opt_fun, x0,
                       method=method, jac=jac, options=options)

        uks = np.reshape(res['x'], (len(self.sys_para.ops_c),
                                    int(len(res['x'])/len(self.sys_para.ops_c))))

        print(self.method + ' optimization done', flush=True)

        g, l, rl = self.session.run(
            [self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss])

        if self.sys_para.show_plots == False:
            print(res.message, flush=True)
            print(("Error = %1.2e" % l), flush=True)
            print(("Total time is " + str(time.time() - self.start_time)), flush=True)

        self.get_end_results()
