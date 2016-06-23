import tensorflow as tf
from tensorflow.python.framework import ops
import os

def register_gradient(matrix_exp_module):

    @ops.RegisterGradient("MatrixExpVecs")
    def _matrix_exp_vecs_grad(op, grad):

        matrix_array = tf.constant(op.get_attr("matrix"),dtype=tf.float32)

        size = op.get_attr("size")
        input_num = op.get_attr("input_num")
        exp_num = op.get_attr("exp_num")
        vecs_num = op.get_attr("vecs_num")
        matrix = op.get_attr("matrix")

        coeff_grad = []

        coeff_grad.append(tf.constant(0,dtype=tf.float32))
        for ii in range(1,input_num):
            coeff_grad.append(tf.reduce_sum(tf.mul(grad,tf.matmul(tf.reshape(matrix_array[ii*size**2:(ii+1)*size**2],[size,size]),op.outputs[0]))))
    
    #user_ops_path = '/home/nelson/Simulations/GRAPE-Tensorflow/custom_kernels/build'
    #matrix_exp_module = tf.load_op_library(os.path.join(user_ops_path,'cuda_matexp_vecs.so'))

 
        vec_grad = matrix_exp_module.matrix_exp_vecs_grads(op.inputs[0],grad ,size = size,input_num = input_num, exp_num = exp_num,vecs_num = vecs_num,matrix= matrix)


        return [tf.pack(coeff_grad),vec_grad]
