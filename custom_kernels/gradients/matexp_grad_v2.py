import tensorflow as tf
from tensorflow.python.framework import ops

@ops.RegisterGradient("MatrixExp")
def _matrix_exp_grad(op, grad):
    
    matrix_array = tf.constant(op.get_attr("matrix"),dtype=tf.float32)
    
    size = op.get_attr("size")
    input_num = op.get_attr("input_num")

    coeff_grad = []

    for ii in range(input_num):
    	coeff_grad.append(tf.reduce_sum(tf.mul(grad,tf.matmul(tf.reshape(matrix_array[ii*size**2:(ii+1)*size**2],[size,size]),op.outputs[0]))))

    return [tf.pack(coeff_grad)]
