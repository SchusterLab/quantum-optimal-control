import tensorflow as tf
from tensorflow.python.framework import ops

@ops.RegisterGradient("MatrixExp")
def _matrix_exp_grad(op, grad):
    
    size = op.get_attr("size")
    input_num = op.get_attr("input_num")

    coeff_grad = []

    coeff_grad.append(tf.constant(0,dtype=tf.float32))
    for ii in range(1,input_num):
    	coeff_grad.append(tf.reduce_sum(tf.mul(grad,tf.matmul(tf.reshape(op.inputs[1][ii*size**2:(ii+1)*size**2],[size,size]),op.outputs[0]))))

    return [tf.pack(coeff_grad), tf.zeros(tf.shape(op.inputs[1]),dtype=tf.float32)]
