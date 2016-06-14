import tensorflow as tf
from tensorflow.python.framework import ops

@ops.RegisterGradient("MatrixExp")
def _matrix_exp_grad(op, grad):
    
    matrix_array = tf.constant(op.get_attr("matrix"),dtype=tf.float32)
    
    size = op.get_attr("size")

    coeff_0_grad = tf.reduce_sum(tf.mul(grad,tf.matmul(tf.reshape(matrix_array[0:size**2],[size,size]),op.outputs[0])))
    coeff_1_grad = tf.reduce_sum(tf.mul(grad,tf.matmul(tf.reshape(matrix_array[size**2:2*size**2],[size,size]),op.outputs[0])))
    coeff_2_grad = tf.reduce_sum(tf.mul(grad,tf.matmul(tf.reshape(matrix_array[2*size**2:3*size**2],[size,size]),op.outputs[0])))
    
    return [tf.pack([coeff_0_grad,coeff_1_grad,coeff_2_grad])]
