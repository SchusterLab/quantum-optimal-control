import tensorflow as tf
from tensorflow.python.framework import ops

@ops.RegisterGradient("MatrixExp")
def _matrix_exp_grad(op, grad):
    
    matrix_0_array = tf.constant(op.get_attr("matrix_0"),dtype=tf.float32)
    matrix_1_array = tf.constant(op.get_attr("matrix_1"),dtype=tf.float32)
    matrix_2_array = tf.constant(op.get_attr("matrix_2"),dtype=tf.float32)
    
    size = op.get_attr("size")

    coeff_0_grad = tf.reduce_sum(tf.mul(grad,tf.matmul(tf.reshape(matrix_0_array,[size,size]),op.outputs[0])))
    coeff_1_grad = tf.reduce_sum(tf.mul(grad,tf.matmul(tf.reshape(matrix_1_array,[size,size]),op.outputs[0])))
    coeff_2_grad = tf.reduce_sum(tf.mul(grad,tf.matmul(tf.reshape(matrix_2_array,[size,size]),op.outputs[0])))
    
    return [coeff_0_grad,coeff_1_grad,coeff_2_grad]
