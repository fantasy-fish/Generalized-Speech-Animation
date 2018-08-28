import tensorflow as tf

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "linear"):
        matrix = tf.get_variable("matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        return tf.matmul(input_, matrix) + bias

def dropout(input_,keep_prob):
    return tf.nn.dropout(input_,keep_prob)