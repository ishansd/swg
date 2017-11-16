import tensorflow as tf
import tensorflow.contrib.layers as layers

def generator(z):
  batch_norm = layers.batch_norm
    
  h = z
  with tf.variable_scope("generator") as scope:
    h = layers.fully_connected(inputs=h, num_outputs=4*4*1024, activation_fn=tf.nn.relu, normalizer_fn=batch_norm)
    h = tf.reshape(h, [-1,4,4,1024])
    # [4,4,1024]

    h = layers.conv2d_transpose(inputs = h, num_outputs=512, kernel_size=[4,4],
                                                    stride=2,
                                                    activation_fn=tf.nn.relu,
                                                    normalizer_fn=batch_norm)
    h = layers.conv2d(inputs = h, num_outputs=512, kernel_size=[4,4],
                                                    stride=1,
                                                    activation_fn=tf.nn.relu,
                                                    normalizer_fn=batch_norm)
    # [8,8,512]
    h = layers.conv2d_transpose(inputs = h, num_outputs=256, kernel_size=[4,4],
                                                    stride=2,
                                                    activation_fn=tf.nn.relu,
                                                    normalizer_fn=batch_norm)

    h = layers.conv2d(inputs = h, num_outputs=256, kernel_size=[4,4],
                                                    stride=1,
                                                    activation_fn=tf.nn.relu,
                                                    normalizer_fn=batch_norm)
    # [16,16,256]
    h = layers.conv2d_transpose(inputs = h, num_outputs=128, kernel_size=[4,4],
                                                    stride=2,
                                                    activation_fn=tf.nn.relu,
                                                    normalizer_fn=batch_norm)

    h = layers.conv2d(inputs = h, num_outputs=128, kernel_size=[4,4],
                                                    stride=1,
                                                    activation_fn=tf.nn.relu,
                                                    normalizer_fn=batch_norm)

    # [32,32,128]
    h = layers.conv2d_transpose(inputs = h, num_outputs=3, kernel_size=[4,4],
                                                    stride=2,
                                                    activation_fn=tf.nn.tanh,
                                                    biases_initializer=None)
    # [64,64,3]

    x_hat = layers.flatten(h)
    return (x_hat+1)/2
