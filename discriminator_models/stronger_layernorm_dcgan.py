import tensorflow as tf
import tensorflow.contrib.layers as layers

def discriminator(x, reuse=False):
  norm = layers.layer_norm

  h = x
  with tf.variable_scope("discriminator", reuse=reuse) as scope:
    h = tf.reshape(h,[-1,64,64,3])
    h = layers.conv2d(inputs=h, num_outputs=2*64, kernel_size=[4,4],stride=2,activation_fn=tf.nn.leaky_relu, normalizer_fn=norm)
    # 32,32,64
    h = layers.conv2d(inputs=h, num_outputs=2*128, kernel_size=[4,4],stride=2,activation_fn=tf.nn.leaky_relu, normalizer_fn=norm)
    # 16,16,128
    h = layers.conv2d(inputs=h, num_outputs=2*256, kernel_size=[4,4],stride=2,activation_fn=tf.nn.leaky_relu, normalizer_fn=norm)
    # 8,8,256
    h = layers.conv2d(inputs=h, num_outputs=2*512, kernel_size=[4,4],stride=2,activation_fn=tf.nn.leaky_relu, normalizer_fn=norm)
    # 4,4,512
    h = layers.flatten(h)
    y = layers.fully_connected(inputs=h, num_outputs=1, activation_fn=None, biases_initializer=None)
  return y,h
