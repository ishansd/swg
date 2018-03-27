import tensorflow as tf
import tensorflow.contrib.layers as layers


def discriminator(x, reuse=False):
    """discriminator
      Network to classify fake and true samples.
      params:
        x: Input images [batch size, 64, 64, 3]
      returns:
        y: Unnormalized probablity of sample being real [batch size, 1]
        h: Features from penultimate layer of discriminator 
          [batch size, feature dim]
    """
    batch_norm = layers.layer_norm

    h = x
    with tf.variable_scope("discriminator", reuse=reuse) as scope:
        h = layers.conv2d(
            inputs=h,
            num_outputs=64,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        # [32,32,64]

        h = layers.conv2d(
            inputs=h,
            num_outputs=128,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        # [16,16,128]

        h = layers.conv2d(
            inputs=h,
            num_outputs=256,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        # [8,8,256]

        h = layers.conv2d(
            inputs=h,
            num_outputs=512,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        # [4,4,512]

        h = layers.flatten(h)
        y = layers.fully_connected(
            inputs=h,
            num_outputs=1,
            activation_fn=None,
            biases_initializer=None)
    return y, h
