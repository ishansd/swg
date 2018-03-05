import tensorflow as tf
import tensorflow.contrib.layers as layers

"""
    generator
    Network to produce samples.
    params:
        z:  Input noise [batch size, latent dimension]
    returns:
        x_hat: Artificial image [batch size, 64, 64, 3]
"""
def generator(z):
    batch_norm = layers.batch_norm

    h = z
    with tf.variable_scope("generator") as scope:
        h = layers.fully_connected(inputs=h,
                                   num_outputs=4 * 4 * 1024,
                                   activation_fn=tf.nn.relu,
                                   normalizer_fn=batch_norm)
        h = tf.reshape(h, [-1, 4, 4, 1024])
        # [4,4,1024]

        h = layers.conv2d_transpose(inputs=h,
                                    num_outputs=512,
                                    kernel_size=4,
                                    stride=2,
                                    activation_fn=tf.nn.relu,
                                    normalizer_fn=batch_norm)
        # [8,8,512]
        h = layers.conv2d_transpose(inputs=h,
                                    num_outputs=256,
                                    kernel_size=4,
                                    stride=2,
                                    activation_fn=tf.nn.relu,
                                    normalizer_fn=batch_norm)

        # [16,16,256]
        h = layers.conv2d_transpose(inputs=h,
                                    num_outputs=128,
                                    kernel_size=4,
                                    stride=2,
                                    activation_fn=tf.nn.relu,
                                    normalizer_fn=batch_norm)

        # This is an extra conv layer like the WGAN folks.
        h = layers.conv2d(inputs=h,
                          num_outputs=128,
                          kernel_size=4,
                          stride=1,
                          activation_fn=tf.nn.relu,
                          normalizer_fn=batch_norm)

        # [32,32,128]
        x_hat = layers.conv2d_transpose(inputs=h,
                                    num_outputs=3,
                                    kernel_size=4,
                                    stride=2,
                                    activation_fn=tf.nn.sigmoid,
                                    biases_initializer=None)
        # [64,64,3]

        return x_hat
