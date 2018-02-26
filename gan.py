from __future__ import print_function

import numpy as np
import tensorflow as tf

import cv2

import argparse


"""
For saving plots on the cluster,
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


from generator import generator
from discriminator import discriminator

import time

"""
    flags
    My wrapper for all hyperparameters. This will be passed to the generative model.
"""


class flags_object():

    def __init__(self,
                 learning_rate=1e-4,
                 batch_size=64,
                 latent_dim=100,
                 max_iters=10000,
                 num_projections=20000,
                 use_discriminator=True):

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iters = max_iters

        # For input noise
        self.latent_dim = latent_dim

        # SWG specific params
        self.num_projections = num_projections
        self.use_discriminator = use_discriminator

        return


"""
    swg
    The generative model.
"""


class swg():

    def __init__(self,
                 flags=None,
                 model_name='test_experiment'):

        self.image_width = 64
        self.num_channels = 3
        self.image_size = self.num_channels * (self.image_width**2)

        self.flags = flags

        self.base_dir = 'results/' + model_name
        import os
        import errno

        try:
            os.makedirs(self.base_dir)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(self.base_dir):
                pass

        self.build_model()

        return

    """
  Load the data
    Assumes the data is in a single numpy array. Loads it into memory.
  """

    def read_data(self):
        path = '/tmp/cropped_celeba.npy'
        im = np.load(path)
        im = np.reshape(im, [-1, self.image_size])
        return im

    """
  Sliced-Wasserstein loss
    Projects the images onto randomly chosen directions and computes the Wasserstein distance
    between the two empirical distributions. The loss is the sum of the distances along all
    such projections.
    @params t_ Samples from the true distribution
    @params f_ Samples from the generator
    @params num_projections Number of random directions to project images onto
    @params reference If true, use a fixed set of directions. This will be used for comparison
  """

    def sw_loss(self, t, f):
        s = t.get_shape().as_list()[-1]

        theta = tf.random_normal(shape=[s, self.flags.num_projections])
        normal_theta = tf.nn.l2_normalize(theta, dim=0)

        x_t = tf.transpose(tf.matmul(t, normal_theta))
        x_f = tf.transpose(tf.matmul(f, normal_theta))

        sorted_true, _ = tf.nn.top_k(x_t, self.flags.batch_size)
        sorted_fake, fake_indices = tf.nn.top_k(x_f, self.flags.batch_size)

        flat_true = tf.reshape(sorted_true, [-1])
        rows = np.asarray([self.flags.batch_size * int(np.floor(i * 1.0 / self.flags.batch_size))
                           for i in range(self.flags.num_projections * self.flags.batch_size)])

        flat_idx = tf.reshape(fake_indices, [-1, 1]) + np.reshape(rows, [-1, 1])

        shape = tf.constant([self.flags.batch_size * self.flags.num_projections])
        rearranged_true = tf.reshape(tf.scatter_nd(flat_idx, flat_true, shape),
                                     [self.flags.num_projections, self.flags.batch_size])

        return tf.reduce_mean(tf.square(x_f - rearranged_true))

    """
  Creates the computation graph
  """

    def build_model(self):

        # Input images from the TRUE distribution
        self.x = tf.placeholder(tf.float32, [None, self.image_size])

        # Latent variable
        self.z = tf.placeholder(tf.float32, [None, self.flags.latent_dim])

        # Output images from the GAN
        self.x_hat = generator(self.z)

        if self.flags.use_discriminator:
            print("Using discriminator")
            self.y, self.y_to_match = discriminator(self.x)
            self.y_hat, self.y_hat_to_match = discriminator(self.x_hat, reuse=True)


            true_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.y),
                                                                logits=self.y)
            fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.y_hat),
                                                                logits=self.y_hat)
            self.discriminator_loss = tf.reduce_mean(true_loss + fake_loss)

            discriminator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                   scope='discriminator')

            self.generator_loss = self.sw_loss(self.y_to_match, self.y_hat_to_match)

            self.d_optimizer = tf.train.AdamOptimizer(self.flags.learning_rate,
                                                      beta1=0.5).minimize(self.discriminator_loss,
                                                                            var_list=discriminator_vars)

        else:
            self.generator_loss = self.sw_loss(self.x, self.x_hat)

        generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

        self.g_optimizer = tf.train.AdamOptimizer(self.flags.learning_rate,
                                                  beta1=0.5).minimize(self.generator_loss,
                                                                      var_list=generator_vars)

        # self.merged_summary_op = tf.summary.merge_all()
        return

    """
  Main training loop. Saves a checkpoint and sample images after every epoch.
  """

    def train(self):
        dfreq = 1
        diter = 1

        print("Loading data into memory.")
        data = self.read_data()
        max_examples = data.shape[0]
        print("Loaded {} examples".format(max_examples))

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())

        # summary_writer = tf.summary.FileWriter(self.base_dir,sess.graph)

        curr_time = time.time()
        print("Starting code")
        for iteration in range(int(self.flags.max_iters)):
 
            x = data[np.random.randint(0, max_examples, self.flags.batch_size)]
            z = np.random.uniform(-1, 1, size=[self.flags.batch_size, self.flags.latent_dim])

            sess.run(self.g_optimizer, feed_dict={self.x: x, self.z: z})

            if self.flags.use_discriminator:
                if iteration % dfreq == 0:
                    for diteration in range(diter):
                        sess.run(self.d_optimizer, feed_dict={self.x: x, self.z: z})

            if iteration % 50 == 0:
                l = sess.run(self.generator_loss, feed_dict={self.x: x, self.z: z})
                print("Time elapsed: {}, Loss at iteration {}: {}".format(time.time() - curr_time,
                                                                          iteration,
                                                                          l))
                curr_time = time.time()

            if iteration % 1000 == 0:
                z = np.random.uniform(-1, 1, size=[36, self.flags.latent_dim])
                im = sess.run(self.x_hat, feed_dict={self.z: z})
                im = np.reshape(im, (-1, self.image_width, self.num_channels))
                im = np.hstack(np.split(im, 6))

                # I made an error while creating the numpy array for LSUN, which swapped B and R
                # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

                plt.imshow(im)
                plt.axis('off')
                fig = plt.gcf()
                fig.set_size_inches(12, 12)
                plt.savefig(self.base_dir + '/Iteration_{}.png'.format(iteration),
                            bbox_inches='tight')
                plt.close()

            if iteration % 10000 == 0:
                saver.save(sess, self.base_dir + '/checkpoint.ckpt')

        return

    """
  Method to generate samples using a pre-trained model 
  """

    def generate_images(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(self.base_dir + '/'))

        z = np.random.uniform(-1, 1, size=[36, self.flags.latent_dim])

        im = sess.run(self.x_hat, feed_dict={self.z: z})

        im = np.reshape(im, (-1, self.image_width, self.num_channels))
        im = np.hstack(np.split(im, 6))
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(im)
        plt.axis('off')
        fig = plt.gcf()
        fig.set_size_inches(12, 12)
        plt.savefig(self.base_dir + '/Samples.png', bbox_inches='tight')
        plt.close()
        return


def main(argv=None):
    parser = argparse.ArgumentParser(description='SWGAN')

    parser.add_argument(
        '--name', metavar='output folder', default="test", help='Output folder')

    parser.add_argument('--train',
                        dest='train',
                        action='store_true',
                        help='Use to train')

    parser.add_argument('--learning_rate',
                        metavar='learning rate',
                        default=1e-4,
                        help='Learning rate for optimizer')

    parser.add_argument('--max_iters',
                        metavar='max iters',
                        default=10000,
                        help='Number of iterations to train')

    parser.add_argument('--num_projections',
                        metavar='num projections',
                        default=10000,
                        help='Number of projections to use at every step')

    parser.add_argument('--batch_size',
                        metavar='batch size',
                        default=64,
                        help='Batch size')

    parser.add_argument('--use_discriminator',
                        dest='use_discriminator',
                        action='store_true',                        
                        help='Enable discriminator')

    args = parser.parse_args()

    print((args.use_discriminator))

    np.random.seed(np.random.randint(0, 10))
    tf.set_random_seed(np.random.randint(0, 10))
    tf.reset_default_graph()

    flags = flags_object(learning_rate=args.learning_rate,
                         max_iters=args.max_iters,
                         batch_size=args.batch_size,
                         num_projections=args.num_projections,
                         use_discriminator=args.use_discriminator)

    g = swg(model_name=args.name, flags=flags)

    if args.train:
        g.train()
    g.generate_images()

    return

if __name__ == '__main__':
    main()
