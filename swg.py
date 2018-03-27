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

from utils.flags_wrapper import flags_wrapper


class swg():
    """swg
    The generative model.

    params:
        flags: A flags_wrapper object with all hyperparams
        model_name: Name for output folder. Will be created in "results/"     
    """

    def __init__(
            self,
            flags=None,
            model_name='test_experiment'):
        """initialization 
        """
        self.image_width = 64
        self.num_channels = 3
        self.image_size = self.num_channels * (self.image_width**2)

        self.flags = flags

        self.base_dir = 'results/' + model_name
        import os
        import errno

        try:
            ""
            os.makedirs(self.base_dir)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(self.base_dir):
                pass

        self.build_model()

        return

    def read_data(self):
        """read_data
        Assumes the data is in a single numpy array. Loads it into memory. Can be 
        replaced by tf queues.
        todo: Read from disk

        params: None

        returns:
            im: numpy array of flattened images [number of images, 64*64*3] 
        """
        path = '/tmp/cropped_celeba.npy'
        im = np.load(path)
        return im

    def sw_loss(self, true_distribution, generated_distribution):
        """sw_loss
        Computes the sliced Wasserstein distance between two sets of samples in the
        following way:
        1. Projects the samples onto random (Gaussian) directions (unit vectors).
        2. For each direction, computes the Wasserstein-2 distance by sorting the 
        two projected sets (which results in the lowest distance matching).
        3. Adds distance over all directions.

        NOTE:
        This will create ops that require a fixed batch size.

        params:
            true_distribution: Samples from the true distribution 
                [batch size, disc. feature size]
            generated_distribution: Samples from the generator 
                [batch size, disc. feature size]

        returns:
            sliced Wasserstein distance
        """
        s = true_distribution.get_shape().as_list()[-1]

        # theta contains the projection directions as columns :
        # [theta1, theta2, ...]
        theta = tf.random_normal(shape=[s, self.flags.num_projections])
        theta = tf.nn.l2_normalize(theta, axis=0)

        # project the samples (images). After being transposed, we have tensors
        # of the format: [projected_image1, projected_image2, ...].
        # Each row has the projections along one direction. This makes it
        # easier for the sorting that follows.
        projected_true = tf.transpose(
            tf.matmul(true_distribution, theta))

        projected_fake = tf.transpose(
            tf.matmul(generated_distribution, theta))

        sorted_true, true_indices = tf.nn.top_k(
            projected_true,
            self.flags.batch_size)

        sorted_fake, fake_indices = tf.nn.top_k(
            projected_fake,
            self.flags.batch_size)

        # For faster gradient computation, we do not use sorted_fake to compute
        # loss. Instead we re-order the sorted_true so that the samples from the
        # true distribution go to the correct sample from the fake distribution.
        # This is because Tensorflow did not have a GPU op for rearranging the
        # gradients at the time of writing this code.

        # It is less expensive (memory-wise) to rearrange arrays in TF.
        # Flatten the sorted_true from [batch_size, num_projections].
        flat_true = tf.reshape(sorted_true, [-1])

        # Modify the indices to reflect this transition to an array.
        # new index = row + index
        rows = np.asarray(
            [self.flags.batch_size * np.floor(i * 1.0 / self.flags.batch_size)
             for i in range(self.flags.num_projections * self.flags.batch_size)])
        rows = rows.astype(np.int32)
        flat_idx = tf.reshape(fake_indices, [-1, 1]) + np.reshape(rows, [-1, 1])

        # The scatter operation takes care of reshaping to the rearranged matrix
        shape = tf.constant([self.flags.batch_size * self.flags.num_projections])
        rearranged_true = tf.reshape(
            tf.scatter_nd(flat_idx, flat_true, shape),
            [self.flags.num_projections, self.flags.batch_size])

        return tf.reduce_mean(tf.square(projected_fake - rearranged_true))

    def build_model(self):
        """build_model
        Creates the computation graph.
        """

        # Input images from the true distribution
        self.x = tf.placeholder(
            tf.float32,
            [None, self.image_width, self.image_width, self.num_channels])

        # Latent variable
        self.z = tf.placeholder(tf.float32, [None, self.flags.latent_dim])

        # Output images from the GAN
        self.x_hat = generator(self.z)

        if self.flags.use_discriminator:
            # The discriminator returns the output (unnormalized) probability
            # of fake/true, and also a feature vector for the image.
            self.y, self.y_to_match = discriminator(self.x)
            self.y_hat, self.y_hat_to_match = discriminator(
                self.x_hat,
                reuse=True)

            # The discriminator is trained for simple binary classification.
            true_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.y),
                logits=self.y)
            fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(self.y_hat),
                logits=self.y_hat)
            self.discriminator_loss = tf.reduce_mean(true_loss + fake_loss)

            discriminator_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope='discriminator')
            self.d_optimizer = tf.train.AdamOptimizer(
                self.flags.learning_rate,
                beta1=0.5).minimize(self.discriminator_loss,
                                    var_list=discriminator_vars)

            self.generator_loss = self.sw_loss(
                self.y_to_match,
                self.y_hat_to_match)

        else:
            self.generator_loss = self.sw_loss(
                tf.reshape(self.x, [-1, self.image_size]),
                tf.reshape(self.x_hat, [-1, self.image_size]))

        generator_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='generator')
        self.g_optimizer = tf.train.AdamOptimizer(
            self.flags.learning_rate,
            beta1=0.5).minimize(self.generator_loss,
                                var_list=generator_vars)

        # self.merged_summary_op = tf.summary.merge_all()
        return

    def train(self):
        """train
        Main training loop. Saves a checkpoint and sample images periodically.
        """
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

        # Prefer not to use summaries, they seem to slow down execution over
        # time.
        # summary_writer = tf.summary.FileWriter(self.base_dir,sess.graph)

        curr_time = time.time()
        print("Starting code")
        for iteration in range(self.flags.max_iters):

            x = data[np.random.randint(0, max_examples, self.flags.batch_size)]
            z = np.random.uniform(
                low=-1,
                high=1,
                size=[self.flags.batch_size, self.flags.latent_dim])

            sess.run(self.g_optimizer, feed_dict={self.x: x, self.z: z})

            if self.flags.use_discriminator:
                if iteration % dfreq == 0:
                    for diteration in range(diter):
                        sess.run(
                            self.d_optimizer,
                            feed_dict={self.x: x, self.z: z})

            if iteration % 50 == 0:
                loss = sess.run(
                    self.generator_loss,
                    feed_dict={self.x: x, self.z: z})
                print(
                    "Time elapsed: {}, Loss after iteration {}: {}".format(
                        time.time() - curr_time,
                        iteration,
                        loss))
                curr_time = time.time()

            if iteration % 1000 == 0:
                z = np.random.uniform(
                    low=-1,
                    high=1,
                    size=[36, self.flags.latent_dim])
                im = sess.run(self.x_hat, feed_dict={self.z: z})
                im = np.reshape(im, (-1, self.image_width, self.num_channels))
                im = np.hstack(np.split(im, 6))

                plt.imshow(im)
                plt.axis('off')
                fig = plt.gcf()
                fig.set_size_inches(12, 12)
                plt.savefig(self.base_dir + '/Iteration_{}.png'.format(
                    iteration),
                    bbox_inches='tight')
                plt.close()

            if iteration % 10000 == 0:
                saver.save(sess, self.base_dir + '/checkpoint.ckpt')

        return

    def generate_images(self):
        """generate_images
        Method to generate samples using a pre-trained model 
        """
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(self.base_dir + '/'))

        z = np.random.uniform(
            low=-1,
            high=1,
            size=[36, self.flags.latent_dim])

        im = sess.run(self.x_hat, feed_dict={self.z: z})

        im = np.reshape(im, (-1, self.image_width, self.num_channels))
        im = np.hstack(np.split(im, 6))

        plt.imshow(im)
        plt.axis('off')
        fig = plt.gcf()
        fig.set_size_inches(12, 12)
        plt.savefig(self.base_dir + '/Samples.png', bbox_inches='tight')
        plt.close()
        return

    def generate_tsne(self):
        """generate_tsne
        Method to visualize TSNE with random samples from the ground truth and
        generated distribution. This might help in catching mode collapse. If
        there is an obvious case of mode collapse, then we should see several
        points from the ground truth without any generated samples nearby.
        Purely a sanity check.
        """
        from sklearn.manifold import TSNE

        num_points = 1000
        data = self.read_data()[:num_points]
        data = np.reshape(data, [num_points, -1])

        print("Loaded ground truth.")

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(self.base_dir + '/'))
        z = np.random.uniform(-1, 1, size=[num_points, self.flags.latent_dim])

        generated = sess.run(self.x_hat, feed_dict={self.z: z})
        generated = np.reshape(generated, [num_points, -1])

        X = np.vstack((data, generated))

        print("Computing TSNE.")
        X_embedded = TSNE(n_components=2).fit_transform(X)
        print("Plotting data.")

        plt.scatter(
            X_embedded[:num_points, 0],
            X_embedded[:num_points, 1],
            color='r',
            label='GT')

        plt.scatter(
            X_embedded[num_points:, 0],
            X_embedded[num_points:, 1],
            color='b',
            label='Generated',
            alpha=0.25)

        plt.axis('off')
        fig = plt.gcf()
        fig.set_size_inches(12, 12)
        plt.savefig(self.base_dir + '/TSNE.png', bbox_inches='tight')
        plt.close()

        return


def main(argv=None):
    parser = argparse.ArgumentParser(description='SWGAN')

    parser.add_argument(
        '--name',
        metavar='output folder',
        default="test",
        help='Output folder')

    parser.add_argument(
        '--train',
        dest='train',
        action='store_true',
        help='Use to train')

    parser.add_argument(
        '--learning_rate',
        metavar='learning rate',
        default=1e-4,
        help='Learning rate for optimizer')

    parser.add_argument(
        '--max_iters',
        metavar='max iters',
        default=10000,
        help='Number of iterations to train')

    parser.add_argument(
        '--num_projections',
        metavar='num projections',
        default=10000,
        help='Number of projections to use at every step')

    parser.add_argument(
        '--batch_size',
        metavar='batch size',
        default=64,
        help='Batch size')

    parser.add_argument(
        '--use_discriminator',
        dest='use_discriminator',
        action='store_true',
        help='Enable discriminator')

    args = parser.parse_args()

    np.random.seed(np.random.randint(0, 10))
    tf.set_random_seed(np.random.randint(0, 10))
    tf.reset_default_graph()

    flags = flags_wrapper(
        learning_rate=args.learning_rate,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        num_projections=args.num_projections,
        use_discriminator=args.use_discriminator)

    g = swg(model_name=args.name, flags=flags)

    if args.train:
        g.train()
    g.generate_images()
    g.generate_tsne()
    return

if __name__ == '__main__':
    main()
