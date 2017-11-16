from __future__ import print_function

import tensorflow as tf

import numpy as np

import cv2

"""
For saving plots on the cluster
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

"""
All generator and discriminator models are in folders
generator_models/
discriminator_models/
Each model is in xyz.py, and must have a function named
generator()
discriminator()
"""
from generator_models import *
from discriminator_models import *

import time

"""
The generative model.
  @params GAN type Either sliced Wasserstein GAN, or Wasserstein GAN
  @params generator_model File name for generator model
  @params discriminator_model File name for discriminator model
  @params learning_rate LR for Adam Optimizer
  @params max_epochs Max epochs to train
  @params batch_size Equivalent to sample size for SW distance
  @params num_theta Number of projections to use in every iteration
  @params d_freq Relative frequency of generator:discriminator training
  @params d_iter Number of successive iterations for training discriminator
  @params model_name Output folder name
"""
class gan():
  def __init__(
      self, 
      gan_type='swg',
      generator_model='dcgan',
      discriminator_model='layernorm_dcgan',
      learning_rate=2e-4,
      max_epochs=20,
      batch_size=64,
      num_theta=5000,
      dfreq=1,
      diter=1,
      model_name='test_experiment'):

    np.random.seed(np.random.randint(0,10))
    tf.set_random_seed(np.random.randint(0,10))

    self.image_width = 64
    self.num_channels = 3
    self.image_size = self.num_channels*(self.image_width**2)
    
    self.gan_type = gan_type

    # For input noise
    self.latent_dim = 100

    self.num_theta = num_theta

    self.batch_size = batch_size
    self.is_training = True
    self.max_epochs = max_epochs
    self.learning_rate = learning_rate

    self.diter = diter
    self.dfreq = dfreq

    self.generator_model = generator_model
    self.discriminator_model = discriminator_model

    self.base_dir = 'results/' + model_name
    import os
    import errno    

    try:
      os.makedirs(self.base_dir)
    except OSError as exc:  # Python >2.5
      if exc.errno == errno.EEXIST and os.path.isdir(self.base_dir):
        pass

    self.build_model()

    print("Loading data into memory.")
    self.data = self.read_data()
    self.max_examples = self.data.shape[0]
    print("Loaded {} examples".format(self.max_examples))
    return  


  """
  Load the data
    Assumes the data is in a single numpy array. Loads it into memory.
  """
  def read_data(self):
    path = '/home/ideshpa2/data/lsun_64x64.npy'
    im = np.load(path)
    return im

  """
  Sliced-Wasserstein loss
    Projects the images onto randomly chosen directions and computes the Wasserstein distance
    between the two empirical distributions. The loss is the sum of the distances along all
    such projections.
    @params t_ Samples from the true distribution
    @params f_ Samples from the generator
    @params num_theta Number of random directions to project images onto 
    @params reference If true, use a fixed set of directions. This will be used for comparison
  """
  def sw_loss(self, t, f, reference=False):
    s = t.get_shape().as_list()[-1]

    theta = tf.random_normal(shape=[s, self.num_theta])
    normal_theta = tf.nn.l2_normalize(theta, dim=0)
    
    x_t = tf.transpose(tf.matmul(t, normal_theta))
    x_f = tf.transpose(tf.matmul(f, normal_theta))

    sorted_true,_ = tf.nn.top_k(x_t, self.batch_size)
    sorted_fake, fake_indices = tf.nn.top_k(x_f, self.batch_size)
 
    flat_true = tf.reshape(sorted_true,[-1])
    rows = np.asarray([self.batch_size*int(np.floor(i*1.0/self.batch_size)) for i in range(self.num_theta*self.batch_size)])
    flat_idx = tf.reshape(fake_indices,[-1,1]) + np.reshape(rows,[-1,1])

    shape = tf.constant([self.batch_size*self.num_theta])
    rearranged_true = tf.reshape(tf.scatter_nd(flat_idx, flat_true, shape), [self.num_theta, self.batch_size])
 
    return tf.reduce_mean(tf.square(x_f - rearranged_true))

  """    
  Creates the computation graph
  """
  def build_model(self):

    # Input images from the TRUE distribution
    self.x = tf.placeholder(tf.float32,[None, self.image_size])

    # Latent variable
    self.z = tf.placeholder(tf.float32,[None, self.latent_dim])

    generator = eval(self.generator_model).generator
    # Output images from the GAN
    self.x_hat = generator(self.z)

    generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

    # Different GAN variants have different losses for the generator
    discriminator = eval(self.discriminator_model).discriminator
    self.y, self.y_to_match = discriminator(self.x)
    self.y_hat, self.y_hat_to_match = discriminator(self.x_hat, reuse=True)

    if self.gan_type == "swg":
      self.generator_loss = self.sw_loss(self.y_to_match, self.y_hat_to_match)
      true_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(self.y),
        logits=self.y)
      fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(self.y_hat),
        logits=self.y_hat)  
      self.discriminator_loss = tf.reduce_mean(true_loss + fake_loss) 

    else: # gan == "wgan"
      epsilon = tf.random_uniform([], 0.0, 1.0)
      x_sample = epsilon * self.x + (1 - epsilon) * self.x_hat
      d_hat = discriminator(x_sample,reuse=True)

      ddx = tf.gradients(d_hat, x_sample)[0]
      print("Gradients \n \n \n \n \n", ddx.get_shape().as_list())
      ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
      scale = 10
      ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)

      self.discriminator_loss = -tf.reduce_mean(self.y_hat) + tf.reduce_mean(self.y) + ddx
      self.generator_loss = tf.reduce_mean(self.y_hat)

    tf.summary.scalar("discriminator_loss", self.discriminator_loss)
    tf.summary.scalar("generator_loss", self.generator_loss)      

    # We also track the sliced Wasserstein distance between the generated images and fake images
    self.sliced_wasserstein_distance = self.sw_loss(self.x, self.x_hat)
    tf.summary.scalar("sliced_wasserstein_distance", self.sliced_wasserstein_distance)
      
    # Discriminator Optimizer
    discriminator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
    self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(
      self.discriminator_loss,
      var_list = discriminator_vars)

    self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(
      self.generator_loss,
      var_list = generator_vars)

    self.merged_summary_op = tf.summary.merge_all()
    return

  """
  Main training loop. Saves a checkpoint and sample images after every epoch.
  """     
  def train(self):
    config=tf.ConfigProto()
    config.gpu_options.allow_growth = False
    sess = tf.Session(config=config) 
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    summary_writer = tf.summary.FileWriter(self.base_dir,sess.graph)

    # Can be used later for computing, e.g., the KL divergence using the ITE toolbox.
    data_from_run = dict()
    collected_samples = []

    curr_time = time.time()
    print("Starting code " + self.gan_type)
    for epoch in range(self.max_epochs):  
      for iteration in range(int(self.max_examples/self.batch_size)):
 
        x = self.data[iteration*self.batch_size: (iteration+1)*self.batch_size]
        z = np.random.uniform(-1,1,size=[self.batch_size, self.latent_dim])
        sess.run(self.g_optimizer, feed_dict={self.x: x, self.z: z})       

        if iteration % self.dfreq == 0:
          for diter in range(self.diter):
            sess.run(self.d_optimizer, feed_dict={self.x: x, self.z: z})

        if (iteration)%50 == 0:
          l = sess.run(self.generator_loss  , feed_dict={self.x: x, self.z: z})
          print(
           "Epoch {}, Time elapsed: {}, Loss at iteration {}: {}".format(
            epoch,time.time()-curr_time,iteration, l))
          curr_time = time.time()

      im = self.get_random_samples(sess, num_samples = 36)
      im = np.reshape(im,(-1, self.image_width, self.num_channels))
      im = np.hstack(np.split(im,6))

      # I made an error while creating the numpy array for LSUN, which swapped B and R
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

      plt.imshow(im)
      plt.axis('off')
      fig = plt.gcf()
      fig.set_size_inches(12, 12)
      plt.savefig(self.base_dir+'/Epoch_{}.png'.format(epoch), bbox_inches='tight')
      plt.close()

      summ  = sess.run(self.merged_summary_op, feed_dict={self.x: x, self.z: z})
      summary_writer.add_summary(summ, iteration)
      saver.save(sess,self.base_dir+'/checkpoint.ckpt')

      z = np.random.uniform(-1,1,size=[500, self.latent_dim])
      x = self.data[np.ranomd.randint(0,self.max_examples,500)]
      samples_iter, projected_true, projected_fake = sess.run(
        [self.x_hat, self.y_to_match, self.y_hat_to_match], feed_dict={self.x: x, self.z: z})
      collected_samples.append(np.asarray(samples_iter))
      collected_projected_true.append(np.asarray(projected_true))
      collected_projected_fake.append(np.asarray(projected_fake))      

    data_from_run["reference"] = np.asarray(x)
    data_from_run["samples"] = np.asarray(collected_samples)
    data_from_run["projected_true"] = np.asarray(collected_projected_true)
    data_from_run["projected_fake"] = np.asarray(collected_projected_fake)     

    import pickle

    with open(self.base_dir+'/data_from_run.pickle', 'wb') as handle:
      pickle.dump(data_from_run, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


  """
  Generates a set of random images from the generator 
    @params sess Tensorflow session
    @params num_samples Number of examples to generate
  """
  def get_random_samples(self, sess, num_samples=100):
    code = np.random.uniform(-1,1,size=[num_samples,self.latent_dim])
    result = sess.run([(self.x_hat)], feed_dict={self.z: code})
    return result[0]

  """
  Method to generate samples using a pre-trained model 
  """
  def generate_images(self):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(self.base_dir + '/'))

    im = self.get_random_samples(sess)

    im = np.reshape(im[:36],(-1, self.image_width,self.num_channels))
    im = np.hstack(np.split(im,6))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    plt.savefig(self.base_dir+ '/Samples.png', bbox_inches='tight')
    plt.close()
    return
