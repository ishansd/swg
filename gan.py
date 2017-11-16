from __future__ import print_function

import tensorflow as tf

from tensorflow.python.client import timeline

import numpy as np
import scipy.stats as ss

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import tensorflow.contrib.layers as layers

from generator_models import *
from discriminator_models import *
import time

'''
  Base class for GAN
  The GAN can be one of the following:
  1. Sliced Wasserstein GAN (swgan)
  2. Original GAN (ogan)
  3. GAN with -log D trick (dgan)
  4. Wasserstein GAN (wgan)
'''
class gan():
  def __init__(
      self, 
      gan_type='swgan',
      generator_model='fc',
      discriminator_model='d_fc',
      learning_rate=2e-4,
      max_epochs=10,
      batch_size=64,
      num_theta=5000,
      dfreq=1,
      diter=1,
      model_name='default_name'):

    np.random.seed(np.random.randint(0,10))
    tf.set_random_seed(np.random.randint(0,10))

    self.image_width = 64
    self.num_channels = 3
    self.image_size = self.num_channels*(self.image_width**2)
    
    self.gan_type = gan_type

    self.latent_dim = 100

    self.num_theta = num_theta

    self.batch_size = batch_size
    self.is_training = True
    self.max_epochs = max_epochs
    self.learning_rate = learning_rate

    self.diter = diter
    self.dfreq = dfreq


    if model_name == 'default_name':
      model_name = gan_type + '_g_' + generator_model + '_d_' + discriminator_model
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

    print("Reading data for fixed")
    self.data = self.read_data()
    self.max_examples = self.data.shape[0]
    print("Read data")


    return  

  '''
    Read dataset
  '''
  def read_data(self):
#    path = '/home/ideshpa2/data/celeba/celeba.npy'
    path = '/home/ideshpa2/data/lsun_64x64.npy'
    print(path)
    im = np.load(path)
    return im

  '''
    Sliced-Wasserstein loss
      Projects the images onto randomly chosen directions and computes the Wasserstein distance
      between the two empirical distributions. The loss is the sum of the distances along all
      such projections.
      @params t_ Samples from the true distribution
      @params f_ Samples from the generator
      @params num_theta Number of random directions to project images onto 
      @params reference If true, use a fixed set of directions. This will be used for comparison
  '''
  def sw_loss(self, t, f, num_theta=20000, reference=False):
    # theta = np.random.randn(self.image_size, num_theta).astype(np.float32)
    # normal_theta = theta / np.sqrt(np.sum(np.square(theta), axis=0, keepdims=True))

    s = t.get_shape().as_list()[-1]
    print(s)

    theta = tf.random_normal(shape=[s, num_theta])
    normal_theta = tf.nn.l2_normalize(theta, dim=0)
    
    x_t = tf.transpose(tf.matmul(t, normal_theta))
    x_f = tf.transpose(tf.matmul(f, normal_theta))

    sorted_true,_ = tf.nn.top_k(x_t, self.batch_size)
    sorted_fake, fake_indices = tf.nn.top_k(x_f, self.batch_size)
 
    flat_true = tf.reshape(sorted_true,[-1])
    rows = np.asarray([self.batch_size*int(np.floor(i*1.0/self.batch_size)) for i in range(num_theta*self.batch_size)])
    flat_idx = tf.reshape(fake_indices,[-1,1]) + np.reshape(rows,[-1,1])

    shape = tf.constant([self.batch_size*num_theta])
    rearranged_true = tf.reshape(tf.scatter_nd(flat_idx, flat_true, shape), [num_theta, self.batch_size])
 
    return tf.reduce_mean(tf.square(x_f - rearranged_true))

  '''
    Creates the computation graph
  '''
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

    true_loss = tf.nn.sigmoid_cross_entropy_with_logits( labels=tf.ones_like(self.y),
                              logits=self.y)
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits( labels=tf.zeros_like(self.y_hat),
                              logits=self.y_hat)  
    self.discriminator_loss = tf.reduce_mean(true_loss + fake_loss) 

    self.generator_loss = self.sw_loss(self.y_to_match, self.y_hat_to_match, num_theta=self.num_theta)

    tf.summary.scalar("discriminator_loss", self.discriminator_loss)
      
    # Discriminator Optimizer
    discriminator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
    self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(
      self.discriminator_loss,
      var_list = discriminator_vars)
    self.discriminator_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in discriminator_vars]

    tf.summary.scalar("generator_loss", self.generator_loss)      
    self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(
      self.generator_loss,
      var_list = generator_vars)


    self.grad = tf.gradients(self.generator_loss, self.z)

    self.merged_summary_op = tf.summary.merge_all()
    return

  '''
    Train the model using the optimizer created above. The number of iterations and the learning
    rate is passed to the constructor. It periodically saves a checkpoint of the model, as well as
    samples at that stage.
  '''
  def train(self):

    config=tf.ConfigProto()
    config.gpu_options.allow_growth = False
    sess = tf.Session(config=config) 
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    summary_writer = tf.summary.FileWriter(self.base_dir,sess.graph)

    data_from_run = dict()
    collected_samples = []

    curr_time = time.time()
    print("Starting 1 code " + self.gan_type)
    for epoch in range(self.max_epochs):  
      for iteration in range(int(self.max_examples/self.batch_size)):
 
        x = self.data[iteration*self.batch_size: (iteration+1)*self.batch_size]
        z = np.random.uniform(-1,1,size=[self.batch_size, self.latent_dim])
        sess.run(self.g_optimizer, feed_dict={self.x: x, self.z: z})       

        if iteration % self.dfreq == 0:
          for diter in range(self.diter):
            sess.run(self.d_optimizer, feed_dict={self.x: x, self.z: z})

        # if (iteration+1)%50 == 0:
        #   l,g = sess.run([self.generator_loss,self.grad]  , feed_dict={self.x: x, self.z: z})
        #   g = np.mean(np.sum(np.square(g)))
        #   print(
        #    "Epoch {}, Time elapsed: {}, Loss at iteration {}: {}, Gradient: {}".format(epoch,time.time()-curr_time,iteration+1, l,g))
        #   curr_time = time.time()

      im = self.get_random_samples(sess, num_samples = 64)
      im = np.reshape(im,(-1, self.image_width,self.num_channels))
      im = np.hstack(np.split(im,8))
      plt.imshow(im)
      plt.axis('off')
      fig = plt.gcf()
      fig.set_size_inches(12, 12)
      plt.savefig(self.base_dir+'/Epoch_{}.png'.format(epoch), bbox_inches='tight')
      plt.close()
      saver.save(sess,self.base_dir+'/checkpoint.ckpt')

      z = np.random.uniform(-1,1,size=[500, self.latent_dim])
      samples_iter = sess.run(self.x_hat, feed_dict={self.z: z})
      collected_samples.append(np.asarray(samples_iter))

    data_from_run["reference"] = np.asarray(x)
    data_from_run["samples"] = np.asarray(collected_samples)

    import pickle

    with open(self.base_dir+'/data_from_run.pickle', 'wb') as handle:
      pickle.dump(data_from_run, handle, protocol=pickle.HIGHEST_PROTOCOL)


    return


  '''
    Generates a set of random images from the generator 
    @params sess a Tensorflow session
  '''
  def get_random_samples(self, sess, num_samples=100):

    code = np.random.uniform(-1,1,size=[num_samples,self.latent_dim])
    result = sess.run([(self.x_hat)], feed_dict={self.z: code})

    return result[0]


  '''
  '''
  def generate_images(self):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(self.base_dir + '/'))

    im = self.get_random_samples(sess)

    im = np.reshape(im[:64],(-1, self.image_width,self.num_channels))
    im = np.hstack(np.split(im,8))
    plt.imshow(im)
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    plt.savefig(self.base_dir+ '/Final.png', bbox_inches='tight')
    plt.close()
    return
