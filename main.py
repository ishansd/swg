from __future__ import print_function

import numpy as np

from gan import gan
import tensorflow as tf

import argparse

  
def main(argv=None):
  parser = argparse.ArgumentParser(description='SWGAN')

  parser.add_argument(
    'generator', metavar='generator_file_name', default='dcgan',help='Generator model to use')
  parser.add_argument(
    'discriminator', metavar='discriminator_file_name',default='d_dcgan', help='Discriminator model to use')

  parser.add_argument(
    'gan_type', metavar='gan_type', help='GAN to use: SWGAN, OGAN, DGAN, WGAN', default='dgan')  
  parser.add_argument(
    'name', metavar='output_folder_name', help='Output folder to use')

  parser.add_argument('--train', dest='train', action='store_true', help='Use to train')
  parser.add_argument('--images', dest='images', action='store_true', help='Generate Images')

  parser.add_argument(
    '--d_freq', metavar='disc freq',default=1, help='Frequency of training discriminator')
  parser.add_argument(
    '--d_iter', metavar='disc rep',default=1, help='Number of training iterations for discriminator')

  parser.add_argument(
    '--learning_rate', metavar='learning rate',default=2e-4, help='Learning rate for optimizer')
  parser.add_argument(
    '--max_epochs', metavar='max epochs', default=10, help='Number of epochs to train')
  parser.add_argument(
    '--num_theta', metavar='num projections', default=5000, help='Number of projections to use at every step')

  parser.add_argument(
    '--batch_size', metavar='batch size', default=64, help='Batch size')

  parser.add_argument(
    '--num_runs', metavar='number of runs', default=1, help='Number of different runs')

  args = parser.parse_args()
  print(args.learning_rate)


  if int(args.num_runs) == 1:
    tf.reset_default_graph()
    g = gan(
      generator_model=args.generator,
      discriminator_model=args.discriminator,
      model_name=args.name, 
      gan_type=args.gan_type,
      learning_rate=float(args.learning_rate), 
      max_epochs=int(args.max_epochs),
      batch_size=int(args.batch_size),
      num_theta=int(args.num_theta),
      diter=int(args.d_iter),
      dfreq=int(args.d_freq)
      )

    if args.train:
      g.train()
      g.generate_images()
    elif args.images:
      g.generate_images()

  else:
    for run in range(1,int(args.num_runs)+1): 
      tf.reset_default_graph()
      g = gan(
        generator_model=args.generator,
        discriminator_model=args.discriminator,
        model_name=args.name+'_{}'.format(run), 
        gan_type=args.gan_type,
        learning_rate=float(args.learning_rate), 
        max_epochs=int(args.max_epochs),
        batch_size=int(args.batch_size),
        num_theta=int(args.num_theta),
        diter=int(args.d_iter),
        dfreq=int(args.d_freq)
        )
      if args.train:
        g.train()
        g.generate_images()      
      if args.images:
        g.generate_images()      

  return

if __name__ == '__main__':
  tf.app.run
  main()
