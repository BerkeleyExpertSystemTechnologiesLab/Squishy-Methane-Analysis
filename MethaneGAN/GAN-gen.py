import numpy as np 
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

import imageio

import os
import sys 
from tqdm import tqdm

path = './models/oct12-gan/'
cond_gan = tf.keras.models.load_model(path)


number_nonleak = 42881
number_leak = 304911
number_diff = number_leak-number_nonleak
number_gen = number_diff

def generate_noise_labels(number_generation:int):
    latent_dim = 128
    interpolation_noise = tf.random.normal(shape=(1, latent_dim))
    interpolation_noise = tf.repeat(interpolation_noise, repeats=number_generation)
    interpolation_noise = tf.reshape(interpolation_noise, (number_generation, latent_dim))

    labels = tf.zeros((number_generation,1))
    noise_and_labels = tf.concat([interpolation_noise, (labels)], 1)
    return noise_and_labels

def generate_image(trained_gen,noise_and_labels):
    fake = trained_gen.predict(noise_and_labels)    
    return fake

if __name__ == '__main__':

    trained_gen = cond_gan.generator
    print('generating ',number_gen, ' images')
    noise_and_labels = generate_noise_labels(1)
    os.makedirs('generated_images', exist_ok=True)

    for i in tqdm(range(number_gen)):

        fake_image = generate_image(trained_gen,noise_and_labels)

        fake_image *= 255.0
        converted_images = fake_image.astype(np.uint8)
        
        imageio.imsave(f'generated_images/fake_nonleak_image_{i}.png', converted_images[0])
