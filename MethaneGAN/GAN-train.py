import numpy as np 
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

import imageio

import os

batch_size = 32
num_channels = 3 # actually there are three channels in this image
num_classes = 1
image_size = (240,320)
latent_dim = 128


frame_data_dir = '/home/bestlab/Desktop/Squishy-Methane-Analysis/MethaneModel/frame_data_movingAvg/'
frame_train_data_dir = os.path.join(frame_data_dir, 'train')
frame_test_data_dir = os.path.join(frame_data_dir, 'test')
frame_train_nonleak_data_dir = os.path.join(frame_train_data_dir,'Nonleaks')
frame_train_leak_data_dir = os.path.join(frame_train_data_dir,'Leak')


nonleak_frame_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=frame_train_nonleak_data_dir,
    labels=None,
    batch_size=32,
    image_size=(240,320),
    shuffle=True,
    seed=42,
)
leak_frame_dataset = tf.keras.utils.image_dataset_from_directory(
    directory = frame_train_leak_data_dir,
    labels=None,
    batch_size=32,
    image_size=(240,320),
    shuffle=True,
    seed=42,
)

n_nonleak = len(nonleak_frame_dataset)
print(n_nonleak)
leak_frame_dataset_downsampled = leak_frame_dataset.take(n_nonleak)
print(len(leak_frame_dataset_downsampled))

all_digits = nonleak_frame_dataset.concatenate(leak_frame_dataset_downsampled)
def generate_label(label,batch_size,dataset_size):
    temp = np.array([[label]*batch_size]*dataset_size)
    temp = temp.reshape((-1,32,1))
    return temp.astype('float32')

label_nonleak = tf.data.Dataset.from_tensor_slices(generate_label(0,32,n_nonleak))
label_leak = tf.data.Dataset.from_tensor_slices(generate_label(1,32,n_nonleak))
all_labels = label_nonleak.concatenate(label_leak)

dataset = tf.data.Dataset.zip((all_digits,all_labels))

generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)


# Create the discriminator.
discriminator = keras.Sequential(
    [
        keras.layers.InputLayer((240,320, discriminator_in_channels)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)

# Create the generator.
generator = keras.Sequential(
    [
        keras.layers.InputLayer((generator_in_channels,)),
        # We want to generate 128 + num_classes coefficients to reshape into a
        # 7x7x(128 + num_classes) map.
        # layers.Dense(7 * 7 * generator_in_channels),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Reshape((7, 7, generator_in_channels)),
        # layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
        layers.Dense(15 * 20 * generator_in_channels),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((15, 20, generator_in_channels)),
        
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Conv2D(3, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)

class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        
    def get_config(self):
            return {
                "discriminator_config": self.discriminator.get_config(),
                "generator_config": self.generator.get_config(),
                "latent_dim": self.latent_dim,
            }

    @classmethod
    def from_config(cls, config,discriminator,generator):
        discriminator = discriminator.from_config(config["discriminator_config"])
        generator = generator.from_config(config["generator_config"])
        return cls(discriminator=discriminator, generator=generator, latent_dim=config["latent_dim"])


    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def call(self, data):
        x = self.forward(data)
        return x
    
    def forward(self,data):
        # Unpack the data.
        real_images, one_hot_labels = data
        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[240 * 320]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, 240, 320, num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)

        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        return combined_images,image_one_hot_labels
    
    def train_step(self, data):

        real_images, one_hot_labels = data
        batch_size = tf.shape(real_images)[0]

        combined_images, image_one_hot_labels = self.forward(data)
        
        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }
    
cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=latent_dim
)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

for real_images, one_hot_labels in dataset.take(1):
    cond_gan((real_images,one_hot_labels))

cond_gan.fit(dataset, epochs=20)
cond_gan.save('./models/oct12-gan/')
