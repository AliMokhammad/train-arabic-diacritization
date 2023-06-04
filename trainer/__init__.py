import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import json

import ..configs as training_configs

class Trainer():
    def __init__(self,generator, discriminator, train_dataset):
        self.generator = generator
        self.discriminator = discriminator
        self.train_dataset = train_dataset
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        self.discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        self.disc_loss_list = []
        self.gen_loss_list = []

    def save_loss_list(self, loss_list, filename):
        with open(filename, 'w') as file:
            json.dump(loss_list, file)

    def save_discriminator(self):
        self.discriminator.save(training_configs.discriminator_model_model_path)
        self.save_loss_list(
            loss_list={"disc_loss_list": self.disc_loss_list},
            filename=training_configs.discriminator_loss_path
        )

    def save_generator(self):
        self.generator.save(training_configs.ganarator_model_path)
        self.save_loss_list(
            loss_list={"gen_loss_list": self.gen_loss_list},
            filename=training_configs.ganarator_loss_path
        )
    
    def start_training(self, epochs, lambda_val, noise_dim = 100):
        pbar = tqdm(total=epochs, desc="Training Progress")

        for epoch in range(epochs):
            for real_data, _ in self.train_dataset:
                batch_size = real_data.shape[0]

                # Generate fake data
                noise = tf.random.normal([batch_size, noise_dim])
                fake_data = self.generator(noise, training=True)

                # Train the discriminator
                with tf.GradientTape() as disc_tape:
                    real_output = self.discriminator(real_data, training=True)
                    fake_output = self.discriminator(fake_data, training=True)

                    disc_loss = self.loss_fn(tf.ones_like(real_output), real_output) +  self.loss_fn(tf.zeros_like(fake_output), fake_output)

                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
                self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

                # Train the generator
                with tf.GradientTape() as gen_tape:
                    fake_output = self.discriminator(fake_data, training=True)

                    adverserial_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
                    mle_loss = tf.reduce_mean(tf.abs(fake_data - real_data))

                    gen_loss = lambda_val * adverserial_loss + (1 - lambda_val) * mle_loss

                gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))


            # Collect loss values for plots
            self.disc_loss_list.append(disc_loss)
            self.gen_loss_list.append(gen_loss)

            # Update progress bar
            pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            pbar.update(1)

        pbar.close()
        
        self.save_discriminator()
        self.save_generator()


