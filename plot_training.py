import json
import matplotlib.pyplot as plt
import configs as training_configs

def print_training_progress(epochs, generator_losses, discriminator_losses):
    plt.plot(epochs, generator_losses, label='Generator Loss')
    plt.plot(epochs, discriminator_losses, label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig('training_progress.png')  # Save the plot as an image
    plt.show()

ganarator_loss_path = training_configs.ganarator_loss_path
discriminator_loss_path = training_configs.discriminator_loss_path

with open(ganarator_loss_path, 'r') as file:
    ganarator_loss = json.load(file).get('gen_loss_list')

with open(discriminator_loss_path, 'r') as file:
    discriminator_loss = json.load(file).get('disc_loss_list')

epochs = list(range(len(ganarator_loss)))

print_training_progress(epochs, ganarator_loss, discriminator_loss)