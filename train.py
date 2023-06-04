from .generator import Generator
from .discriminator import Discriminator
from .trainer import Trainer
from .dataset import Dataset


generator = Generator()
discriminator = Discriminator()
dataset = Dataset()
trainer = Trainer(generator=generator, discriminator=discriminator, train_dataset=dataset)

# Set hyperparameters
epochs = 100 # 200
lambda_val = 0.5

trainer.start_training(epochs=epochs, lambda_val=lambda_val)
Print("Training is Done ...")

