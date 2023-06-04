import tensorflow as tf
import ..configs as training_configs


class Dataset():
    def __init__(self, data_file = training_configs.dataset_path, batch_size=2):
        self.data_file = data_file
        self.batch_size = batch_size

    def prepare_dataset(self):
        with open(self.data_file, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Preprocess the data
        data = []
        for line in lines:
            diacritized, non_diacritized = line.strip().split("|")
            diacritized = utils.preprocess_sentence(diacritized)
            non_diacritized = utils.preprocess_sentence(non_diacritized)
            data.append((diacritized, non_diacritized))

        # Convert the data to TensorFlow Dataset
        dataset = tf.data.Dataset.from_tensor_slices(data)

        # Shuffle and batch the train dataset
        train_dataset = dataset.shuffle(buffer_size=len(train_dataset)).batch(self.batch_size)

        # Prefetch the datasets for better performance
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        return train_dataset
