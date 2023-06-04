import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import configs as training_configs
import utils

generator = load_model(training_configs.ganarator_model_path)
test_file_path = training_configs.test_file_path

with open(test_file_path, 'r', encoding='utf-8') as file:
    sentences = file.readlines()

#### Start tests ####
test_results = []
i = 1
for sentence in sentences:
    start_time = time.time()
    real_diacritized, non_diacritized = sentence.strip().split("|")

    preprocessed_sentence = utils.preprocess_sentence(non_diacritized)

    gen_diacritized = generator(preprocessed_sentence, training=False)

    iteration_time = (time.time() - start_time) * 1000

    DER, WER = utils.calculate_accurecy(gen_diacritized, real_diacritized)

    test_results.append((i, gen_diacritized, DER, WER, iteration_time))
    i += 1
#### End tests ####


#### Start calculate speed ####
total_time = 0
for test in test_results:
    ii = test[0]
    txt = test[1]
    time = test[4]
    print(f"({ii}) {txt} | Time (ms): {time}" )

average_time = total_time / len(test_results)
print(f"Avg Time (ms): {average_time}" )
#### End calculate speed ####

#### Start calculate accuracy ####
total_DER = 0
total_WER = 0
for test in test_results:
    ii = test[0]
    txt = test[1]
    DER = test[2]
    WER = test[3]
    print(f"({ii}) {txt} | DER: {DER} | WER: {WER}" )

average_DER = total_DER / len(test_results)
average_WER = total_WER / len(test_results)
print(f"Avg DER: {average_DER} | Avg WER: {average_WER}" )
#### End calculate accuracy ####