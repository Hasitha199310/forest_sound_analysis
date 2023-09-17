import tensorflow as tf
import scipy
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.fft import fftshift
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import models
import os
import json

YEAR = '2023'
MONTH = '08'
DAY = '23'

DATE = YEAR + MONTH + DAY
#DATAPATH = str.join('test',DATE,'/')

LABELS =['chainsaw', 'chirping_birds', 'crackling_fire', 'engine','footsteps']
outputs = list()

def get_spectrogram(waveform):
  # Getting the spectogram
  spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def classify(labels,filepath):
    x = filepath
    x = tf.io.read_file(str(x))
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,) 
    x = tf.squeeze(x, axis=-1) 
    waveform = x
    x = get_spectrogram(x)
    x = x[tf.newaxis,...]
    prediction = model(x)
    prediction_max = np.argmax(prediction[0]) 
    return labels[prediction_max]

def run():
    

    model = tf.keras.models.load_model('forest_cnn_model.keras')
    test_file_path = 'test/'+ DATE

    for file in os.listdir(test_file_path):
        #run prediction
        filepath = test_file_path + '/' +str(file)
        output = classify(labels=LABELS,filepath=filepath)
        outputs.append(output) 
    return outputs
    
if __name__ == '__main__':
    outputs = run()
    dict1 = {"saw":0, "tresspassing":0}

    for element in outputs:
        if element == "chainsaw":
            dict1["saw"] = dict1["saw"]+1
        elif element == "footsteps":
            dict1["tresspassing"] = dict1["tresspassing"]+1
        else:
            pass
    
    # the json file where the output must be stored
    out_file = open("static/alert.json", "w")
    json.dump(dict1, out_file, indent = 6)
    out_file.close()

    import json
    with open('data.json', 'w', encoding='utf-8') as f:
        print("running")
        json.dump(dict1, f, ensure_ascii=False, indent=4)
    