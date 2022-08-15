#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import listdir
from os.path import isdir, join
from tensorflow.keras import layers, models, metrics
import numpy as np
import librosa
import matplotlib.pyplot as plt
import IPython
import pyaudio
import python_speech_features
import time
import os

# In[2]:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Function: Create MFCC from given path
def calc_mfcc(signal, fs):

    #Load wavefile
    #signal, fs = librosa.load(path, sr=sample_rate)

    # Create MFCCs from sound clip
    mfccs = python_speech_features.base.mfcc(signal,
                                            samplerate=fs,
                                            winlen=0.256,
                                            winstep=0.05,
                                            numcep=num_mfcc,
                                            nfilt=16,
                                            nfft=2048,
                                            preemph=0.0,
                                            ceplifter=0,
                                            appendEnergy=False,
                                            winfunc=np.hanning)
    return mfccs.transpose()


# In[4]:


stop_words = ['forward', 'nine', 'seven', 'backward', 'cat', 'up', 'eight', 'visual', 'tree', 'happy', 'stop', 'wow', 'off', 'no', 'follow', 'yes', 'bird', 'marvin', 'dog', 'go', 'on', 'sheila', 'two', 'left', 'right', 'down', 'four', 'six', 'bed', 'learn', 'zero', 'house', 'three', 'five', 'one', ]
print(stop_words)
# TEST: Load model and run it against test set

model_filename = '/home/anass/Desktop/Speechgene/model_done.h5'
#model_filename = 'gunshot_detection_model.h5'
model = models.load_model(model_filename)


# In[5]:


# create input buffer from microphone
CHUNKSIZE = 4000 # fixed chunk size
RATE = 8000
num_mfcc = 16
len_mfcc = 16

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

audio_buffer = np.zeros(8000)
print(audio_buffer.size)


while(True):
    # Read chunk and load it into numpy array.
    data = stream.read(CHUNKSIZE)
    current_window = np.frombuffer(data, dtype=np.float32)

    audio_buffer[0:4000] = audio_buffer[4000:8000]
    audio_buffer[4000:8000] = current_window
    #print(audio_buffer.size)

    start_time = time.time()
    # Create MFCCs
    mfccs = calc_mfcc(np.array(audio_buffer), RATE)
    mfccs =  mfccs.reshape(mfccs.shape[0],
                          mfccs.shape[1],
                         1)
    #print(mfccs.shape)
    output = model.predict(np.expand_dims(mfccs, 0))
    for index, stop_word in enumerate(stop_words):
        if output[0][index] > 0.5:
            print(stop_word)
            print("--- %s seconds ---" % (time.time() - start_time))
            #print(output)
            #plt.plot(np.array(audio_buffer))
            #plt.show()


# In[5]:


# close stream
stream.stop_stream()
stream.close()
p.terminate()


# In[ ]:
