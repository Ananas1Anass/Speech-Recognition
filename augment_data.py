import random
import librosa
import numpy as np
import soundfile as sf
import os



def add_white_noise(signal, noise_percentage_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_signal = signal + noise * noise_percentage_factor
    return augmented_signal


def time_stretch(signal, time_stretch_rate):
    return librosa.effects.time_stretch(signal, time_stretch_rate)


def pitch_scale(signal, sr, num_semitones):
    return librosa.effects.pitch_shift(signal, sr, num_semitones)


def random_gain(signal, min_factor=0.1, max_factor=0.12):
    gain_rate = random.uniform(min_factor, max_factor)
    augmented_signal = signal * gain_rate
    return augmented_signal


def invert_polarity(signal):
    return signal * -1


if __name__ == "__main__":
    i=0
    for (dirpath, dirnames, filenames) in os.walk("/home/anass/Desktop/Speechgene/normalized1/Sub8/"):
        print(filenames)
        for filename in filenames:
            i=i+1
            filepath = dirpath + '/' + filename
            print(filepath)
            (path, file_extension) = os.path.splitext(filepath)
            file_extension_final = file_extension.replace('.', '')
            try:
                signal, sr = librosa.load(filepath)
                augmented_signal = invert_polarity(signal)
                random_gain_signal = random_gain(signal)
                pitch_scale_signal = pitch_scale(signal,sr,2)
                add_white_noise_signal_40 = add_white_noise(signal, 0.4)
                add_white_noise_signal_20 = add_white_noise(signal, 0.2)


                #sf.write(filepath + f'{i}' + '.wav' , augmented_signal, sr)
                sf.write(filepath +'random_gain'+ f'{i}' + '.wav' , random_gain_signal, sr)
                sf.write(filepath + 'pitch_scale'+ f'{i}' + '.wav' , pitch_scale_signal, sr)
                sf.write(filepath + 'add_white_noise_40'+ f'{i}' + '.wav' , add_white_noise_signal_40, sr)
                sf.write(filepath + 'add_white_noise_20'+ f'{i}' + '.wav' , add_white_noise_signal_20, sr)

            except:
                print("ERROR AUGMENTING " + str(filepath))    
