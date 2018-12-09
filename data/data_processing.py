import librosa
import numpy as np
import os
import pyworld
def world_encode_spectral_envelop(sp, fs, dim=36):
    # Get Mel-cepstral coefficients (MCEPs)
    #sp = sp.astype(np.float64)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)
    return coded_sp

def world_decompose(wav, fs, frame_period = 5.0):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    return f0, timeaxis, sp, ap

def world_speech_synthesis(f0, coded_sp, ap, fs, frame_period):
    decoded_sp = world_decode_spectral_envelop(coded_sp, fs)
    # TODO
    min_len = min([len(f0), len(coded_sp), len(ap)])
    f0 = f0[:min_len]
    coded_sp = coded_sp[:min_len]
    ap = ap[:min_len]
    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    # Librosa could not save wav if not doing so
    wav = wav.astype(np.float32)
    return wav

def world_decode_spectral_envelop(coded_sp, fs):
    # Decode Mel-cepstral to sp
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)
    return decoded_sp

path1 = "data/VCC2SF1/"
path2 = "data/VCC2SM1/"
files_1 = os.listdir(path1)
files_2 = os.listdir(path2)
#wav_file = "VCC2SF1/10001.wav"
sampling_rate, num_mcep, frame_period=22050, 40, 5
length = len(files_1)
for i in range(length): 
    print("Iteration ", i)
    wav, _ = librosa.load(path1+files_1[i], sr=sampling_rate, mono=True)
    f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
    coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
    # print("This is the feature we want -> coded_sp")
    # print("Type of coded_sp: ", type(coded_sp))
    # print("shape of coded_sp: ", coded_sp.shape)
    np.save("data/parameters/VCC2SF1/coded_sp/"+str(i)+".npy", coded_sp)
    np.save("data/parameters/VCC2SF1/f0/"+str(i)+".npy", f0)
    np.save("data/parameters/VCC2SF1/ap/"+str(i)+".npy", ap)

    #print("ap: ", ap)
    # wav_transformed = world_speech_synthesis(f0=f0, coded_sp=coded_sp, 
    #                                                 ap=ap, fs=sampling_rate, frame_period=frame_period)
    # librosa.output.write_wav("generate/"+path1+files_1[i], wav_transformed, sampling_rate)

    # sampling_rate, num_mcep, frame_period=22050, 40, 5

    wav, _ = librosa.load(path2+files_2[i], sr=sampling_rate, mono=True)
    f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
    coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
    np.save("data/parameters/VCC2SM1/coded_sp/"+str(i)+".npy", coded_sp)
    np.save("data/parameters/VCC2SM1/f0/"+str(i)+".npy", f0)
    np.save("data/parameters/VCC2SM1/ap/"+str(i)+".npy", ap)
    # wav_transformed = world_speech_synthesis(f0=f0, coded_sp=coded_sp, 
    #                                                 ap=ap, fs=sampling_rate, frame_period=frame_period)
    # librosa.output.write_wav("data/generate/"+path2+files_2[i], wav_transformed, sampling_rate)                           