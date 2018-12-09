import librosa
import numpy as np
import os
import pyworld

def world_decode_spectral_envelop(coded_sp, fs):
    # Decode Mel-cepstral to sp
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)
    return decoded_sp

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

sampling_rate, frame_period=22050, 5
aps = np.load("data/generate/test_ap_2.npy")
f0s = np.load("data/generate/test_f0_2.npy")
coded_sps = np.load("data/predict_data.npy")

for i in range(len(aps)):
    f0 = f0s[i]
    ap = aps[i]
    lenth = ap.shape[0]
    coded_sp = coded_sps[i][0][:lenth].astype("double")
    wav_transformed = world_speech_synthesis(f0=f0, coded_sp=coded_sp, 
                                                    ap=ap, fs=sampling_rate, frame_period=frame_period)
    librosa.output.write_wav("data/generate/"+str(i) + ".wav", wav_transformed, sampling_rate)