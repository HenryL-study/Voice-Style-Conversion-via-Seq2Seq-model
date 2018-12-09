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

wav_file = "data/VCC2SF1/10001.wav"
sampling_rate, num_mcep, frame_period=22050, 40, 5

wav, _ = librosa.load(wav_file, sr=sampling_rate, mono=True)
f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
# f0_converted = pitch_conversion(f0=f0, 
#                 mean_log_src=test_loader.logf0s_mean_src, std_log_src=test_loader.logf0s_std_src, 
#                 mean_log_target=test_loader.logf0s_mean_trg, std_log_target=test_loader.logf0s_std_trg)
coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
print("This is the feature we want -> coded_sp")
print("Type of coded_sp: ", type(coded_sp))
print("shape of coded_sp: ", coded_sp.shape)
wav_transformed = world_speech_synthesis(f0=f0, coded_sp=coded_sp, 
                                                    ap=ap, fs=sampling_rate, frame_period=frame_period)
librosa.output.write_wav("data/generate/10001.wav", wav_transformed, sampling_rate)
                                