#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Functions taht create bat and non-bat like sounds 
to train a neural network.
Created on Wed Sep 19 13:25:55 2018

@author: tbeleyur
"""
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
import numpy as np 
import scipy.signal as signal 
import scipy.io.wavfile as WAV


def generate_a_samplesound(duration,fs=192000, shape='linear',
                          window='tukey', freqs=[90000, 20000], SNR=15,
                          **kwargs):
    '''
    Generates a 10millisecond audio file with the required sound in the middle
    placed at a random position. These 10ms audio files are intended to train 
    a neural network to detect bat/bat-like calls.
    
    Parameters:
        
        duration: float>0. duration of the batlike call in seconds. 
        
        fs : int >0. sampling rate of the generated signal 

        shape: string. Type of fm chirp to be generated. See the entries
                possible in the 'method' argument of signal.chirp. Defaults
                to 'linear'
        
        window : string. The kind of envelope window to be added to the 
                signal. Only a Tukey window is currently implemented
        
        freqs : 1x2 array-like. Two entries with the starting and end frequency. 
                in Hz. 
        
        SNR : float. Signal to noise ratio in dB rms
        
        Keyword arguments:
            
            background_noise : float. background noise level in decibels. 
                               Defaults to -80 dB rms of white noise.  
            fullrec_durn : float. Duration of whole audio clip. Defaults to 
                            10ms if not specified
            
        
    Returns:
        
        samplesound : 1 X Nsamples np.array. The required output signal 
                      of Nsamples length according to the specified duration
                      and sampling rate.
                             
    '''
    if 'fullrec_durn' not in kwargs.keys():
        nsamples_fullrec = int(fs*0.010)        
    else:
        nsamples_fullrec = int(fs*kwargs['fullrec_durn'])
        
    
    nsamples_chirp = int(fs*duration)
    samplesound = np.zeros(nsamples_fullrec)

    if not 'background_noise' in kwargs.keys():
        background_noise = -80
        backg_noise_linear = 10.0**(background_noise/20.0)
    else:
        background_noise = kwargs['background_noise']
        backg_noise_linear = 10.0**(background_noise/20.0)
    
    samplesound += np.random.normal(0,backg_noise_linear, nsamples_fullrec)

    t = np.linspace(0, duration, nsamples_chirp)        
    chirp = signal.chirp(t, freqs[0], t[-1], freqs[1], method=shape)
    chirp *= signal.tukey(nsamples_chirp, 0.9)
    # adjust the dB rms of the     

    chirp_rmsadj = adjust_rms_to_targetSNR(SNR, background_noise, chirp)

    # insert the rms adjusted chirp into the snippet w background noise
    max_allowed_startind = nsamples_fullrec - nsamples_chirp
    start_index = int(np.random.choice(range(max_allowed_startind),1))
   
    samplesound[start_index:start_index+chirp.size] += chirp_rmsadj
    
    return(samplesound) 




def generate_noise_as_samplesound(noise_dBrms, **kwargs):
    '''Generates white noise of 10 ms length at 192000 sampling rate
    If no keyword arguments are given, the function uses its default values.
    
    Parameters:
        
        noise_dBrms : integer. dBrms level of the noise signal to be output.s
    
        Keyword arguments:
   
            nsamples: intger. Number of samples the output noise signal should be. 
                     Defaults to 1920.

    Returns:

        noise_sound : 1 x nsamples np.array.         
    '''
    if not 'nsamples' in kwargs.keys():
        nsamples = 1920
    else : 
        nsamples = kwargs['nsamples']

    rms_linear = 10**(noise_dBrms/20.0)
    noise_sound = np.random.normal(0, rms_linear, nsamples)
    return(noise_sound)
    

def timescramble_spectrogram_array():
    ''' TO BE COMPLETED!! 
    Keeps overall frequency content the same but scrambles their components
    in time. 
    
    Given an output matrix with from a spectrogram calculating 
    function, where rows are the frequency bands and columns are the 
    FFT window outputs. 
    
    Parameter:
        specgram_array : Nbands x Nwindows np.array. 
    
    Return:
        timescrambled_specgram : Nbands x Nwindows np.array. 
    '''
    
    pass



    
def rms(input_signal):
    '''UNTESTED
    '''
    signal_sq = input_signal**2.0
    mean_sqsignal = np.mean(signal_sq)
    root_meansq = mean_sqsignal**0.5
    return(root_meansq)
    
def adjust_rms_to_targetSNR(targetSNR_dBrms, bgnoise_dBrms, signal):
    '''
    UNTESTED
    '''
    rawsignal_dBrms = 20*np.log10(rms(signal))
    current_SNR = rawsignal_dBrms - bgnoise_dBrms    
    required_dBrms_adjustment = targetSNR_dBrms - current_SNR

    signal_dBrmscorrected = 10**(required_dBrms_adjustment/20.0) * signal

    return(signal_dBrmscorrected)


mean_subtract = lambda X : X - np.mean(X)

def mean_subtract_rowwise(input_array):
    row_meansubt = np.apply_along_axis(mean_subtract, 1, input_array)
    return(row_meansubt)



if __name__ == '__main__':
    y = generate_a_samplesound(0.001,fs=192000, freqs=[30000, 50000],
                               background_noise=-10, SNR=-3, fullrec_durn=0.005)
    
    z = generate_noise_as_samplesound(-15)
    plt.figure(figsize=(8,6))
    s,f,t,im = plt.specgram(y, NFFT=64, noverlap=14, Fs=192000, cmap='Greys');
    plt.imshow(20*np.log10(mean_subtract_rowwise(s)), cmap='Greys')
    