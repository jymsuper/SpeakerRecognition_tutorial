from __future__ import print_function

import librosa
import sys
import os
import pandas as pd
import numpy as np
import configure as c
from DB_wav_reader import read_DB_structure

def read_audio(filename, sample_rate=c.SAMPLE_RATE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.flatten()
    speaker_folder = filename.split('/')[-2]
    speaker_label = speaker_folder
    return audio, speaker_label

def calc_wav_length_and_truncate(filename):
    # return the wav length in seconds
    wav, label = read_audio(filename, c.SAMPLE_RATE)
    start_point = 0.5
    start_sample = int(start_point * c.SAMPLE_RATE)
    truncated_wav = wav[start_sample:]
    return truncated_wav

def aggregate_wav_files(DB, n_test_files, spk):
    # Select DB for input speaker
    spk_DB = DB[DB['speaker_id']==spk]
    spk_DB = spk_DB.sort_values(by='filename', ascending=True)
    spk_DB = spk_DB.reset_index(drop=True) # index reset is important!
    
    # shuffle the rows
    # if random_select==True:
        # spk_DB = spk_DB.sample(frac=1)
        # spk_DB = spk_DB.reset_index(drop=True) # index reset is important!
    tot_enroll_num = 0
    tot_test_num = 0
    
    aggregated_enroll_wav = np.empty((0,), np.float32)
    aggregated_test_wav = np.empty((0,), np.float32)
    
    for i in range(0, n_test_files):
        filename = spk_DB['filename'][i]
        truncated_wav = calc_wav_length_and_truncate(filename)
        aggregated_enroll_wav = np.hstack([aggregated_enroll_wav, truncated_wav])
        
    for i in range(n_test_files, n_test_files+n_test_files):
        filename = spk_DB['filename'][i]
        truncated_wav = calc_wav_length_and_truncate(filename)
        aggregated_test_wav = np.hstack([aggregated_test_wav, truncated_wav])
        
    return aggregated_enroll_wav, aggregated_test_wav

def save_aggregated_enroll_wav(aggregated_wav, spk, result_dir):
    filename = 'enroll.wav'
    save_dir = os.path.join(result_dir, spk)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)
    librosa.output.write_wav(save_path, aggregated_wav, c.SAMPLE_RATE)

def save_aggregated_test_wav(aggregated_wav, spk, result_dir):
    filename = 'test.wav'
    save_dir = os.path.join(result_dir, spk)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)
    librosa.output.write_wav(save_path, aggregated_wav, c.SAMPLE_RATE)
    
def main():
    # Test data directory
    test_dir = '/home/admin/Desktop/read_25h_2/test'
    
    result_dir = 'test_wavs'
    # Get the dataframe for test DB
    test_offline_DB = read_DB_structure(test_dir)
    
    # Set the duration of test data (ex. 4 <-> 15 sec)
    n_test_files = 4
    
    # Speaker and DB_list
    """ 
    '103F3021', '207F2088', '213F5100', '217F3038', '225M4062', 
    '229M2031', '230M4087', '233F4013', '236M3043', '240M3063'
    '""" 
    spk_list = ['103F3021','207F2088', '213F5100', '217F3038', '225M4062', \
    '229M2031', '230M4087', '233F4013', '236M3043', '240M3063']
    #random_select = True
    #margin = 0.5 # how long silence between utterances
    
    for spk in spk_list:
        # Aggregate the test wave file up to "test_len"
        aggregated_enroll_wav, aggregated_test_wav = aggregate_wav_files(test_offline_DB, n_test_files, spk)
        
        # Save the results
        save_aggregated_enroll_wav(aggregated_enroll_wav, spk, result_dir)
        save_aggregated_test_wav(aggregated_test_wav, spk, result_dir)

if __name__ == '__main__':
    main()