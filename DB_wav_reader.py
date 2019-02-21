"""
Modification of the function 'DBspeech_wav_reader.py' of the deep-speaker created by philipperemy 
Working on python 3
Input : DB path
Output : 1) Make DB structure using pd.DataFrame which has 3 columns (file id, file path, speaker id, DB id)
            => 'read_DB_structure' function
         2) Read a wav file from DB structure
            => 'read_audio' function
"""
import logging
import os
from glob import glob

import librosa
import numpy as np
import pandas as pd

from configure import SAMPLE_RATE

np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)


def find_wavs(directory, pattern='**/*.wav'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def find_feats(directory, pattern='**/*.p'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def read_audio(filename, sample_rate=SAMPLE_RATE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.flatten()
    return audio

def read_DB_structure(directory):
    DB = pd.DataFrame()
    DB['filename'] = find_wavs(directory) # filename
    DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/')) # normalize windows paths
    DB['speaker_id'] = DB['filename'].apply(lambda x: x.split('/')[-2]) # speaker folder name
    DB['dataset_id'] = DB['filename'].apply(lambda x: x.split('/')[-3]) # dataset folder name
    num_speakers = len(DB['speaker_id'].unique())
    logging.info('Found {} files with {} different speakers.'.format(str(len(DB)).zfill(7), str(num_speakers).zfill(5)))
    logging.info(DB.head(10))
    return DB

def read_feats_structure(directory):
    DB = pd.DataFrame()
    DB['filename'] = find_feats(directory) # filename
    DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/')) # normalize windows paths
    DB['speaker_id'] = DB['filename'].apply(lambda x: x.split('/')[-2]) # speaker folder name
    DB['dataset_id'] = DB['filename'].apply(lambda x: x.split('/')[-3]) # dataset folder name
    num_speakers = len(DB['speaker_id'].unique())
    logging.info('Found {} files with {} different speakers.'.format(str(len(DB)).zfill(7), str(num_speakers).zfill(5)))
    logging.info(DB.head(10))
    return DB
    
def test():
    DB_dir = '/home/administrator/Desktop/DB/Speaker_robot_train_DB'
    DB = read_DB_structure(DB_dir)
    test_wav = read_audio(DB[0:1]['filename'].values[0])
    return DB, test_wav


if __name__ == '__main__':
    DB, test_wav = test()