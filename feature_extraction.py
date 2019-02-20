#import argparse
import librosa
import os
import numpy as np
import configure as c
import pickle # For python3 
import pandas as pd

from python_speech_features import fbank, logfbank, delta
from DB_wav_reader import read_DB_structure
from SR_Dataset import convert_wav_to_feat_name


def mk_MFB(filename):
    audio, sr = librosa.load(filename, sr=c.SAMPLE_RATE, mono=True)
    features, energies = fbank(audio, samplerate=c.SAMPLE_RATE, nfilt=c.FILTER_BANK, winlen=0.025)
    
    if c.USE_LOGSCALE:
        features = 20 * np.log10(np.maximum(features,1e-5))
    
    if c.USE_DELTA:
        delta_1 = delta(features, N=1)
        delta_2 = delta(delta_1, N=1)
        
        features = normalize_frames(features, Scale=c.USE_SCALE)
        delta_1 = normalize_frames(delta_1, Scale=c.USE_SCALE)
        delta_2 = normalize_frames(delta_2, Scale=c.USE_SCALE)
        
        total_features = np.hstack([features, delta_1, delta_2])
    else:
        features = normalize_frames(features, Scale=c.USE_SCALE)
        total_features = features
    
    filename_only = filename.split('/')[-1].replace('.wav','.p')
    speaker_folder = filename.split('/')[-2]
    output_foldername, output_filename = convert_wav_to_feat_name(filename)
    
    speaker_label = speaker_folder # set label as a folder name (recommended). Convert this to speaker index when training

    feat_and_label = {'feat':total_features, 'label':speaker_label}
    
    if not os.path.exists(output_foldername):
        os.makedirs(output_foldername)
    
    if os.path.isfile(output_filename) == 1:
        print("\"" + filename_only + "\"" + " file already extracted!")
    else:
        with open(output_filename, 'wb') as fp:
            pickle.dump(feat_and_label, fp)

def normalize_frames(m,Scale=True):
    if Scale:
        return (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))

def select_train_DB(train_dataroot_dir, DB_list):
    train_DB_all = read_DB_structure(train_dataroot_dir)
    train_DB = pd.DataFrame()
    for i in DB_list:
        train_DB = train_DB.append(train_DB_all[train_DB_all['dataset_id'] == i], ignore_index=True)
    return train_DB

def feat_extraction(dataroot_dir, mode):
    
    DB = read_DB_structure(dataroot_dir)
    speaker_list = sorted(set(DB['speaker_id']))
    count = 0
    
    for i in range(len(DB)):
        mk_MFB(DB['filename'][i])
        count = count+1
        print("feature extraction (%s DB). step : %d, file : \"%s\"" %(mode, count, DB['filename'][i]))
    print("Feature extraction done")

if __name__ == '__main__':
    #feat_extraction(dataroot_dir=c.TRAIN_WAV_DIR, mode='train')
    feat_extraction(dataroot_dir=c.TEST_WAV_DIR, mode='test')
    #feat_extraction(dataroot_dir=c.DEV_WAV_DIR, mode='dev')
    