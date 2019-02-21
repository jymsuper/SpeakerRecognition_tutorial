# Wave path
TRAIN_WAV_DIR = '/home/admin/Desktop/read_25h_2/train'
DEV_WAV_DIR = '/home/admin/Desktop/read_25h_2/dev'
TEST_WAV_DIR = 'test_wavs'

# Feature path
TRAIN_FEAT_DIR = 'feat_logfbank_nfilt40/train'
TEST_FEAT_DIR = 'feat_logfbank_nfilt40/test'

# Context window size
NUM_WIN_SIZE = 100 #10

# Settings for feature extraction
USE_LOGSCALE = True
USE_DELTA = False
USE_SCALE = False
SAMPLE_RATE = 16000
FILTER_BANK = 40