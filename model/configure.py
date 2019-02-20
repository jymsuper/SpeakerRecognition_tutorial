
TRAIN_DATAROOT_DIR = '/data/DB/Speaker_robot_train_DB_dist'

# Choose the enroll and test distance
TEST_DATAROOT_DIR = '/data/DB/Speaker_robot_test_DB/1M0D' # /1M0D/ or /3M0D/
ENROLL_DATAROOT_DIR = '/data/DB/Speaker_robot_test_DB/1M0D' # /1M0D/ or /3M0D/

FEAT_DIR = '/data/DB/'
TEST_FEAT_DATAROOT_DIR = '/data/DB/Speaker_robot_test_DB/1M0D/kaist_10h_refmotor_snr103c'

NUM_PREVIOUS_FRAME = 50 #30
NUM_NEXT_FRAME = 50 #10

NUM_FRAMES = NUM_PREVIOUS_FRAME + NUM_NEXT_FRAME
USE_LOGSCALE = True
USE_DELTA = False
USE_SCALE = False
SAMPLE_RATE = 16000
TRUNCATE_SOUND_FIRST_SECONDS = 0.5
FILTER_BANK = 40