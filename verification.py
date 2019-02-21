import torch
import torch.nn.functional as F
from torch.autograd import Variable

import pandas as pd
import math
import os
import configure as c

from DB_wav_reader import read_feats_structure
from SR_Dataset import read_MFB, ToTensorTestInput
from model.model import background_resnet

def load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes):
    model = background_resnet(embedding_size=embedding_size, num_classes=n_classes)
    if use_cuda:
        model.cuda()
    print('=> loading checkpoint')
    # original saved file with DataParallel
    checkpoint = torch.load(log_dir + '/checkpoint_' + str(cp_num) + '.pth')
    # create new OrderedDict that does not contain `module.`
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def split_enroll_and_test(dataroot_dir):
    DB_all = read_feats_structure(dataroot_dir)
    enroll_DB = pd.DataFrame()
    test_DB = pd.DataFrame()
    
    enroll_DB = DB_all[DB_all['filename'].str.contains('enroll.p')]
    test_DB = DB_all[DB_all['filename'].str.contains('test.p')]
    
    # Reset the index
    enroll_DB = enroll_DB.reset_index(drop=True)
    test_DB = test_DB.reset_index(drop=True)
    return enroll_DB, test_DB

def load_enroll_embeddings(embedding_dir):
    embeddings = {}
    for f in os.listdir(embedding_dir):
        spk = f.replace('.pth','')
        # Select the speakers who are in the 'enroll_spk_list'
        embedding_path = os.path.join(embedding_dir, f)
        tmp_embeddings = torch.load(embedding_path)
        embeddings[spk] = tmp_embeddings
        
    return embeddings

def get_embeddings(use_cuda, filename, model, test_frames):
    input, label = read_MFB(filename) # input size:(n_frames, n_dims)
    
    tot_segments = math.ceil(len(input)/test_frames) # total number of segments with 'test_frames' 
    activation = 0
    with torch.no_grad():
        for i in range(tot_segments):
            temp_input = input[i*test_frames:i*test_frames+test_frames]
            
            TT = ToTensorTestInput()
            temp_input = TT(temp_input) # size:(1, 1, n_dims, n_frames)
    
            if use_cuda:
                temp_input = temp_input.cuda()
            temp_activation,_ = model(temp_input)
            activation += torch.sum(temp_activation, dim=0, keepdim=True)
    
    activation = l2_norm(activation, 1)
                
    return activation

def l2_norm(input, alpha):
    input_size = input.size()  # size:(n_frames, dim)
    buffer = torch.pow(input, 2)  # 2 denotes a squared operation. size:(n_frames, dim)
    normp = torch.sum(buffer, 1).add_(1e-10)  # size:(n_frames)
    norm = torch.sqrt(normp)  # size:(n_frames)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
    output = output * alpha
    return output

def perform_verification(use_cuda, model, embeddings, enroll_speaker, test_filename, test_frames, thres):
    enroll_embedding = embeddings[enroll_speaker]
    test_embedding = get_embeddings(use_cuda, test_filename, model, test_frames)

    score = F.cosine_similarity(test_embedding, enroll_embedding)
    score = score.data.cpu().numpy() 
        
    if score > thres:
        result = 'Accept'
    else:
        result = 'Reject'
        
    test_spk = test_filename.split('/')[-2].split('_')[0]
    print("\n=== Speaker verification ===")
    print("True speaker: %s\nClaimed speaker : %s\n\nResult : %s\n" %(enroll_speaker, test_spk, result))
    print("Score : %0.4f\nThreshold : %0.2f\n" %(score, thres))

def main():
    
    log_dir = 'model_saved' # Where the checkpoints are saved
    embedding_dir = 'enroll_embeddings' # Where embeddings are saved
    test_dir = 'feat_logfbank_nfilt40/test/' # Where test features are saved
    
    # Settings
    use_cuda = True # Use cuda or not
    embedding_size = 128 # Dimension of speaker embeddings
    cp_num = 24 # Which checkpoint to use?
    n_classes = 240 # How many speakers in training data?
    test_frames = 100 # Split the test utterance 

    # Load model from checkpoint
    model = load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes)
    
    # Get the dataframe for test DB
    enroll_DB, test_DB = split_enroll_and_test(c.TEST_FEAT_DIR)
    
    # Load enroll embeddings
    embeddings = load_enroll_embeddings(embedding_dir)
    
    """ Test speaker list
    '103F3021', '207F2088', '213F5100', '217F3038', '225M4062', 
    '229M2031', '230M4087', '233F4013', '236M3043', '240M3063'
    """ 
    
    # Set the true speaker
    enroll_speaker = '230M4087'
    
    # Set the claimed speaker
    test_speaker = '230M4087' 
    
    # Threshold
    thres = 0.95
    
    test_path = os.path.join(test_dir, test_speaker, 'test.p')
    
    # Perform the test 
    perform_verification(use_cuda, model, embeddings, enroll_speaker, test_path, test_frames, thres)

if __name__ == '__main__':
    main()