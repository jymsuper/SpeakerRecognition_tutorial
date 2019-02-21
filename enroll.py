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

def enroll_per_spk(use_cuda, test_frames, model, DB, embedding_dir):
    """
    Output the averaged d-vector for each speaker (enrollment)
    Return the dictionary (length of n_spk)
    """
    n_files = len(DB) # 10
    enroll_speaker_list = sorted(set(DB['speaker_id']))
    
    embeddings = {}
    
    # Aggregates all the activations
    print("Start to aggregate all the d-vectors per enroll speaker")
    
    for i in range(n_files):
        filename = DB['filename'][i]
        spk = DB['speaker_id'][i]
        
        activation = get_embeddings(use_cuda, filename, model, test_frames)
        if spk in embeddings:
            embeddings[spk] += activation
        else:
            embeddings[spk] = activation
            
        print("Aggregates the activation (spk : %s)" % (spk))
        
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)
        
    # Save the embeddings
    for spk_index in enroll_speaker_list:
        embedding_path = os.path.join(embedding_dir, spk_index+'.pth')
        torch.save(embeddings[spk_index], embedding_path)
        print("Save the embeddings for %s" % (spk_index))
    return embeddings
    
def main():
        
    # Settings
    use_cuda = True
    log_dir = 'model_saved'
    embedding_size = 128
    cp_num = 24 # Which checkpoint to use?
    n_classes = 240
    test_frames = 200
    
    # Load model from checkpoint
    model = load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes)
    
    # Get the dataframe for enroll DB
    enroll_DB, test_DB = split_enroll_and_test(c.TEST_FEAT_DIR)
    
    # Where to save embeddings
    embedding_dir = 'enroll_embeddings'
    
    # Perform the enrollment and save the results
    enroll_per_spk(use_cuda, test_frames, model, enroll_DB, embedding_dir)
    
    """ Test speaker list
    '103F3021', '207F2088', '213F5100', '217F3038', '225M4062', 
    '229M2031', '230M4087', '233F4013', '236M3043', '240M3063'
    """ 

if __name__ == '__main__':
    main()