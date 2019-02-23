import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import os
import pickle # For python3 
import numpy as np
import configure as c
from DB_wav_reader import read_DB_structure

def read_MFB(filename):
    with open(filename, 'rb') as f:
        feat_and_label = pickle.load(f)
        
    feature = feat_and_label['feat'] # size : (n_frames, dim=40)
    label = feat_and_label['label']
    """
    VAD
    """
    start_sec, end_sec = 0.5, 0.5
    start_frame = int(start_sec / 0.01)
    end_frame = len(feature) - int(end_sec / 0.01)
    ori_feat = feature
    feature = feature[start_frame:end_frame,:]
    assert len(feature) > 40, (
                'length is too short. len:%s, ori_len:%s, file:%s' % (len(feature), len(ori_feat), filename))
    return feature, label

class TruncatedInputfromMFB(object):
    """
    input size : (n_frames, dim=40)
    output size : (1, n_win=40, dim=40) => one context window is chosen randomly
    """
    def __init__(self, input_per_file=1):
        super(TruncatedInputfromMFB, self).__init__()
        self.input_per_file = input_per_file
    
    def __call__(self, frames_features):
        network_inputs = []
        num_frames = len(frames_features)
        
        win_size = c.NUM_WIN_SIZE
        half_win_size = int(win_size/2)
        #if num_frames - half_win_size < half_win_size:
        while num_frames - half_win_size <= half_win_size:
            frames_features = np.append(frames_features, frames_features[:num_frames,:], axis=0)
            num_frames =  len(frames_features)
            
        for i in range(self.input_per_file):
            j = random.randrange(half_win_size, num_frames - half_win_size)
            if not j:
                frames_slice = np.zeros(num_frames, c.FILTER_BANK, 'float64')
                frames_slice[0:(frames_features.shape)[0]] = frames_features.shape
            else:
                frames_slice = frames_features[j - half_win_size:j + half_win_size]
            network_inputs.append(frames_slice)
        return np.array(network_inputs)


class TruncatedInputfromMFB_test(object):
    def __init__(self, input_per_file=1):
        super(TruncatedInputfromMFB_test, self).__init__()
        self.input_per_file = input_per_file

    def __call__(self, frames_features):
        network_inputs = []
        num_frames = len(frames_features)

        for i in range(self.input_per_file):

            for j in range(c.NUM_PREVIOUS_FRAME, num_frames - c.NUM_NEXT_FRAME):
                frames_slice = frames_features[j - c.NUM_PREVIOUS_FRAME:j + c.NUM_NEXT_FRAME]
                # network_inputs.append(np.reshape(frames_slice, (32, 20, 3)))
                network_inputs.append(frames_slice)
        return np.array(network_inputs)

class TruncatedInputfromMFB_CNN_test(object):
    def __init__(self, input_per_file=1):
        super(TruncatedInputfromMFB_CNN_test, self).__init__()
        self.input_per_file = input_per_file

    def __call__(self, frames_features):
        network_inputs = []
        num_frames = len(frames_features)

        for i in range(self.input_per_file):

            for j in range(c.NUM_PREVIOUS_FRAME, num_frames - c.NUM_NEXT_FRAME):
                frames_slice = frames_features[j - c.NUM_PREVIOUS_FRAME:j + c.NUM_NEXT_FRAME]
                #network_inputs.append(np.reshape(frames_slice, (-1, c.NUM_PREVIOUS_FRAME+c.NUM_NEXT_FRAME, c.FILTER_BANK)))
                network_inputs.append(frames_slice)
        network_inputs = np.expand_dims(network_inputs, axis=1)
        assert network_inputs.ndim == 4, 'Data is not a 4D tensor. size:%s' % (np.shape(network_inputs),)
        return np.array(network_inputs)

class ToTensorInput(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, np_feature):
        """
        Args:
            feature (numpy.ndarray): feature to be converted to tensor.
        Returns:
            Tensor: Converted feature.
        """
        if isinstance(np_feature, np.ndarray):
            # handle numpy array
            ten_feature = torch.from_numpy(np_feature.transpose((0,2,1))).float() # output type => torch.FloatTensor, fast
            
            # input size : (1, n_win=200, dim=40)
            # output size : (1, dim=40, n_win=200)
            return ten_feature

class ToTensorDevInput(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, np_feature):
        """
        Args:
            feature (numpy.ndarray): feature to be converted to tensor.
        Returns:
            Tensor: Converted feature.
        """
        if isinstance(np_feature, np.ndarray):
            # handle numpy array
            np_feature = np.expand_dims(np_feature, axis=0)
            assert np_feature.ndim == 3, 'Data is not a 3D tensor. size:%s' %(np.shape(np_feature),)
            ten_feature = torch.from_numpy(np_feature.transpose((0,2,1))).float() # output type => torch.FloatTensor, fast
            # input size : (1, n_win=40, dim=40)
            # output size : (1, dim=40, n_win=40)
            return ten_feature

class ToTensorTestInput(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, np_feature):
        """
        Args:
            feature (numpy.ndarray): feature to be converted to tensor.
        Returns:
            Tensor: Converted feature.
        """
        if isinstance(np_feature, np.ndarray):
            # handle numpy array
            np_feature = np.expand_dims(np_feature, axis=0)
            np_feature = np.expand_dims(np_feature, axis=1)
            assert np_feature.ndim == 4, 'Data is not a 4D tensor. size:%s' %(np.shape(np_feature),)
            ten_feature = torch.from_numpy(np_feature.transpose((0,1,3,2))).float() # output type => torch.FloatTensor, fast
            # input size : (1, 1, n_win=200, dim=40)
            # output size : (1, 1, dim=40, n_win=200)
            return ten_feature

def collate_fn_feat_padded(batch):
    """
    Sort a data list by frame length (descending order)
    batch : list of tuple (feature, label). len(batch) = batch_size
        - feature : torch tensor of shape [1, 40, 80] ; variable size of frames
        - labels : torch tensor of shape (1)
    ex) samples = collate_fn([batch])
        batch = [dataset[i] for i in batch_indices]. ex) [Dvector_train_dataset[i] for i in [0,1,2,3,4]]
        batch[0][0].shape = torch.Size([1,64,774]). "774" is the number of frames per utterance. 
        
    """
    batch.sort(key=lambda x: x[0].shape[2], reverse=True)
    feats, labels = zip(*batch)
    
    # Merge labels => torch.Size([batch_size,1])
    labels = torch.stack(labels, 0)
    labels = labels.view(-1)
    
    # Merge frames
    lengths = [feat.shape[2] for feat in feats] # in decreasing order 
    max_length = lengths[0]
    # features_mod.shape => torch.Size([batch_size, n_channel, dim, max(n_win)])
    padded_features = torch.zeros(len(feats), feats[0].shape[0], feats[0].shape[1], feats[0].shape[2]).float() # convert to FloatTensor (it should be!). torch.Size([batch, 1, feat_dim, max(n_win)])
    for i, feat in enumerate(feats):
        end = lengths[i]
        num_frames = feat.shape[2]
        while max_length > num_frames:
            feat = torch.cat((feat, feat[:,:,:end]), 2)
            num_frames = feat.shape[2]
        
        padded_features[i, :, :, :] = feat[:,:,:max_length]
    
    return padded_features, labels

class DvectorDataset(data.Dataset):
    def __init__(self, DB, loader, spk_to_idx, transform=None, *arg, **kw):
        self.DB = DB
        self.len = len(DB)
        self.transform = transform
        self.loader = loader
        self.spk_to_idx = spk_to_idx
    
    def __getitem__(self, index):
        feat_path = self.DB['filename'][index]
        feature, label = self.loader(feat_path)
        label = self.spk_to_idx[label]
        label = torch.Tensor([label]).long()
        if self.transform:
            feature = self.transform(feature)
        
        return feature, label
    
    def __len__(self):
        return self.len
        
def main():
    train_DB = read_DB_structure(c.TRAIN_DATAROOT_DIR)
    transform = transforms.Compose([
        truncatedinputfromMFB(),
        totensor_DNN_input()
    ])
    file_loader = read_MFB
    speaker_list = sorted(set(train_DB['speaker_id']))
    spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}
    batch_size = 128
    Dvector_train_dataset = Dvector_Dataset(DB=train_DB, loader=file_loader, transform=transform, spk_to_idx=spk_to_idx)
    Dvector_train_loader = torch.utils.data.DataLoader(dataset=Dvector_train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False)

if __name__ == '__main__':
    main()