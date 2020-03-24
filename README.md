# SpeakerRecognition_tutorial

- A pytorch implementation of d-vector based speaker recognition system.  
- ResNet-based feature extractor, global average pooling and softmax layer with cross-entropy loss.  
- All the features (log Mel-filterbank features) for training and testing are uploaded.  
- Korean manual is included ("2019_LG_SpeakerRecognition_tutorial.pdf").  

## Requirements
python 3.5+  
pytorch 1.0.0  
pandas 0.23.4  
numpy 1.13.3  
pickle 4.0  
matplotlib 2.1.0  

## Datasets
We used the dataset collected through the following task.
- No. 10063424, 'development of distant speech recognition and multi-task dialog processing technologies for in-door conversational robots'

Specification
- Korean read speech corpus (ETRI read speech)
- Clean speech at a distance of 1m and a direction of 0 degrees
- 16kHz, 16bits  

We uploaded 40-dimensional log Mel-filterbank energy features extracted from the above dataset.  
[python_speech_features](https://github.com/jameslyons/python_speech_features) library is used.

#### 1. Train
24000 utterances, 240 folders (240 speakers)  
Size : 3GB  
```feat_logfbank_nfilt40 - train```

#### 2. Enroll & test
20 utterances, 10 folders (10 speakers)  
Size : 11MB  
```feat_logfbank_nfilt40 - test```

## Usage
### 1. Training  
Background model (ResNet based speaker classifier) is trained.  
You can change settings for training in 'train.py' file.

```python train.py```  

### 2. Enrollment  
Extract the speaker embeddings (d-vectors) using 10 enrollment speech files.  
They are extracted from the last hidden layer of the background model.  
All the embeddings are saved in 'enroll_embeddings' folder.  

```python enroll.py```  

### 3. Testing
For speaker verification,  you can change settings in 'verification.py' file.  

```python verification.py```  

For speaker identification,  you can change settings in 'identification.py' file.  

```python identification.py```

## How to train using your own dataset
#### 1. Modify the line 21 in train.py  
``` train_DB, valid_DB = split_train_dev(c.TRAIN_FEAT_DIR, val_ratio) ```  
- 'c.TRAIN_FEAT_DIR' in configure.py should be the path of your dataset  
- 'c.TRAIN_FEAT_DIR' should have the structure as: FEAT_DIR/speaker_folders/features_files.p  

#### 2. Modify the line 31 in DB_wav_reader.py  
```
def find_feats(directory, pattern='**/*.p'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)
```  
- I assumed that all the features are extracted in '.p' format.  
- If you want to change the extension, please change line 31 in DB_wav_reader.py  
- pattern='**/*.p' should be changed according to your feature format.  
- If you didn't extract features yet, please do that using python_speech_features library.  
- I didn't upload the code for feature extraction.  
- Of course, you can use other libraries.  

#### 3. Change the line 12 in SR_Dataset.py  
``` 
def read_MFB(filename):
    with open(filename, 'rb') as f:
        feat_and_label = pickle.load(f)
        
    feature = feat_and_label['feat'] # size : (n_frames, dim=40)
    label = feat_and_label['label']
```
- It is assumed that the feature file format is pickle. You need to change the code according to the format.  
- You have to change the function 'read_MFB' according to your situation.  
- From line 12 to line 16, we load feature (it is assumed the feature is saved using pickle) and label.  
- Feature size should be (n_frames, dim) as written in the comment. Label should be the speaker identity in string.  
- You can remove from line 20 to 24 because it is assumed that the front and back of the utterance is silence.  

#### 4. Change other options
- Be aware that all the settings are set to the small dataset as the training set in this tutorial is very small.  
- According to your dataset, you can make the model wider (increase the number of channels) and deeper (change to the ResNet-34) or increase the number of input frames ('NUM_WIN_SIZE' in configure.py).  
- More advanced loss function or pooling method (attentive pooling...) also can be used (not implemented here).  

## Author
Youngmoon Jung (dudans@kaist.ac.kr) at KAIST, South Korea
