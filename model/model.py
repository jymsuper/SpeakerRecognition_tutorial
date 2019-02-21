import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import model.resnet as resnet


class background_resnet(nn.Module):
    def __init__(self, embedding_size, num_classes, backbone='resnet18'):
        super(background_resnet, self).__init__()
        self.backbone = backbone
        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=False)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=False)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=False)
        elif backbone == 'resnet18':
            self.pretrained = resnet.resnet18(pretrained=False)
        elif backbone == 'resnet34':
            self.pretrained = resnet.resnet34(pretrained=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
            
        self.fc0 = nn.Linear(128, embedding_size)
        self.bn0 = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU()
        self.last = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        # input x: minibatch x 1 x 40 x 40
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)
        
        out = F.adaptive_avg_pool2d(x,1) # [batch, 128, 1, 1]
        out = torch.squeeze(out) # [batch, n_embed]
        # flatten the out so that the fully connected layer can be connected from here
        out = out.view(x.size(0), -1) # (n_batch, n_embed)
        spk_embedding = self.fc0(out)
        out = F.relu(self.bn0(spk_embedding)) # [batch, n_embed]
        out = self.last(out)
        
        return spk_embedding, out