import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict

# File creating code that creates resnext, got from kenshoro github
import resnext

PRETRAINED_RESNET_PATH = 'pretrained_resnet/resnext-101-kinetics.pth'

class Baseline(nn.Module):
    def __init__(self, input_size):
        super(Baseline, self).__init__()

        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 100)
        self.linear3 = nn.Linear(100, 2)

    def forward(self, input):
        input = input.view(input.shape[0], -1)
        out = F.relu(self.linear1(input))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        return out


class VGGLSTM(nn.Module):
    def __init__(self):
        super(VGGLSTM, self).__init__()
        original = models.vgg16(pretrained = True)
       # print(*list(original.features.children()))
        self.features = nn.Sequential(*list(original.features.children()))
        for param in self.features.parameters():
            param.requires_grad = False
        self.pool = nn.AvgPool2d(2, stride = 2)
        self.lstm = nn.LSTM(input_size = 512, hidden_size = 256, num_layers = 1)
        self.linear1 = nn.Linear(256, 32)
        self.linear2 = nn.Linear(32,2)

    #frames is size (batch_size, 8, 3, 96, 64)
    # this is because in dataloader, we turned greyscale into stacked greyscale
    # for vgg reasons
    def forward(self, frames):
        batch_size = frames.shape[0]
        inputs = frames.reshape(-1, 3, 96, 64)
        #print("Inputs", inputs.shape)
        # feats is shape (batch_size * 8, feature_map size of vgg)
        feats = self.features(inputs)
        feats = self.pool(feats)
        #print("After vgg", feats.shape)
        feats = feats.reshape(8, batch_size, -1)
        #print("After reshape", feats.shape)
        outputs, (h,c) = self.lstm(feats)
        # last_hidden should be shape (batch_size x hidden_size)
        last_hidden = h[-1]
        #print("Last hidden", last_hidden.shape)
        output = F.relu(self.linear1(last_hidden))
        # output should be shape (batch_size x 32)
       # print("After first linear", output.shape)
        output = self.linear2(output)
        #return these 2 values per so we can calculate ce loss
        return output 


class TimeCNN(nn.Module):
    def __init__(self):
        super(TimeCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size = 5, padding = 2, stride = 1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv3d(16, 16, kernel_size = 3, padding = 1, stride = 1)
        self.bn2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv3d(16, 32, 3, stride = 2, padding = 1)
        
        #self.bn3 = nn.BatchNorm3d(32)
        #self.pool3 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        #self.conv4 = nn.Conv3d(32, 64, 5, padding = 2, stride = 1)

        # TODO: FIGURE OUT THIS SHAPE lol
        #self.linear1 = nn.Linear(6144, 3072)
        self.linear1 = nn.Linear(3072, 1024)
        self.dropout1 = nn.Dropout(.5)
        self.linear2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(.5)
        self.linear3 = nn.Linear(256, 2)
        #self.linear2 = nn.Linear(3072, 1536)
        #self.dropout2 = nn.Dropout(.5)
        #self.linear3 = nn.Linear(1536, 512)
        #self.dropout3 = nn.Dropout(.5)
        #self.linear4 = nn.Linear(512, 256)
        #self.dropout4 = nn.Dropout(.5)
        #self.linear5 = nn.Linear(256, 2)
        print("hello")
        

    def forward(self, frames):
        batch_size = frames.shape[0]
        #print(f'Frames shape: {frames.shape}')
        inputs = torch.unsqueeze(frames, dim = 1)
        #print(f'Inputs shape: {inputs.shape}')
        feats = self.conv1(inputs)
        #print(f'After conv1: {feats.shape}')
        feats = self.bn1(feats)
        feats = self.pool1(feats)
        #print(f'After pool1: {feats.shape}')
        feats = self.conv2(feats)
        #print(f'After conv2: {feats.shape}')
        feats = self.bn2(feats)
        feats = self.pool2(feats)
        #print(f'After pool2: {feats.shape}')
        feats = self.conv3(feats)
        #feats = self.bn3(feats)
        #feats = self.pool3(feats)
        #feats = self.conv4(feats)
        #print(f'After conv3: {feats.shape}')
        
        # Resize to be batch_size x features
        feats = feats.reshape((batch_size, -1))
        outputs = self.linear1(feats)
        #print(f'After linear1: {outputs.shape}')
        outputs = self.dropout1(outputs)
        outputs = self.linear2(outputs)
        outputs = self.dropout2(outputs)
        outputs = self.linear3(outputs)
        #outputs = self.dropout3(outputs)
        #outputs = self.linear4(outputs)
        #outputs = self.dropout4(outputs)
        #outputs = self.linear5(outputs)
        #print(f'After linear2: {outputs.shape}')
        return outputs

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.resnet = resnext.resnet101(shortcut_type='B', sample_size=64, sample_duration=16)
        self.resnet = nn.DataParallel(self.resnet)
        #num_final_in = self.resnet.module.fc.in_features
        #self.resnet.fc = nn.Linear(num_final_in, 2)

        model_data = torch.load(PRETRAINED_RESNET_PATH)
        #new_model_data = OrderedDict()
        #for k, v in model_data.items():
        #    name = k.replace("module.", "")
        #    new_model_data[name] = v

        #self.resnet.load_state_dict(new_model_data)
        self.resnet.load_state_dict(model_data['state_dict'])

        print(len(list(self.resnet.children())))
        #print(list(self.resnet.children()))
        print(type(self.resnet.children()))

        for layer in self.resnet.children():
            layer.fc = nn.Linear(4096, 2)
        #self.resnet = nn.Sequential(*[*list(self.resnet.children()), Flatten()])

        for param in self.resnet.parameters():
            param.requires_grad = False
        
        for layer in self.resnet.children():
            layer.fc = nn.Linear(4096, 2)

        #self.linear1 = nn.Linear(4096, 400)
        #self.relu = nn.ReLU(inplace=True)
        #self.linear2 = nn.Linear(400, 2)
        
        #num_final_in = self.resnet.module.fc.in_features
        #self.resnet = nn.Sequential(*[*list(self.resnet.children())[:-1], Flatten(), nn.Linear(num_final_in, 2)])
        #self.resnet.fc = nn.Linear(num_final_in, 2)

    def forward(self, frames):
        return self.resnet.forward(frames)
        
        #feats = self.resnet(frames)
        #output = self.relu(self.linear1(feats))
        #return self.linear2(output)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
        
class VGGLinear(nn.Module):
    def __init__(self):
        super(VGGLinear, self).__init__()
        original = models.vgg16(pretrained = True)
       # print(*list(original.features.children()))
        self.features = nn.Sequential(*list(original.features.children()))
#        for param in self.features.parameters():
 #           param.requires_grad = False
        self.pool = nn.AvgPool2d(2, stride = 2)
        self.linear1 = nn.Linear(4096, 512)
        self.linear2 = nn.Linear(512, 64)
        self.linear3 = nn.Linear(64,2)
        self.dropout1 = nn.Dropout(.5)
        self.dropout2 = nn.Dropout(.5)
    #frames is size (batch_size, 8, 3, 96, 64)
    # this is because in dataloader, we turned greyscale into stacked greyscale
    # for vgg reasons
    def forward(self, frames):
        batch_size = frames.shape[0]
        inputs = frames.reshape(-1, 3, 96, 64)
        #print("Inputs", inputs.shape)
        # feats is shape (batch_size * 8, feature_map size of vgg)
        feats = self.features(inputs)
        #print("After vgg", feats.shape)
        feats = self.pool(feats)
        feats = feats.reshape(batch_size, -1)
        #print("After reshape", feats.shape)
        # last_hidden should be shape (batch_size x hidden_size)]
        #print("Last hidden", last_hidden.shape)
        output = F.relu(self.linear1(feats))
        output = self.dropout1(output)
        output = F.relu(self.linear2(output))
        output = self.dropout2(output)
        # output should be shape (batch_size x 32)
       # print("After first linear", output.shape)
        output = self.linear3(output)
        #return these 2 values per so we can calculate ce loss
        return output 
