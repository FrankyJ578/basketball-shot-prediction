import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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
        self.pool1 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv3d(16, 16, kernel_size = 3, padding = 1, stride = 1)
        self.pool2 = nn.MaxPool3d(kernel_size = 2, padding = 2)
        self.conv3 = nn.Conv3d(16, 32, 3, stride = 2, padding = 1)
        # TODO: FIGURE OUT THIS SHAPE lol
        self.linear1 = nn.Linear(69, 256)
        self.linear2 = nn.Linear(256, 2)
        

    def forward(self, frames):

        inputs = torch.unsqueeze(frames, dim = 1)
        
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


