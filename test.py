import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util
import argparse
from collections import OrderedDict
from layers import Baseline, VGGLSTM, VGGLinear, Resnet, TimeCNN
from tqdm import tqdm
from tensorboardX import SummaryWriter
from util import collate_fn, Shots
import os

BATCH_SIZE = 32

def list_models_from_dir(dir):
    """ Returns all the saved models in the given directory"""
    all_files = [os.path.join(dir, f) for f in os.listdir(dir)]
    models = []
    for f in all_files:
        path_components = f.strip().split(".")
        if path_components[-1] == 'tar':
            models.append(f)

    return models

def main():
    #save_dir = util.get_save_dir('save','vgglinear', training=False)
    #log = util.get_logger(save_dir, 'vgglinear')
    save_dir = util.get_save_dir('save','TimeCNN', training=False)
    log = util.get_logger(save_dir, 'TimeCNN')
    device, gpu_ids = util.get_available_devices()
    tbx = SummaryWriter(save_dir)

    #save_dir = 'save/train/TimeCNN-wd0.01-epoch100-01'
    save_dir = 'save/train/TimeCNN-epoch30-1024-01'
    models = list_models_from_dir(save_dir)
    #print(models)
    best_accuracy, best_path = 0, ''
    for path in models:
        'save/train/TimeCNN-wd0.01-epoch100-01/best.pth.tar'
        #path = 'save/train/Resnet-82/best.pth.tar'
        #path = 'save/train/TimeCNN-epoch30-1024-01/best.pth.tar'
        #path = 'save/train/vgglinear-02/best.pth.tar'
        #build model here
        log.info("Building model")
        #model = Baseline(8 * 96 * 64)
        model = TimeCNN()
        #model = Resnet()
        #model = VGGLinear()
        model = nn.DataParallel(model, gpu_ids)
        model = util.load_model(model, path, gpu_ids, return_step=False)
        model = model.to(device)
        model = model.double()
        model.eval()

        log.info("Building Dataset")
        test_dataset = Shots("videos/test.h5py", "labels/test.npy") 
        test_loader = data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)
        num_correct = 0
        num_samles = 0
        missed_1, missed_0 = 0, 0
        num_1_predicted = 0
        with torch.no_grad():
            for frames, y in test_loader:
                frames = frames.to(device)
                y = y.to(device)
                scores = model(frames)
                loss = F.cross_entropy(scores, y)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                
                # This accumulates how many 1's and 0's were misclassified
                for i in range(y.shape[0]):
                    if y[i] == 1 and preds[i] == 0:
                        missed_1 += 1
                    elif y[i] == 0 and preds[i] == 1:
                        missed_0 += 1
                num_samples += preds.shape[0]
                num_1_predicted += (preds == 1).sum()
        
        acc = float(num_correct)/num_samples
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_path = path
        
        log.info("Path: {}".format(path))
        log.info("Accuracy on test set is {}".format(acc))
        log.info("Missed 1's: {}, Missed 0's: {}".format(missed_1, missed_0))
        log.info("Number 1's predicted: {}".format(num_1_predicted))
        log.info('-----------------')
    
    log.info("Best Accuracy on test set is {} and path was {}".format(best_accuracy, best_path))

if __name__ == '__main__':
    main()
