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
from layers import Baseline, VGGLSTM, VGGLinear
from tqdm import tqdm
from tensorboardX import SummaryWriter
from util import collate_fn, Shots

BATCH_SIZE = 32

def main():
    save_dir = util.get_save_dir('save','vgglinear', training=False)
    log = util.get_logger(save_dir, 'vgglinear')
    device, gpu_ids = util.get_available_devices()
    tbx = SummaryWriter(save_dir)



    path = 'save/train/vgglinear-02/best.pth.tar'
    #build model here
    log.info("Building model")
    model = VGGLinear()
    model = nn.DataParallel(model, gpu_ids)
    model = util.load_model(model, path, gpu_ids, return_step=False)
    model = model.to(device)
    model = model.double()
    model.eval()

    log.info("Building Dataset")
    test_dataset = Shots("videos/test.h5py", "labels/test.npy") 
    test_loader = data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for frames, y in test_loader:
            frames = frames.to(device)
            y = y.to(device)
            scores = model(frames)
            loss = F.cross_entropy(scores, y)

            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.shape[0]
    acc = float(num_correct)/num_samples
    log.info("Accuracy on test set is {}".format(acc))



if __name__ == '__main__':
    main()
