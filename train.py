''' 
Nolan Handali and Franklin Jia, 2019.
Some training code adapted from Chris Chute's starter code for 224N

Usage: python train.py -n NAME
'''


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
from layers import Baseline, VGGLSTM
from tqdm import tqdm
from tensorboardX import SummaryWriter
from util import collate_fn, Shots

BATCH_SIZE = 32 

def main(args):
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    device, args.gpu_ids = util.get_available_devices()
    tbx = SummaryWriter(args.save_dir)

    #this lets use save model
    saver = util.CheckpointSaver(args.save_dir, max_checkpoints = 15, metric_name='accuracy', maximize_metric = True, log=log)


    #build model here
    log.info("Building model")
    model = VGGLSTM()
    model = model.double()
    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info("Loading checkpoints")
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr = 0.001, betas=(.8,.999), eps=1e-07, weight_decay=.001)
    log.info("Building Dataset")
    train_dataset = Shots("videos/train.h5py", "labels/train.npy")
    train_loader = data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)

    dev_dataset = Shots("videos/dev.h5py", "labels/dev.npy")
    dev_loader = data.DataLoader(dev_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers=4, collate_fn = collate_fn)
    
    #print(len(train_loader.dataset))
    log.info("Training")
    steps_til_eval = 2000

    for epoch in range(100):
        with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
            for frames, ys in train_loader:
                batch_size = frames.shape[0]
                step += batch_size
                frames = frames.to(device)
                ys = ys.to(device)
                optimizer.zero_grad()

                #forwards pass      
                scores = model(frames)
                loss = F.cross_entropy(scores, ys)
                loss_val = loss.item()

                #Backwards pass
                loss.backward()
                optimizer.step()

                #some logging
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch, NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                steps_til_eval -= batch_size
                if steps_til_eval <= 0:
                    steps_til_eval = 2000
                    results, loss = evaluate(model, dev_loader, device)
                    # save checkpoint
                    saver.save(step, model, results, device)
                    log.info("Dev Accuracy " + str(results))
                    log.info("Dev loss " + str(loss))
                    #logging to tensorboard
                    tbx.add_scalar('dev_accuracy', results, step)
                    tbx.add_scalar("dev_loss", loss, step)
    
    tbx.close()

def evaluate(model, loader, device):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for frames, y in loader:
            frames = frames.to(device)
            y = y.to(device)
            scores = model(frames)
            loss = F.cross_entropy(scores, y)

            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.shape[0]
        acc = float(num_correct)/num_samples
    model.train()
    return acc, loss.item()

def get_args():
    parser = argparse.ArgumentParser('Train a model on shots')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')

    return parser.parse_args()

if __name__ == '__main__':
    main(get_args())
