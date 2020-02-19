#!/home/ykhassanov/.conda/envs/py37_avsr/bin/python
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import sys
from dataset import MyDataset
import numpy as np
import time
from model import LipNet
import torch.optim as optim
import re
import json
from tensorboardX import SummaryWriter
import pdb

if(__name__ == '__main__'):
    opt = __import__('options')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    writer = SummaryWriter()

def dataset2dataloader(dataset, num_workers=opt.num_workers, shuffle=True):
    return DataLoader(dataset,
        batch_size = opt.batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = False)

def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()

def ctc_decode(y):
    result = []
    y = y.argmax(-1)
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]

def test(model, net):
    with torch.no_grad():
        dataset = MyDataset(opt.video_path, opt.anno_path, opt.val_list, opt.vid_padding,
                            opt.txt_padding, 'test')

        print('num_test_data:{}'.format(len(dataset.data)))
        model.eval()
        loader = dataset2dataloader(dataset, shuffle=False)
        loss_list = []
        wer = []
        cer = []
        crit = nn.CTCLoss()
        tic = time.time()
        for (i_iter, input) in enumerate(loader):
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()
 
            y = net(vid)

            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1),
                        txt_len.view(-1)).detach().cpu().numpy()
            loss_list.append(loss)
            pred_txt = ctc_decode(y)

            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            wer.extend(MyDataset.wer(pred_txt, truth_txt))
            cer.extend(MyDataset.cer(pred_txt, truth_txt))
            if(i_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(i_iter+1)
                eta = v * (len(loader)-i_iter) / 3600.0
 
                print(''.join(101*'-'))
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-'))
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101 *'-'))
                print('test_iter={0:}, eta={1:.4f}, wer={2:.4f}, cer={3:.4f}'.format(i_iter,
                        eta,np.array(wer).mean(), np.array(cer).mean()))
                print(''.join(101 *'-'))

        return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean())

def train(model, net):
    dataset = MyDataset(opt.video_path, opt.anno_path, opt.train_list, opt.vid_padding,
                        opt.txt_padding, 'train')

    loader = dataset2dataloader(dataset)
    optimizer = optim.Adam(model.parameters(), lr=opt.base_lr, weight_decay=opt.weight_decay,
                           amsgrad=True)

    #scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                    patience=opt.patience, verbose=True, threshold=1e-4)

    print('num_train_data:{}'.format(len(dataset.data)))
    crit = nn.CTCLoss()
    tic = time.time()

    train_wer = []
    for epoch in range(opt.max_epoch):
        tic_epoch = time.time()
        total_loss = 0
        for (i_iter, input) in enumerate(loader):
            model.train()
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()

            optimizer.zero_grad()
            y = net(vid)
            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))
            total_loss += loss.item()
            loss.backward()
            if(opt.is_optimize):
                optimizer.step()

            tot_iter = i_iter + epoch*len(loader)
            pred_txt = ctc_decode(y)
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt))

            if(tot_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(tot_iter+1)
                eta = (len(loader)-i_iter)*v/3600.0
                writer.add_scalar('train loss', loss, tot_iter)
                writer.add_scalar('train wer', np.array(train_wer).mean(), tot_iter)
                print(''.join(101*'-'))
                print('{:<50}|{:>50}'.format('predict', 'truth'))                
                print(''.join(101*'-'))

                for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101*'-'))
                print('epoch={0:}, tot_iter={1:}, eta={2:.4f}, loss={3:.6f}, train_wer={4:.4f}'.format(
                        epoch, tot_iter, eta, loss, np.array(train_wer).mean()))
                print(''.join(101*'-'))

            if(tot_iter % opt.test_step == 0 and tot_iter != 0):
                (loss, wer, cer) = test(model, net)
                print('\n' + ''.join(101*'*'))
                print('TEST SET: i_iter={0:}, lr={1:}, loss={2:.6f}, wer={3:.4f}, cer={4:.4f}'.format(
                        tot_iter, show_lr(optimizer), loss, wer, cer))
                print(''.join(101*'*') + '\n')
                scheduler.step(loss)
                writer.add_scalar('val loss', loss, tot_iter)
                writer.add_scalar('wer', wer, tot_iter)
                writer.add_scalar('cer', cer, tot_iter)
                savename = ('{0:}_bs{1:}_lr{2:}_wd{3:}_patience{4:}_drop{5:}_epoch{6:}_loss{7:.6f}'
                    '_wer{8:.4f}_cer{9:.4f}.pt').format(opt.save_prefix, opt.batch_size,
                    opt.base_lr, opt.weight_decay, opt.patience, opt.drop, epoch, loss, wer, cer)
                (path, name) = os.path.split(savename)
                if(not os.path.exists(path)):
                    os.makedirs(path)
                torch.save(model.state_dict(), savename)
                if(not opt.is_optimize):
                    exit()

        print('\n' + ''.join(101*'*'))
        print('EPOCH={0:}, total loss={1:6f}, time={2:.2f}m'.format(epoch, total_loss/len(dataset),
                                                         (time.time()-tic_epoch)/60))
        print(''.join(101*'*') + '\n')


if(__name__ == '__main__'):
    print("Loading options...")
    model = LipNet(opt.drop)
    model = model.cuda()
    net = nn.DataParallel(model).cuda()

    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    train(model, net)

