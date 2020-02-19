import os
import glob
import numpy as np
import cv2
from multiprocessing import Pool
import pdb
from torch.utils.data import DataLoader, Dataset
import time
import dlib


class MyDataset(Dataset):
    def __init__(self, dataset='LRS3'):
        self.dataset = dataset
        if self.dataset == 'GRID':
            self.IN = 'GRID/'
            self.OUT = 'GRID_imgs/'
            self.wav = 'GRID_wavs/'

            with open('GRID_files.txt', 'r') as f:
                files = [line.strip() for line in f.readlines()]
                self.files = []
                for file in files:
                    _, ext = os.path.splitext(file)
                    if(ext == '.XML'): continue
                    self.files.append(file)
                    print(file)
                    wav = file.replace(self.IN, self.wav).replace(ext, '.wav')
                    path = os.path.split(wav)[0]
                    if(not os.path.exists(path)):
                        os.makedirs(path)
        elif self.dataset == 'LRS3':
            self.IN = '../dataset/LRS3/trainval/'
            self.OUT = '../dataset/LRS3/trainval_imgs/'
            self.files = sorted(glob.glob(os.path.join(self.IN, '*/*.mp4')))
 
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pdb.set_trace()
        file = self.files[idx]
        _, ext = os.path.splitext(file)
        dst = file.replace(self.IN, self.OUT).replace(ext, '')

        if(not os.path.exists(dst)):
            os.makedirs(dst)

        cap = cv2.VideoCapture(file)
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        #tmp = []
        #results = torch.zeros(videoMaxLen, 120, 120)
        while(True):
        # Capture frame-by-frame
            ret, frame = cap.read()
        #fig, ax = plt.subplots()
        #ax.imshow(frame)
        #plt.show()
            if ret:
            # Our operations on the frame come here
                gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (120, 120)).reshape(1, 120, 120)
            #gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (120, 120))
            #fig, ax = plt.subplots()
            #ax.imshow(gray, cmap='gray')
            #plt.show()
                tmp.append(gray)
            # Display the resulting frame
            #plt.imshow(gray, cmap='gray')
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        #cmd = 'ffmpeg -i \'{}\' -qscale:v 2 -r 25 \'{}/%d.jpg\''.format(file, dst)
        #os.system(cmd)

        return dst


if(__name__ == '__main__'):
    pdb.set_trace()
    dataset = MyDataset()
    loader = DataLoader(dataset, num_workers=0, batch_size=128, shuffle=False, drop_last=False)
    tic = time.time()
    for (i, batch) in enumerate(loader):
        eta = (1.0*time.time()-tic)/(i+1) * (len(loader)-i)
        print('eta:{}'.format(eta/3600.0))
