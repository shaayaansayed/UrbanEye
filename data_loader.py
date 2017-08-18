import random
import numpy as np 
import skimage 
from scipy.misc import imread, imresize
import os 
import glob 

class DataLoader :
    def __init__(self, data_dir, im_csz, val_split=0.15) :
        self.im_csz = im_csz
        
        sat_paths = glob.glob(os.path.join(data_dir, 'satellite', '*'))
        label_paths = glob.glob(os.path.join(data_dir, 'labels', '*'))
        combined = list(zip(sat_paths, label_paths))
        random.shuffle(combined)
        sat_paths, label_paths = zip(*combined)

        num_val = int(len(sat_paths)*val_split)
        self.val_sat = sat_paths[:num_val]
        self.val_label = label_paths[:num_val]
        self.train_sat = sat_paths[num_val:]
        self.train_label = label_paths[num_val:]

        print('{} - train, {} - val'.format(len(self.train_sat), len(self.val_sat)))
        self.train_ix = 0
        self.val_ix = 0 

    def next_train_batch(self, batch_size) :
        inputs = np.zeros((batch_size, self.im_csz, self.im_csz, 3))
        targets = np.zeros((batch_size, self.im_csz, self.im_csz))

        wrap = False
        for ix in range(batch_size) :

            img_path = self.train_sat[self.train_ix]
            I = imread(img_path)
            img_sz = I.shape[0]
            c = np.random.randint(0, img_sz-self.im_csz)

            I = I[c:c+self.im_csz, c:c+self.im_csz, :]

            label_path = self.train_label[self.train_ix]
            label = np.load(label_path)[c:c+self.im_csz, c:c+self.im_csz]

            inputs[ix, :, :, :] = I
            targets[ix, :, :] = label

            if self.train_ix + 1 >= len(self.train_sat) :
                self.train_ix = 0
                wrap = True
            else :
                self.train_ix += 1

        return inputs, targets, wrap

    def next_val_batch(self, batch_size) :

        wrap = False
        if self.val_ix + batch_size >= len(self.val_sat) :
            batch_size_ = len(self.val_sat) - 1 - self.val_ix + 1
            wrap = True
        else :
            batch_size_ = batch_size

        inputs = np.zeros((batch_size, self.im_csz, self.im_csz, 3))
        targets = np.zeros((batch_size, self.im_csz, self.im_csz))

        c = 15
        filenames = []
        for ix in range(batch_size_) :
            img_path = self.val_sat[self.val_ix]
            filename = os.path.basename(img_path)
            I = imread(img_path)
            I = I[c:c+self.im_csz, c:c+self.im_csz, :]

            label_path = self.val_label[self.val_ix]
            label = np.load(label_path)[c:c+self.im_csz, c:c+self.im_csz]

            inputs[ix, :, :, :] = I
            targets[ix, :, :] = label

            filenames.append(filename)
            self.val_ix = 0 if wrap else self.val_ix + 1

        return inputs, targets, wrap, filenames