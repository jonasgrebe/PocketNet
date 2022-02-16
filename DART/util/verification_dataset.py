import os
import pickle

import mxnet as mx
import numpy as np
import torch
from mxnet import ndarray as nd
from typing import List

class VerificationDataset(torch.utils.Dataset):

    def __init__(self, data_dir: str ='', dataset_name: str = 'lfw', image_size = [112, 112]):
        # TODO: Add support for multiple datasets at once
        # TODO: Add ability to transform "normal classification dataset" into this format

        path = os.path.join(data_dir, dataset_name + ".bin")

        # datalist is list of images (size 2N)
        # issame is list of boolean genuine/imposter labels (size N)
        self.datalist, self.issame = load_bin(path, image_size)

        pair_img0, pair_img1 = self.datalist[0::2], self.datalist[1::2]

        self.pairs  = list(zip(pair_img0, pair_img1))

        assert len(self.pairs) == len(self.issame)

    def __len__(self):
        return len(self.issame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # images are channel-last numpy arrays (height, width, 3)
        img0, img1 = self.pairs[idx]
        label = self.issame[idx]
        # convert to channel-first numpy arrays (3, height, width) for PyTorch
        img0, img1 = np.transpose(img0, (2,0,1)), np.transpose(img1, (2,0,1))
        # convert the numpy arrays to PyTorch tensors
        img0, img1 = torch.from_numpy(img0), torch.from_numpy(img1)

        return (img0, img1), label






@torch.no_grad()
def load_bin(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
        if idx % 1000 == 0:
            print('loading bin', idx)
    print(data_list[0].shape)
    return data_list, issame_list
