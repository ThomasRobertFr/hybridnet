from __future__ import print_function
import numpy as np
import os
import sys
# noinspection PyUnresolvedReferences
from six.moves.urllib.request import urlretrieve
# noinspection PyUnresolvedReferences
import pickle as pkl
import imageio

from ..misc import config as _config
config = _config.get()

last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 1% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, force=False):
    path = config.datasets.svhn.raw_path + "/" + filename
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(path):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(config.datasets.svhn.url + filename, path, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(path)
    return path

os.makedirs(config.datasets.svhn.raw_path, exist_ok=True)

train_matfile = maybe_download('train_32x32.mat')
test_matfile = maybe_download('test_32x32.mat')
extra_matfile = maybe_download('extra_32x32.mat')

import scipy.io
train_data = scipy.io.loadmat(train_matfile, variable_names='X').get('X')
train_labels = scipy.io.loadmat(train_matfile, variable_names='y').get('y')
test_data = scipy.io.loadmat(test_matfile, variable_names='X').get('X')
test_labels = scipy.io.loadmat(test_matfile, variable_names='y').get('y')
extra_data = scipy.io.loadmat(extra_matfile, variable_names='X').get('X')
extra_labels = scipy.io.loadmat(extra_matfile, variable_names='y').get('y')

train_labels[train_labels == 10] = 0
test_labels[test_labels == 10] = 0
extra_labels[extra_labels == 10] = 0

import random

random.seed()

n_labels = 10
valid_index = []
valid_index2 = []
train_index = []
train_index2 = []
for i in np.arange(n_labels):
    valid_index.extend(np.where(train_labels[:,0] == (i))[0][:400].tolist())
    train_index.extend(np.where(train_labels[:,0] == (i))[0][400:].tolist())
    valid_index2.extend(np.where(extra_labels[:,0] == (i))[0][:200].tolist())
    train_index2.extend(np.where(extra_labels[:,0] == (i))[0][200:].tolist())

random.shuffle(valid_index)
random.shuffle(train_index)
random.shuffle(valid_index2)
random.shuffle(train_index2)

valid_data = np.concatenate((extra_data[:,:,:,valid_index2], train_data[:,:,:,valid_index]), axis=3).transpose((3,0,1,2))
valid_labels = np.concatenate((extra_labels[valid_index2,:], train_labels[valid_index,:]), axis=0)[:,0]
train_data_t = np.concatenate((extra_data[:,:,:,train_index2], train_data[:,:,:,train_index]), axis=3).transpose((3,0,1,2))
train_labels_t = np.concatenate((extra_labels[train_index2,:], train_labels[train_index,:]), axis=0)[:,0]
test_data = test_data.transpose((3,0,1,2))
test_labels = test_labels[:,0]

print(train_data_t.shape, train_labels_t.shape)
print(test_data.shape, test_labels.shape)
print(valid_data.shape, valid_labels.shape)

data = { 'train': {"images": train_data_t, "labels": train_labels_t},
         'test': {"images": test_data, "labels": test_labels},
         'valid': {"images": valid_data, "labels": valid_labels}}

np.random.seed(42)
for dataset in data:
    data[dataset]["labels"] = np.array(data[dataset]["labels"], dtype=np.int32)
    sup_thresh = np.zeros(data[dataset]["labels"].shape)

    n_max = 0
    for i in range(10):
        n_max = max(n_max, int(np.sum(data[dataset]["labels"] == i)))

    for i in range(10):
        n = np.sum(data[dataset]["labels"] == i)
        sup_thresh_i = np.arange(1, n_max + 1) / (n_max + 1)
        sup_thresh_i = sup_thresh_i[:n]
        np.random.shuffle(sup_thresh_i)
        sup_thresh[data[dataset]["labels"] == i] = sup_thresh_i
    data[dataset]["sup_thresh"] = sup_thresh

# Save dataset as pickle
with open(config.datasets.svhn.path, 'wb') as f:
    pkl.dump(data, f, -1)

# TODO using config instead of fixed paths here
for key in data:
    for i in range(10):
        os.makedirs("data/processed/svhn/images/" + key + "/" + str(i), exist_ok=True)

    for i in range(len(data[key]["labels"])):
        path = "data/processed/svhn/images/" + key + "/" + str(data[key]["labels"][i]) + "/" + str(
            data[key]["labels"][i]) + "_" + str(i) + ".png"
        imageio.imwrite(path, data[key]["images"][i])

os.makedirs("data/processed/svhn/labels")

key = "train"
inds = np.argsort(data[key]['sup_thresh'])

# for key in data:
for N in [250, 500, 1000]:
    out = ""
    for i in inds[0:N]:
        out += str(data[key]["labels"][i]) + "_" + str(i) + ".png " + str(data[key]["labels"][i]) + "\n"
    with open("data/processed/svhn/labels/" + str(N) + ".txt", "w") as f:
        f.write(out)
