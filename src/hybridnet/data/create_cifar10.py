from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tarfile

import numpy as np
import glob

from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib

import tensorflow as tf
import os, sys, pickle
from scipy import linalg

from ..misc import config as _config
config = _config.get().datasets.cifar10

def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()
    return data

def ZCA(data, reg=1e-6):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S) + reg)), U.T)
    whiten = np.dot(data - mean, components.T)
    return components, mean, whiten



if not os.path.exists(config.raw_path):
    os.makedirs(config.raw_path)
filename = config.url.split('/')[-1]
filepath = os.path.join(config.raw_path, filename)
if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(config.url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(config.raw_path)


# Training set
print("Loading training data...")
train_images = np.zeros((config.n_train, config.x * config.y * config.z), dtype=np.float32)
train_labels = []
for i, data_fn in enumerate(
        sorted(glob.glob(config.raw_path + '/cifar-10-batches-py/data_batch*'))):
    batch = unpickle(data_fn)
    train_images[i * 10000:(i + 1) * 10000] = batch['data']
    train_labels.extend(batch['labels'])
train_images = (train_images - 127.5) / 255.
train_labels = np.asarray(train_labels, dtype=np.int64)

rand_ix = np.random.permutation(config.n_train)
train_images = train_images[rand_ix]
train_labels = train_labels[rand_ix]

# Test set
print("Loading test data...")
test = unpickle(config.raw_path + '/cifar-10-batches-py/test_batch')
test_images = test['data'].astype(np.float32)
test_images = (test_images - 127.5) / 255.
test_labels = np.asarray(test['labels'], dtype=np.int64)

# Copy plain data
train_images_plain = np.array(train_images) * 2
test_images_plain = np.array(test_images) * 2

# ZCA
print("Apply ZCA whitening")
components, mean, train_images = ZCA(train_images)
np.save('{}/components'.format(config.raw_path), components)
np.save('{}/mean'.format(config.raw_path), mean)
test_images = np.dot(test_images - mean, components.T)

train_images = train_images.reshape(
    (config.n_train, config.z, config.x, config.y)).transpose((0, 2, 3, 1)).reshape((config.n_train, -1))
test_images = test_images.reshape(
    (config.n_test, config.z, config.x, config.y)).transpose((0, 2, 3, 1)).reshape((config.n_test, -1))

data = {'train': {"images": train_images, "labels": train_labels},
        'test':  {"images": test_images,  "labels": test_labels}}

os.makedirs(os.path.dirname(config.path), exist_ok=True)
with open(config.path, 'wb') as f:
    pickle.dump(data, f, -1)
