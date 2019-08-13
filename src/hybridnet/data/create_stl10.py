from six.moves import urllib

import os, sys, pickle

from ..misc import config as _config
config = _config.get().datasets.stl10

import os, sys, tarfile
import numpy as np

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib  # ugly but works
else:
    import urllib

SIZE = config.x * config.y * config.z

def read_labels(path_to_labels):
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    with open(path_to_data, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def read_single_image(image_file):
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    image = np.reshape(image, (3, 96, 96))
    image = np.transpose(image, (2, 1, 0))
    return image

def download_and_extract():
    dest_directory = config.raw_path
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = config.url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                                                          float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(config.url, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

download_and_extract()

# load data
X_train = read_all_images(config.raw_path+'/stl10_binary/train_X.bin')
X_test = read_all_images(config.raw_path+'/stl10_binary/test_X.bin')
X_unsup = read_all_images(config.raw_path+'/stl10_binary/unlabeled_X.bin')

y_train = read_labels(config.raw_path+'/stl10_binary/train_y.bin') - 1
y_test = read_labels(config.raw_path+'/stl10_binary/test_y.bin') - 1
y_unsup = np.zeros(X_unsup.shape[0], dtype=np.uint8)

sup_thresh = np.zeros(X_train.shape[0] + X_unsup.shape[0])
sup_thresh[X_train.shape[0]:] = 1.1

X_train = np.r_[X_train, X_unsup]
y_train = np.r_[y_train, y_unsup]
del X_unsup, y_unsup

mu = np.mean(X_train, axis=(0, 1, 2), keepdims=True)
sigma = np.std(X_train, axis=(0, 1, 2), keepdims=True)

data = {"trainval": {"images": X_train, "labels": y_train, "sup_thresh": sup_thresh},
        "test": {"images": X_test, "labels": y_test},
        "mean": mu, "std": sigma,
        "valfolds": []}

with open(config.raw_path+'/stl10_binary/fold_indices.txt', "r") as f:
    for fold in f.readlines():
        fold = fold.strip().split(" ")
        fold = set(map(lambda _: int(_), fold))
        valinds = list(set(range(5000)) - fold)
        val = np.zeros(X_train.shape[0], dtype=np.bool)
        val[valinds] = True
        data["valfolds"].append(val)

with open(config.path, 'wb') as f:
    pickle.dump(data, f, -1)
