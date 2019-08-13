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
X_train = read_all_images(config.raw_path + '/stl10_binary/train_X.bin')
X_test = read_all_images(config.raw_path + '/stl10_binary/test_X.bin')
X_unsup = read_all_images(config.raw_path + '/stl10_binary/unlabeled_X.bin')

y_train = read_labels(config.raw_path + '/stl10_binary/train_y.bin') - 1
y_test = read_labels(config.raw_path + '/stl10_binary/test_y.bin') - 1
# y_unsup = np.zeros(X_unsup.shape[0], dtype=np.uint8)

classes = open(config.raw_path + '/stl10_binary/class_names.txt', "r").read().splitlines()

import matplotlib.image

for i in range(X_train.shape[0]):
    os.makedirs(config.imgfolder_path + "/images/train/" + classes[y_train[i]], exist_ok=True)
    matplotlib.image.imsave(config.imgfolder_path + "/images/train/" + classes[y_train[i]] + '/' + str(i) + '.png', X_train[i])


for i in range(X_test.shape[0]):
    os.makedirs(config.imgfolder_path + "/images/test/" + classes[y_test[i]], exist_ok=True)
    matplotlib.image.imsave(config.imgfolder_path + "/images/test/" + classes[y_test[i]] + '/' + str(i) + '.png',  X_test[i])

for i in range(X_unsup.shape[0]):
    os.makedirs(config.imgfolder_path + "/images/train/z_unlabeled", exist_ok=True)
    matplotlib.image.imsave(config.imgfolder_path + "/images/train/z_unlabeled/" + str(i + X_train.shape[0]) + '.png', X_unsup[i])

os.makedirs(config.imgfolder_path + "/labels", exist_ok=True)
with open(config.raw_path + '/stl10_binary/fold_indices.txt', "r") as f:
    for k, fold in enumerate(f.readlines()):
        fold = fold.strip().split(" ")
        fold = ["%s.png %s" % (img, classes[y_train[int(img)]]) for img in fold]

        with open("%s/labels/%02d.txt" % (config.imgfolder_path, k), "w") as f2:
            f2.write("\n".join(fold))



