 ### GENERAL CONFIG

project = hybridnet
python = export PYTHONPATH=src; python
include .env


### ENVIRONMENTS VARS

.env: config.yml # config-private.yml
	$(python) -m $(project).misc.config env

### DATASETS

datasets: cifar10 svhn stl10
svhn: $(datasets_svhn_path)
cifar10: $(datasets_cifar10_path) $(datasets_cifar10_folder_path)
stl10: $(datasets_stl10_path) $(datasets_stl10_imgfolder_path)

$(datasets_svhn_path): src/$(project)/data/create_svhn.py
	$(python) -m $(project).data.create_svhn

$(datasets_cifar10_path): # src/$(project)/data/create_cifar10.py
	$(python) -m $(project).data.create_cifar10
$(datasets_cifar10_folder_path):
	$(python) -m $(project).data.create_cifar10_folder

$(datasets_stl10_path): # src/$(project)/data/create_stl10.py
	$(python) -m $(project).data.create_stl10

$(datasets_stl10_imgfolder_path):
	$(python) -m $(project).data.stl10_folder

### MODELS

# Select the GPU with the most free RAM by default
GPU ?= $(shell nvidia-smi --format=csv,noheader,nounits --query-gpu=memory.free,index | sort -n | tail -n 1 | sed 's/[0-9]\+, //g')

jobs: datasets
	export CUDA_VISIBLE_DEVICES=$(GPU); $(python) -m $(project).misc.jobs $(ARGS)

all: jobs

