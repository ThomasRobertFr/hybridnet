# HybridNet: Classification and Reconstruction Cooperation for SSL

This repository contains the official code for the paper:  
[**HybridNet: Classification and Reconstruction Cooperation for Semi-Supervised Learning**, T. Robert, N. Thome, M. Cord, _ECCV 2018_](https://arxiv.org/abs/1807.11407).

This code is based on the code of [Mean Teacher](https://github.com/CuriousAI/mean-teacher) by Antti Tarvainen and Harri Valpola.

The Python packages requirements are indicated in the file `requirements.txt` and are for Python 3.5+.

The `Makefile` takes care of downloading and preprocessing the data. It then runs the experiments described in `experiments.yml` by running the file `hybridnet/misc/jobs.py`. 

The models are described in `hybridnet/models/main.py` for the original Mean Teacher model and in `hybridnet/models/main_hybrid.py` for HybridNet. The architectures can be found in `hybridnet/models/architectures.py`.

Data will be stored in the `data` folder (should be a local disk for efficiency), the results (logs, model weights, etc.) will be in `results` and the tensorboard files will be in `tensorboard` folder.

