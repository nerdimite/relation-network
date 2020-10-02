# Relation Network
## Sort-of-CLEVR Dataset

Relation Network is a plug-n-play module to incorporate relational reasoning abilities to deep learning models.

This repository implements Relation Network module in PyTorch for the Sort-of-CLEVR dataset

## Requirements
* PyTorch
* OpenCV
* Numpy
* Pickle
* Tqdm

## Usage
1. Generate the dataset by running the [data_generator.py](data_generator.py) script.
2. Start the training using the [train.py](train.py) script. This script will automatically evaluate on the test set at the end of the training.
3. To make predictions on new data, refer [Relation Networks_Sort-of-CLEVR.ipynb](Relation Networks_Sort-of-CLEVR.ipynb).
4. Alternatively, <a href="https://colab.research.google.com/drive/1qopDwssLAklHkj5qiyv4LWmyEiJLcQLN?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## References
* A simple neural network module for relational reasoning - https://arxiv.org/abs/1706.01427
* https://github.com/kimhc6028/relational-networks
* https://github.com/mesnico/RelationNetworks-CLEVR