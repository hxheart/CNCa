# Supplementary Material for the paper *Learning to Configure Computer Networks with Neural Algorithmic Reasoning*

This repository contains the code for the paper, including the scripts that can be used generate datasets, train models as well as reproduce our evaluation.

## Project Structure

The project is structured as follows:

- `dataset/` contains the code for dealing with fact bases and generating datasets. This includes the code used for BGP/OSPF configuration synthesis.

- `evaluation/` contains the scripts and notebooks used to produce our evaluation results, given a trained synthesizer model.

- `model/` contains the implementation of our neural synthesizer model architecture, including our structural fact base embedding.

- `trained-model/` contains a trained synthesizer model for BGP/OSPF configuration synthesis as used in our evaluation. You can load the provided checkpoint using `train_bgp.py` or any of the evaluation scripts.

- `train_bgp.py` can be used to train neural synthesizer models, given a generated dataset of fact bases of e.g. BGP/OSPF topologies, configurations and specifications.

## Requirements

```
# install PyTorch with GPU support
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# install PyG and other requirements
conda install pyg matplotlib numpy scipy bs4 tensorboard jupyter cudatoolkit=11.3 -c pyg
```

For our experiments, we relied on Python 3.9.12, PyTorch 1.11.0 and torch_geometric/PyG 2.0.4.

## Model Training

To train a synthesizer model, you can use the `train_bgp.py` script. Before training however, make sure you generate a dataset for the BGP/OSPF synthesis setting:

1. Generate dataset (delete the folder `bgp-ospf-dataset` if you want to generate a new or larger dataset)

```
python3 dataset/generate_training_dataset.py
```

2. Train the model

```
python3 train_bgp.py
```

The training procedure will continuously report training metrics and progress in the form of a TensorBoard summary in subfolder `runs/`. Further, it will save trained model snapshots in subfolder `models/` which can be used with our evaluation scripts.

## Evaluation

The different scripts that can be used to reproduce our evaluation can be found in the subfolders of `evaluation/`. See the corresponding README files for further instructions.