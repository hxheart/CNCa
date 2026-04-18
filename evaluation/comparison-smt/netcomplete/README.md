# NetComplete Performance Comparison

This folder contains the files required to do synthesis with NetComplete using the dataset of problem instances located in `../dataset/bgp-reqs-$N` and `../dataset/ospf-reqs-$N`.

Note that NetComplete is implemented using Python 2 and requires a completely separate runtime environment. 

## Setup

First, obtain a copy of NetComplete from https://github.com/nsg-ethz/synet-plus and place it in the folder `synet-plus/`.

```
git clone https://github.com/nsg-ethz/synet-plus.git
```

Then apply a small patch of fixes, to make NetComplete compatible with our evaluation setup.

```
cd synet-plus && git apply ../fixes.patch
```

Next you need to prepare a Python 2 environment with the requirements of  NetComplete. See `synet-plus/requirements.txt`.

## Running the Evaluation

To run the evaluation as included in the paper, you can use the scripts `run-ospf.sh` and `run-bgp.sh`.

## Results

The results as included in the paper can be found in the folder `results/`.

