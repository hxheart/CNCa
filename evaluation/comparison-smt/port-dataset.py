# get this file's directory 
import sys
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_dir, "../../dataset"))
sys.path.append(os.path.join(this_dir, "../../model"))

import argparse

from pydoc import describe

from torch_geometric.data import Data
import torch
from tqdm import tqdm
from factbase import FactBase

parser = argparse.ArgumentParser(description='Port dataset files from the old PyG format to a new dict-based format as supported by FactBase.from_data.')
parser.add_argument('DATASET_IN', type=str)
parser.add_argument('DATASET_OUT', type=str)

args = parser.parse_args()

for file in tqdm(os.listdir(args.DATASET_IN), leave=False):
    if not file.endswith(".logic"): 
        continue

    data = torch.load(os.path.join(args.DATASET_IN, file))
    fb = FactBase.from_data(data)
    torch.save(fb.to_data(), os.path.join(args.DATASET_OUT, file))