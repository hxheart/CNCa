import multiprocessing
from factbase import FactBase

import numpy as np
from torch_geometric.data import Data, InMemoryDataset, DataLoader
import torch
from multiprocessing import Pool
import os
import shutil

class Semantics:
    def __init__(self):
        pass
    
    def sample(self, seed=None):
        raise NotImplementedError() # This is the base class, not implemented

    def check(self, p: FactBase):
        raise NotImplementedError()

def sample_data(semantics, i, total, directory):
    print(f"Generating sample {i+1}/{total}")
    seed = os.getpid() + i * 1024

    program_sample = semantics.sample(seed) # is using "ConfiguredBgpSemantics.sample()" in the other script, instead of "Semantics.sample()" above
    data = program_sample.to_data()
    filepath = directory + "/graph%07d.pt" % i
    torch.save(data, filepath)
    return filepath

class FactBaseSemanticsDataset(InMemoryDataset):
    def __init__(self, semantics, root, num_samples=256, tmp_directory="tmp-dataset", transform=None, pre_transform=None):
        self.semantics = semantics # save the "ConfiguredBgpSemantics" object
        self.num_samples = num_samples
        self.tmp_directory = tmp_directory

        if not os.path.exists(tmp_directory): os.mkdir(tmp_directory)

        super().__init__(root, transform, pre_transform) # !!!! calling the parent class, therefore, calling "InMemoryDataset.__init__()", which will call self.process in PyTorch Geometric
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [] # this explains why the folder "raw" is empty;

    @property
    def processed_file_names(self):
        return ['datapoint-processed.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        pool = Pool(16) # create 16 parallel processes
        data_tasks = []
        for i in range(self.num_samples):
            data_tasks.append(pool.apply_async(sample_data, (self.semantics, i, self.num_samples, self.tmp_directory)))
        data_files = [task.get() for task in data_tasks]
        data_list = list(map(lambda f: torch.load(f), data_files))

        def make_data(d):
            if type(d) is Data: return d
            elif type(d) is dict: return Data(**d)
            else: assert False, "Unknown data type {}".format(type(d))

        data_list = [make_data(d) for d in data_list]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])