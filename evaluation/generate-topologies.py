import sys
sys.path.append("../dataset")
sys.path.append("../model")
import os

from numpy.core.fromnumeric import choose
from nutils import choose_random
from bgp_semantics import BgpRoute, BgpSemantics, compute_forwarding_state, draw_graph
from dataset.topologies import all_topology_files
from networkx.readwrite import generate_graphml, parse_graphml
import json
from factbase import Constant, FactBase
import argparse
import imageio
import torch
import networkx as nx
from tqdm import tqdm

#print(all_topology_files)

for i, topo_file in tqdm(enumerate(all_topology_files), total=len(all_topology_files)):
    try:
        s = BgpSemantics()
        res = s.sample(real_world_topology=True, use_topology_file=topo_file, num_networks=4)
        torch.save(res, f"networks/net{i:03d}.logic")
    except Exception as e:
        print("Problem with topology {}: {}".format(topo_file, e))
        pass