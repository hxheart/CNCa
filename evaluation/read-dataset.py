import sys

sys.path.append("../dataset")
sys.path.append("../model")

from factbase import FactBase
import argparse
import os
import torch
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate topology-zoo-based synthesis problem instances.')
    parser.add_argument('dataset', type=str, help='base directory of the topology dataset')

    args = parser.parse_args()

    if os.path.exists(os.path.join(args.dataset, "small")):
        topologies = [f"{args.dataset}/small/" + f for f in os.listdir(f"{args.dataset}/small")] + \
            [f"{args.dataset}/medium/" + f for f in os.listdir(f"{args.dataset}/medium")] + \
            [f"{args.dataset}/large/" + f for f in os.listdir(f"{args.dataset}/large")]
    else:
        topologies = [f"{args.dataset}/" + f for f in os.listdir(args.dataset)]

    topology_sizes = []

    pbar = tqdm(enumerate(topologies), total=len(topologies))
    for i, t in pbar:
        if not t.endswith(".logic"): continue
        pbar.set_description(t)
        data = torch.load(t)
        factbase = FactBase.from_data(data)
        routers = factbase.query("router")
        rrs = factbase.query("route_reflector")
        topology_sizes.append(len(routers) + len(rrs))
    
    n = list(sorted(list(topology_sizes)))

    if len(n) > 8:
        print("Small", n[0], "-", n[8])
    if len(n) > 16:
        print("Medium", n[8], "-", n[16])
    if len(n) >= 24:
        print("Medium", n[16], "-", n[23] + 1)