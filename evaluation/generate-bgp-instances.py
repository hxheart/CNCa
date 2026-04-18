"""
Generates BGP problem instances given a directory of small/medium/large topologies.
"""

import sys
sys.path.append("../dataset")
sys.path.append("../model")
import os

from numpy.core.fromnumeric import choose
from nutils import choose_random
from bgp_semantics import BgpRoute, BgpSemantics, compute_forwarding_state, draw_graph
from networkx.readwrite import generate_graphml, parse_graphml
import json
from program import Constant
import argparse
import imageio
import torch
import networkx as nx
from program import Program
import numpy as np

def generate_sample(topo, protocol, problem_index):
    s = BgpSemantics()
    d = torch.load(topo)
    prog = Program.from_data(d)
    graph, _ = s.network(prog)

    NUM_NODES = len(graph.nodes())
    NUM_ROUTE_REFLECTORS = max(2, int(NUM_NODES / 5))
    
    # remove existing ibgp edges
    for src,dst in list(graph.edges()):
        t = graph[src][dst]["type"]
        if t == "ospf": continue
        if t == "ebgp": continue
        if t == "network": continue
        assert t == "ibgp"

        if "weight" in graph[src][dst].keys(): 
            graph[src][dst]["type"] = "ospf"
            continue
        graph.remove_edge(src, dst)
    
    # switch to RR-based BGP session layout
    # add route reflector nodes
    router_nodes = [n for n in graph.nodes() if graph.nodes[n]["type"] == "router"]
    route_reflector_nodes = set()
    
    for n in range(NUM_ROUTE_REFLECTORS):
        node_id = choose_random(router_nodes)
        route_reflector_nodes.add(node_id)
        graph.add_node(node_id)
        graph.nodes[node_id]["type"] = "route_reflector"

    # fully mesh route reflectors
    for rr in route_reflector_nodes:
        for other_rr in route_reflector_nodes:
            if rr == other_rr: continue
            graph.add_edge(rr, other_rr, type="ibgp")
            graph.add_edge(other_rr, rr, type="ibgp")

    # every router node is connected to one of the route reflectors
    for r in router_nodes:
        rr = choose_random(list(route_reflector_nodes))
        graph.add_edge(r, rr, type="ibgp")
        graph.add_edge(rr, r, type="ibgp")

    weights = np.random.random([3])
    weights /= weights.sum()

    print(len(router_nodes), len([n for n in graph.nodes() if graph.nodes[n]["type"] == "router"]))

    prog_with_spec = s.sample(basedOnGraph=graph, predicate_semantics_sample_config_overrides={
        "fwd": {
            "n": args.num_reqs,
            "per_network": False
        },
        "reachable": {
            "n": args.num_reqs,
            "per_network": False
        },
        "trafficIsolation": {
            "n": args.num_reqs
        }
    })
    print("fwd", len(prog_with_spec.query("fwd")), 
        "reachable", len(prog_with_spec.query("reachable")),
        "trafficIsolation", len(prog_with_spec.query("trafficIsolation"))
    )

    dataset_dir = args.output_directory

    d = prog_with_spec.to_data()
    torch.save(d, "{}/{}-n{}.logic".format(dataset_dir, protocol, problem_index))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate topology-zoo-based synthesis problem instances.')
    parser.add_argument('num_reqs', type=int, help='the number of path requirements')
    parser.add_argument('output_directory', type=str, default=max, help='the directory to save the files in')
    parser.add_argument('dataset', type=str, help='base directory of the topology dataset')
    parser.add_argument('--protocol', type=str, default="ospf", help='the protocol to use')

    args = parser.parse_args()

    assert args.protocol == "ospf" or args.protocol == "bgp", "unsupported protocol {}".format(args.protocol)

    topologies = [f"{args.dataset}/small/" + f for f in os.listdir(f"{args.dataset}/small")] + \
        [f"{args.dataset}/medium/" + f for f in os.listdir(f"{args.dataset}/medium")] + \
        [f"{args.dataset}/large/" + f for f in os.listdir(f"{args.dataset}/large")]

    for i, t in enumerate(topologies):
        print("Generating sample", i)
        generate_sample(t, args.protocol, i)