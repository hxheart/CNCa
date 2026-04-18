import sys
sys.path.append("../../dataset")
sys.path.append("../../model")
import os

from multiprocessing import Pool

from numpy.core.fromnumeric import choose
from nutils import choose_random
from bgp_semantics import BgpRoute, BgpSemantics, compute_forwarding_state, draw_graph
from networkx.readwrite import generate_graphml, parse_graphml
import json
from factbase import Constant, FactBase
import argparse
import imageio
import torch
import networkx as nx
import numpy as np

MAX_WEIGHT = 32

def generate_sample(topo, protocol, problem_index, unsat_sample_index, num_unsat_req):
    s = BgpSemantics()
    d = torch.load(topo)
    fbase = FactBase.from_data(d)

    graph, _ = s.network(fbase)

    # generate path requirements
    networks = [n for n in graph.nodes() if graph.nodes[n]["type"] == "network"]
    router_nodes = [n for n in graph.nodes() if graph.nodes[n]["type"] == "router"]

    num_paths = args.num_reqs
    fwd_paths = []
    num_unsat = 0

    for i in range(num_paths):
        # use different link weights for last requirement (to provoke unsat path requirements)
        if i == 0 or i >= num_paths - num_unsat_req:
            num_unsat += 1 if i != 0 else 0
            for src, tgt in graph.edges(): 
                weight = np.random.randint(1, MAX_WEIGHT)
                if "weight" in graph[src][tgt].keys():
                    graph[src][tgt]["weight"] = weight
                    graph[tgt][src]["weight"] = weight
            compute_forwarding_state(graph)

        net = networks[0]
        node = choose_random(router_nodes)

        path = []
        
        while graph.nodes[node]["type"] == "router":
            path.append(node)
            src = node
            dst = node
            for n in graph.neighbors(node):
                if "is_forwarding" in graph[node][n].keys():
                    if net in graph[node][n]["is_forwarding"].keys():
                        dst = n
                        break
            if len(list(graph.neighbors(node))) == 0 or src == dst:
                #print("warning: failed to determine forwarding entry at {}".format(src))
                path = []
                node = choose_random(router_nodes)
                continue
            node = dst
            assert graph.nodes[dst]["type"] == "external" or len(fbase.query("connected", src, dst)) > 0 or len(fbase.query("connected", dst, src)) > 0
        path.append(net)
        fwd_paths.append(path)

    spec = [{"type": "path", "path": path} for path in fwd_paths]

    if protocol == "bgp":
        for path in fwd_paths:
            n = path[0]
            t = graph.nodes[n]["type"]
            if t != "router":continue
            net = path[-1]

            next_hop = graph.nodes[n]["next_hop"][net]
            peer = graph.nodes[n]["peer"][net]
            
            path = [n, next_hop]
            if peer != next_hop: path.append(peer)
            path.append(net)

            spec.append({
                "type": "bgp-path",
                "path": path,
            })
            print(path, spec[-1])
    
    for n in graph.nodes():
        if "peer" in graph.nodes[n].keys():
            del graph.nodes[n]["peer"]
        if "next_hop" in graph.nodes[n].keys():
            del graph.nodes[n]["next_hop"]

    for req in spec:
        if req["type"] != "path": continue
        path = req["path"]
        net_node = path[-1]
        net = graph.nodes[net_node]["label"]
        for src, dst in zip(path[0:-2], path[1:-1]):
            c_src = graph.nodes[src]["label"]
            c_dst = graph.nodes[dst]["label"]
            fbase.add_fact("fwd", False, *[Constant(c) for c in [c_src, net, c_dst]])

    print("Generated progam with spec gives consistency:", s.check(fbase))
    print("Num unsat paths ", num_unsat)

    for src,dst in graph.edges():
        if "is_forwarding" in graph[src][dst].keys():
            is_forwarding = graph[src][dst]["is_forwarding"]
            graph[src][dst]["is_forwarding"] = json.dumps(is_forwarding)

    for n in graph.nodes():
        if "bgp_route" in graph.nodes[n].keys():
            route: BgpRoute = graph.nodes[n]["bgp_route"]
            graph.nodes[n]["bgp_route"] = json.dumps({
                "origin": route.origin_type,
                "as_path_len": route.as_path_length,
                "next_hop": route.next_hop,
                "local_pref": route.local_preference,
                "med": route.med
            })

    dataset_dir = args.output_directory

    d = fbase.to_data()
    torch.save(d, "{}/{}-n{}-unsatsample{}.logic".format(dataset_dir, protocol, problem_index, unsat_sample_index))

    with open("{}/{}-n{}-unsatsample{}.graphml".format(dataset_dir, protocol, problem_index, unsat_sample_index), "w") as f:
        for line in generate_graphml(graph):
            f.write(line)

    with open("{}/{}-n{}-unsatsample{}.spec.json".format(dataset_dir, protocol, problem_index, unsat_sample_index), "w") as f:
        json.dump(spec, f)

    #print(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate topology-zoo-based synthesis problem instances.')
    parser.add_argument('num_reqs', type=int, help='the number of path requirements')
    parser.add_argument('output_directory', type=str, default=max, help='the directory to save the files in')
    parser.add_argument('dataset', type=str, help='base directory of the topology dataset')
    parser.add_argument('--protocol', type=str, default="ospf", help='the protocol to use')
    parser.add_argument('--num_unsat_req', type=int, help='the number of likely unsat requirements to include.')

    args = parser.parse_args()

    assert args.protocol == "ospf" or args.protocol == "bgp", "unsupported protocol {}".format(args.protocol)

    topologies = [f"{args.dataset}/small/" + f for f in os.listdir(f"{args.dataset}/small")] + \
        [f"{args.dataset}/medium/" + f for f in os.listdir(f"{args.dataset}/medium")] + \
        [f"{args.dataset}/large/" + f for f in os.listdir(f"{args.dataset}/large")]

    pool = Pool()

    res = []
    for i,t in enumerate(topologies):
        print("Generating sample", i)
        # generate 4 different unsat candidates per sample in dataset
        for j in range(10):
            res.append(pool.apply_async(generate_sample, (t, args.protocol, i, j, args.num_unsat_req)))
    for r in res:
        print(r.get())
        
        