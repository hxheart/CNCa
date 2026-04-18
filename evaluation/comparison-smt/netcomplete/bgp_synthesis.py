#!/usr/bin/env python

"""
An Simple example of an AS with two providers and one customer
The policy is such that the customer traffic prefer once provider over the other
And providers cannot use the network as transit.
"""

import argparse
import logging
from ipaddress import ip_address, ip_interface
from ipaddress import ip_network
from synet.utils.fnfree_smt_context import SolverContext
from synet.utils.bgp_utils import compute_next_hop_map, extract_all_next_hops
from synet.synthesis.connected import ConnectedSyn

from networkx.generators.joint_degree_seq import _neighbor_switch

import datetime
import multiprocessing
from multiprocessing import TimeoutError

from synet.utils.common import VERTEX_TYPE, PathReq
from synet.utils.common import PathOrderReq
from synet.utils.common import KConnectedPathsReq
from synet.utils.common import Protocols
from tekton.graph import VERTEXTYPE

from tekton.utils import VALUENOTSET

from tekton.bgp import BGP_ATTRS_ORIGIN
from tekton.bgp import RouteMapLine
from tekton.bgp import RouteMap
from tekton.bgp import Announcement, Access
from tekton.bgp import Community, CommunityList, MatchCommunitiesList
from tekton.bgp import IpPrefixList, MatchIpPrefixListList, MatchNextHop, MatchSelectOne, ActionSetLocalPref
from tekton.bgp import ActionSetCommunity
from synet.netcomplete import NetComplete
from synet.utils.topo_gen import gen_mesh

from tekton.graph import NetworkGraph

import json
from networkx.readwrite import read_graphml
import networkx as nx

import itertools

from timeit import default_timer as timer

def setup_logging():
    # create logger
    logger = logging.getLogger('synet')
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

def read_spec(filename):
    with open(filename, "r") as f:
        spec = json.load(f)
    return spec

def gen_from_graph(filename, asnum=None):
    g = read_graphml(filename)
    g = sanitise_names(g)

    comms = [Community("100:{}".format(c)) for c in range(1, 4)]
    announcements = []
    
    class NetworkInfo:
        def __init__(self, n):
            self.node_id = n
            self.ip_net = ip_network(u'128.0.{}.0/24'.format(int(n[1:])))
            self.prefix = str(self.ip_net)
        def __repr__(self):
            return "NetworkInfo(id={}, subnet={})".format(self.node_id, self.ip_net)

    network_nodes = [n for n in g.nodes() if g.nodes[n]["type"] == "network"]
    networks = dict([(net,NetworkInfo(net)) for net in network_nodes])

    ngraph = NetworkGraph()
    for node in g.nodes():
        t = g.nodes[node]["type"]

        if t == "router":
            ngraph.add_router(node)
            ngraph.set_bgp_asnum(node, asnum)
            print(node, ngraph.nodes[node])

            ngraph.enable_ospf(node, 100)

            ngraph.set_loopback_addr(node, 'lo100', VALUENOTSET)
        elif t == "external":
            network_node = [n for n in g.neighbors(node) if g.nodes[n]["type"] == "network"][0]
            neighbor_asnum = int(network_node[1:])
            ngraph.add_peer(node)
            print("external node {} asnum is {}".format(node, neighbor_asnum))
            ngraph.set_bgp_asnum(node, neighbor_asnum)
        elif t == "route_reflector": assert False, "route reflectors are not supported"
        elif t == "network": pass
        else: assert False, "skipping node {} of type {}".format(node, t)

    for src,dst in g.edges():
        t = g[src][dst]["type"]
        if t == "ospf" or ("weight" in g[src][dst].keys() and t == "ibgp"):
            ngraph.add_router_edge(src, dst, type=t)
            ngraph.set_edge_ospf_cost(src, dst, g[src][dst]["weight"])
        if t == "ebgp":
            ngraph.add_peer_edge(src, dst, type=t)

            if dst not in ngraph.get_bgp_neighbors(src):
                ngraph.add_bgp_neighbor(src,dst)
        if t == "ibgp":
            if src == dst: continue
            #if dst not in ngraph.get_bgp_neighbors(src):
            #ngraph.add_bgp_neighbor(router_a=src, router_b=dst)
        if t == "network": pass
        #else: assert False, "skipping edge {} of type {}".format([src,dst], t)

    for n1 in ngraph.local_routers_iter():
        for n2 in ngraph.local_routers_iter():
            if n1 == n2: continue
            if n2 not in ngraph.get_bgp_neighbors(n1):
                ngraph.add_bgp_neighbor(n1,n2)

    partially_evaluated = {}
    
    # add externally-advertised routes
    for node in g.nodes():
        t = g.nodes[node]["type"]
        if not t == "external": continue

        network_node = [n for n in g.neighbors(node) if g.nodes[n]["type"] == "network"][0]
        info = networks[network_node]
        route = json.loads(g.nodes[node]["bgp_route"])
        
        origins = [
            BGP_ATTRS_ORIGIN.IGP,
            BGP_ATTRS_ORIGIN.EBGP,
            BGP_ATTRS_ORIGIN.INCOMPLETE
        ]

        ann = Announcement(str(info.ip_net),
                    peer=node,
                    origin=origins[route["origin"]],
                    as_path=[int(info.node_id[1:])],  # We assume it learned from other upstream ASes
                    as_path_len=route["as_path_len"],
                    next_hop='{}Hop'.format(node),
                    local_pref=route["local_pref"],
                    med=route["med"],
                    communities=dict([(c, False) for c in comms]),
                    permitted=True)
        announcements.append(ann)
        print(node, "announces", route, "to", str(info.ip_net))

        ngraph.add_bgp_advertise(node, ann, loopback='lo100')
        ngraph.set_loopback_addr(node, 'lo100', ip_interface(info.ip_net.hosts().next()))

        local_nodes = [n for n in g.neighbors(node) if g.nodes[n]["type"] == "router"]
        for local in local_nodes:
            if "peer" not in ngraph.nodes[local].keys():
                ngraph.nodes[local]["peer"] = []
            ngraph.nodes[local]["peer"].append([node, str(info.ip_net)]) 

    for node in ngraph.local_routers_iter():
        ngraph.add_ospf_network(node, 'lo100', '0.0.0.0')
        ngraph.add_ospf_network(node, 'Fa0/0', '0.0.0.0')
        
        for neighbor in ngraph.get_bgp_neighbors(node):
            imp_name = "{}_import_from_{}".format(node, neighbor)
            exp_name = "{}_export_to_{}".format(node, neighbor)
            imp = RouteMap.generate_symbolic(name=imp_name, graph=ngraph, router=node)
            exp = RouteMap.generate_symbolic(name=exp_name, graph=ngraph, router=node)

            ngraph.add_bgp_import_route_map(node, neighbor, imp.name)
            ngraph.add_bgp_export_route_map(node, neighbor, exp.name)

    return ngraph, announcements, networks

def sanitise_node_name(n):
    return "n{}".format(n)

def sanitise_names(g):
    mapping = dict([(n,sanitise_node_name(n)) for n in g.nodes()])
    return nx.relabel_nodes(g, mapping)

def draw(g):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    pos = nx.drawing.layout.spring_layout(g, weight=None)

    fig = plt.Figure(figsize=(12,12))
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    #plt.axis('off')
        
    def node_label(n): return str(n)

    #nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, ax=ax)
    nx.draw_networkx_labels(g, labels=dict([(n, node_label(n)) for n in g.nodes()]), pos=pos, ax=ax)
    nx.draw(g, ax=ax, arrowsize=20, pos=pos)

    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(canvas.get_width_height()[::-1] + (3,))

    import imageio

    imageio.imwrite('result.png', image)

def run_bgp(output_dir, prefix):
    graph, external_anns, networks = gen_from_graph(prefix + ".graphml", 101)

    # edge weights are symbolic
    for src, dst in graph.edges():
        if graph[src][dst]["type"] == "ospf":
            graph.set_edge_ospf_cost(src, dst, VALUENOTSET)

    ########################## Configuration sketch ###############################
    # Requirements
    spec = read_spec(prefix + ".spec.json")
    reqs = []

    for line in spec:
        t = line["type"]
        if t == "path":
            path = line["path"]
            net = sanitise_node_name(path[-1])
            prefix = str(networks[net].ip_net)
            path = path[:-1]

            reqs.append(PathReq(
                Protocols.OSPF, # all requirements must be BGP
                prefix, # prefix
                [sanitise_node_name(p) for p in path],
                False
            ))
            print("ospf req", line["path"], prefix)
        elif t == "bgp-path":
            path = line["path"]
            net = sanitise_node_name(path[-1])
            prefix = str(networks[net].ip_net)
            path = path[:-1]

            reqs.append(PathReq(
                Protocols.BGP, # all requirements must be BGP
                prefix, # prefix
                [sanitise_node_name(p) for p in path],
                False
            ))
        else: assert False, "unhandled spec predicate {}".format(t)

    draw(graph)

    assert len(reqs) > 0, "no valid requirements found"

    csyn = ConnectedSyn([], graph)
    csyn.synthesize()

    for router in graph.local_routers_iter():
        for iface in graph.get_ifaces(router):
            graph.add_ospf_network(router, iface, 0)
        for lo in graph.get_loopback_interfaces(router):
            graph.add_ospf_network(router, lo, 0)

        if "peer" not in graph.nodes[router].keys(): continue
        peers = graph.nodes[router]["peer"]
        ifaces_by_neighbor = dict([(i[1]["description"].split(" ")[1], i[0]) for i in graph.get_ifaces(router).items()])
        for peer, net in peers:
            iface = ifaces_by_neighbor[peer]
            graph.add_ospf_network(router, net, iface)

    t1 = timer()
    netcomplete = NetComplete(reqs=reqs, topo=graph, external_announcements=external_anns)
    netcomplete.synthesize()
    t2 = timer()
    total = t2 - t1

    return total


if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser(description='BGP performance NetComplete evaluation.')
    parser.add_argument('outdir', type=str, help='output directory for the configuration')
    parser.add_argument('--dataset', type=str, default="../nsynth/predicate/bgp-dataset", help='dataset directory')
    parser.add_argument('--timeout', type=int, default=5*60, help='timeout limit')
    parser.add_argument('--perf-prefix', dest='perfprefix', type=str, default="",
                    help='Prefix to use when writing out the performance measurement results')
    args = parser.parse_args()

    import os
    import sys
    dataset_dir = args.dataset

    result = args.perfprefix + 'result-bgp-{date:%Y-%m-%d_%H:%M:%S}.csv'.format( date=datetime.datetime.now() )
    f = open(result, "w")
    f.close()

    num_nodes = sorted([int(f[len("bgp-n"):-len(".graphml")]) for f in os.listdir(dataset_dir) if f.endswith(".graphml")])

    pool = multiprocessing.Pool(processes=1)

    for n in num_nodes:
        instance = "bgp-n{}".format(n)
        prefix = dataset_dir + "/" + instance

        ttotal = "ERR"        
        try:
            p = pool.apply_async(run_bgp, args=[args.outdir, prefix])
            ttotal = p.get(timeout=args.timeout)
        except TimeoutError:
            ttotal = 999
            print("Timed out after {}s".format(args.timeout), prefix)

        with open(result, "a") as f:
            f.write("{};{}\n".format(instance, ttotal))
