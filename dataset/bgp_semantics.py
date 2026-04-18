
import os
import sys

from networkx.algorithms.components.connected import is_connected
from numpy.random import f
sys.path.append(os.path.dirname(__file__)+"/../")

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
from torch_geometric.data import Data
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

from factbase import PredicateAst
from factbase import *
from semantics import FactBaseSemanticsDataset, Semantics
import topologies
from nutils import prop, choose_random

from predicate_semantics import *

"""
This file contains the protocol semantics for BGP/OSPF. Using the BgpSemantics class
one can generate a dataset of fact bases encoding BGP/OSPF synthesis input/output examples.

This file also includes the simulation code required to simulate BGP/OSPF.
"""

def random_planar_graph(num_nodes, random_state):
    pos = random_state.rand(num_nodes,2)
    simps = Delaunay(pos).simplices
    G = nx.DiGraph()

    def add_edge(src, dst): 
        G.add_edge(src, dst)
        G.add_edge(dst, src)

    for tri in simps:        
        add_edge(tri[0], tri[1])
        add_edge(tri[1], tri[2])
        add_edge(tri[2], tri[0])

    return G, pos

class BgpRoute(object):
    def __init__(self, destination, local_preference, as_path_length, origin_type, med, is_ebgp_learned, bgp_speaker_id, next_hop):
        self.destination = destination
        self.local_preference = local_preference
        self.as_path_length = as_path_length
        self.origin_type = origin_type
        self.med = med
        self.is_ebgp_learned = is_ebgp_learned
        self.bgp_speaker_id = bgp_speaker_id
        self.next_hop = next_hop

        self.igp_costs = None

    def copy(self):
        return BgpRoute(self.destination, self.local_preference, self.as_path_length, self.origin_type,
                        self.med, self.is_ebgp_learned, self.bgp_speaker_id, self.next_hop)
    def __repr__(self):
        return f"<BgpRoute destination={self.destination} LOCAL_PREFERENCE=" + str(self.local_preference) + \
               " AS_PATH_LENGTH=" + str(self.as_path_length) + \
               " ORIGIN_TYPE=" + str(self.origin_type) + \
               " MED=" + str(self.med) + \
               " IS_EBGP_LEARNED=" + str(self.is_ebgp_learned) + \
               " BGP_SPEAKER_ID=" + str(self.bgp_speaker_id) + \
               " NEXT_HOP=" + str(self.next_hop) + ">"
    # hash and compare by speaker ID only (TODO: what if more than one announcement per speaker e.g. redistributing IGP)
    def __eq__(self, other):
        if other is None:
            return False
        return self.bgp_speaker_id == other.bgp_speaker_id and self.destination == other.destination

    def __hash__(self):
        return 10000 * self.bgp_speaker_id + self.destination

BgpRoute.ORIGIN_IGP = 0
BgpRoute.ORIGIN_EGP = 1
BgpRoute.ORIGIN_INCOMPLETE = 2

def generate_random_route_announcement(destination, LOCAL_PREF=None, AS_PATH_LENGTH=None, ORIGIN=None, MED=None, IS_EBGP_LEARNED=None):
    LOCAL_PREF = LOCAL_PREF if LOCAL_PREF is not None else np.random.randint(0,10)
    AS_PATH_LENGTH = AS_PATH_LENGTH if AS_PATH_LENGTH is not None else np.random.randint(1,10)
    ORIGIN = ORIGIN if ORIGIN is not None else choose_random([BgpRoute.ORIGIN_IGP, BgpRoute.ORIGIN_EGP, BgpRoute.ORIGIN_INCOMPLETE])
    MED = MED if MED is not None else choose_random([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    IS_EBGP_LEARNED = choose_random([True, False]) if IS_EBGP_LEARNED is None else IS_EBGP_LEARNED

    return BgpRoute(destination, LOCAL_PREF, AS_PATH_LENGTH, ORIGIN, MED, IS_EBGP_LEARNED, 0, 0)

def lowest(routes, prop_fct):
    v = float("inf")
    for r in routes: v = min(v, prop_fct(r))
    return v

def generate_random_route_announcements(destination, ROUTES_PER_CATEGORY=2, ROUTES_IN_LAST_CATEGORY=3):
    announcements = []
    for i in range(ROUTES_PER_CATEGORY):
        LP = np.random.randint(0,10)
        announcements.append(generate_random_route_announcement(destination, LOCAL_PREF=LP))
    LOCAL_PREF = -lowest(announcements, lambda r: -r.local_preference)

    for i in range(ROUTES_PER_CATEGORY):
        APL = np.random.randint(1,10)
        announcements.append(generate_random_route_announcement(destination, LOCAL_PREF=LOCAL_PREF, AS_PATH_LENGTH=APL))
    AS_PATH_LENGTH = lowest(announcements, lambda r: r.as_path_length)

    for i in range(ROUTES_PER_CATEGORY):
        O = choose_random([BgpRoute.ORIGIN_IGP, BgpRoute.ORIGIN_EGP, BgpRoute.ORIGIN_INCOMPLETE])
        announcements.append(generate_random_route_announcement(destination, LOCAL_PREF=LOCAL_PREF, AS_PATH_LENGTH=AS_PATH_LENGTH, ORIGIN=O))
    ORIGIN = lowest(announcements, lambda r: r.origin_type)

    for i in range(ROUTES_PER_CATEGORY):
        M = np.random.randint(0,30)
        announcements.append(generate_random_route_announcement(destination, LOCAL_PREF=LOCAL_PREF, AS_PATH_LENGTH=AS_PATH_LENGTH, ORIGIN=ORIGIN, MED=M))
    MED = lowest(announcements, lambda r: r.med)

    for i in range(ROUTES_IN_LAST_CATEGORY):
        announcements.append(generate_random_route_announcement(destination, LOCAL_PREF=LOCAL_PREF, AS_PATH_LENGTH=AS_PATH_LENGTH, ORIGIN=ORIGIN, MED=MED))

    return announcements

def prop(e,prop):
    if prop not in e.keys(): return None
    return e[prop]

def generate_graph_with_topology(filename, seed, NUM_NODES=None, NUM_NETWORKS=None, NUM_GATEWAY_NODES=None, NUM_ROUTE_REFLECTORS=None, FULLY_MESHED=False):
    MAX_WEIGHT = 32

    s = np.random.RandomState(seed=seed)
    graph = topologies.read_topology(filename)
    NUM_NODES = len(graph.nodes())

    print("generate sample based on topology", filename)

    if NUM_NETWORKS is None: NUM_NETWORKS = 1
    if NUM_GATEWAY_NODES is None: NUM_GATEWAY_NODES = choose_random([2,3,7], s)
    if NUM_ROUTE_REFLECTORS is None: NUM_ROUTE_REFLECTORS = 2

    # set node types
    nx.set_node_attributes(graph, "router", name="type")

    # initialise link weights
    for src, tgt in graph.edges(): 
        weight = s.randint(1, MAX_WEIGHT)
        graph[src][tgt]["weight"] = weight
        graph[tgt][src]["weight"] = weight

    nx.set_edge_attributes(graph, "ospf", name="type")
    
    router_nodes = set(graph.nodes())
    
    # add network nodes
    network_nodes = set()
    routes_per_network = {}
    
    for n in range(NUM_NETWORKS):
        node_id = len(graph.nodes())
        network_nodes.add(node_id)
        routes_per_network[node_id] = generate_random_route_announcements(node_id)
        graph.add_node(node_id, type="network")
    
    # BGP
    
    ## configure gateway nodes
    ebgp_nodes = set()
    for network_node in network_nodes:
        for n in range(NUM_GATEWAY_NODES):
            ebgp_node = len(graph.nodes())
            gateway_node = s.randint(0, NUM_NODES)
            graph.add_node(ebgp_node, type="external")
            ebgp_nodes.add(ebgp_node)
            
            # connect gateway and ebgp node
            graph.add_edge(ebgp_node, gateway_node, type="ebgp")
            graph.add_edge(gateway_node, ebgp_node, type="ebgp")
            # connect ebgp node and network
            graph.add_edge(ebgp_node, network_node, type="network")
            graph.add_edge(network_node, ebgp_node, type="network")
            
            # choose route to be advertised via gateway
            bgp_route = choose_random(routes_per_network[network_node], s).copy()
            bgp_route.bgp_speaker_id = ebgp_node
            bgp_route.is_ebgp_learned = True
            graph.nodes[ebgp_node]["bgp_route"] = bgp_route
    
    if not FULLY_MESHED:
        # add route reflector nodes
        route_reflector_nodes = set()
        for n in range(NUM_ROUTE_REFLECTORS):
            node_id = s.randint(0, NUM_NODES)
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
            rr = choose_random(list(route_reflector_nodes), s)
            graph.add_edge(r, rr, type="ibgp")
            graph.add_edge(rr, r, type="ibgp")
    else:
        for r1 in router_nodes:
            for r2 in router_nodes:
                if r1 == r2: continue
                graph.add_edge(r1, r2, type="ibgp")
                graph.add_edge(r2, r1, type="ibgp")
        print("generating fully meshed ibgp session layout")
        
    return graph

def generate_graph(seed, NUM_NODES=None, NUM_NETWORKS=None, NUM_GATEWAY_NODES=None, NUM_ROUTE_REFLECTORS=None, FULLY_MESHED=False):
    if NUM_NODES is None: NUM_NODES = 10
    if NUM_NETWORKS is None: NUM_NETWORKS = 2
    if NUM_GATEWAY_NODES is None: NUM_GATEWAY_NODES = min(int(NUM_NODES / 2), 4)
    if NUM_ROUTE_REFLECTORS is None: NUM_ROUTE_REFLECTORS = 2
    MAX_WEIGHT = 32

    s = np.random.RandomState(seed=seed)
    graph, pos = random_planar_graph(NUM_NODES, s)

    # set node types
    nx.set_node_attributes(graph, "router", name="type")

    # initialise link weights
    for src, tgt in graph.edges(): 
        weight = s.randint(1, MAX_WEIGHT)
        graph[src][tgt]["weight"] = weight
        graph[tgt][src]["weight"] = weight

    nx.set_edge_attributes(graph, "ospf", name="type")
    
    router_nodes = set(graph.nodes())
    
    # add network nodes
    network_nodes = set()
    routes_per_network = {}
    
    for n in range(NUM_NETWORKS):
        node_id = len(graph.nodes())
        network_nodes.add(node_id)
        routes_per_network[node_id] = generate_random_route_announcements(node_id)
        graph.add_node(node_id, type="network")
    
    # BGP
    ## configure gateway nodes
    ebgp_nodes = set()
    for network_node in network_nodes:
        for n in range(NUM_GATEWAY_NODES):
            ebgp_node = len(graph.nodes())
            gateway_node = s.randint(0, NUM_NODES)
            graph.add_node(ebgp_node, type="external")
            ebgp_nodes.add(ebgp_node)
            
            # connect gateway and ebgp node
            graph.add_edge(ebgp_node, gateway_node, type="ebgp")
            graph.add_edge(gateway_node, ebgp_node, type="ebgp")
            # connect ebgp node and network
            graph.add_edge(ebgp_node, network_node, type="network")
            graph.add_edge(network_node, ebgp_node, type="network")
            
            # choose route to advertised by gateway
            bgp_route = choose_random(routes_per_network[network_node], s).copy()
            bgp_route.bgp_speaker_id = ebgp_node
            bgp_route.is_ebgp_learned = True
            graph.nodes[ebgp_node]["bgp_route"] = bgp_route

    if FULLY_MESHED:
        for r1 in router_nodes:
            for r2 in router_nodes:
                if r1 == r2: continue
                graph.add_edge(r1, r2, type="ibgp")
                graph.add_edge(r2, r1, type="ibgp")
        print("generating fully meshed ibgp session layout")
    else:
        # add route reflector nodes
        route_reflector_nodes = set()
        for n in range(NUM_ROUTE_REFLECTORS):
            node_id = s.randint(0, NUM_NODES)
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
            rr = choose_random(list(route_reflector_nodes), s)
            graph.add_edge(r, rr, type="ibgp")
            graph.add_edge(rr, r, type="ibgp")
        
    return graph

def draw_graph(G, pos=None, destination=-1, figsize=(10,10), label="", use_node_labels=True):
    G = G.copy()

    pos = pos if pos is not None else nx.drawing.layout.spring_layout(G, weight=None)

    labels = {}
    edge_color = []
    node_color = []
    
    for src, tgt in list(G.edges()):
        is_forwarding = destination in G[src][tgt]["is_forwarding"].keys() \
            if destination != -1 and prop(G[src][tgt], "is_forwarding") is not None \
            else destination == -1
        if not is_forwarding and G[src][tgt]["type"] != "network":
            G.remove_edge(src, tgt)

    for src, tgt in G.edges():
        edge_type = G[src][tgt]["type"]
        is_ospf_link = "weight" in G[src][tgt].keys()
        is_bgp_link = edge_type == "ibgp" or edge_type == "ebgp"
        is_network_link = edge_type == "network"
        
        if is_ospf_link:
            w = G[src][tgt]["weight"]
            labels[(src,tgt)] = f"{w}"
            edge_color.append("gray")
        elif is_bgp_link:
            labels[(src,tgt)] = edge_type
            edge_color.append("gray")
        elif is_network_link:
            labels[(src,tgt)] = "network"
            edge_color.append("yellow")
        else:
            assert False, f"unknown network graph edge type {edge_type}"
            
    for i in G.nodes():
        if prop(G.nodes[i], "type") == "route_reflector":
            node_color.append("green")
        elif prop(G.nodes[i], "bgp_route") is not None:
            node_color.append("yellow")
        elif prop(G.nodes[i], "type") == "network":
            node_color.append("red")
        else:
            node_color.append("red" if i == destination else "lightblue")

    fig = plt.Figure(figsize=figsize)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    plt.axis('off')
        
    def node_label(n):
        if not use_node_labels: return str(n)
        return prop(G.nodes[n], "label") or str(n)

    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, ax=ax)
    nx.draw_networkx_labels(G, labels=dict([(n, node_label(n)) for n in G.nodes()]), pos=pos, ax=ax)
    nx.draw(G, pos=pos, edge_color=edge_color, node_color=node_color, ax=ax, arrowsize=20, arrowstyle="-|>" if destination != -1 else "-")
    
    ax.set_title(label)

    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    return image.reshape(canvas.get_width_height()[::-1] + (3,)), pos

def top_group_for_attr(routes, fct, lower_is_better=True):
    best_val = float("inf") if lower_is_better else -float("inf")
    for r in routes:
        val = fct(r)
        if lower_is_better and val < best_val:
            best_val = val
        elif not lower_is_better and val > best_val:
            best_val = val

    return list(filter(lambda r: abs(fct(r) - best_val) < 0.00001, routes))

def bgp_select(node_id, routes, dist_to_destination):
    if len(routes) < 1: return None

    # remove duplicate entries
    routes = list(set(routes))

    # LOCAL_PREFERENCE
    highest_local_preference = top_group_for_attr(routes, lambda r: r.local_preference, lower_is_better=False)
    # AS_PATH_LENGTH
    lowest_as_path_length = top_group_for_attr(highest_local_preference, lambda r: r.as_path_length)
    # ORIGIN_TYPE
    best_origin = top_group_for_attr(lowest_as_path_length, lambda r: r.origin_type)
    # MED (skipped for now)
    # external routes over internal routes
    best_external_over_internal = top_group_for_attr(best_origin, lambda r: 0 if r.is_ebgp_learned else 1)
    # IGP cost
    def igp_cost(r): 
        dists = dist_to_destination[r.next_hop]
        if not node_id in dists.keys(): return 9999
        return dists[node_id]
    best_by_igp_cost = top_group_for_attr(best_external_over_internal, igp_cost)
    # BGP speaker ID
    best_by_bgp_speaker_id = top_group_for_attr(best_by_igp_cost, lambda r: r.bgp_speaker_id)
    
    if len(best_by_bgp_speaker_id) > 1:
        print("fatal error: BGP decision criteria did not yield a unique result at node", node_id)
        for r in best_by_bgp_speaker_id:
            print(r)

    assert len(best_by_bgp_speaker_id) <= 1, "fatal error: BGP decision criteria did not yield a unique result"
    if len(best_by_bgp_speaker_id) < 1:
        print("warning: BGP decision criteria filtered all routes")
        return None

    return best_by_bgp_speaker_id[0]

def propagate(g, bgp_state, edges):
    for src, dst in edges:
        src_route = bgp_state[src].outbox
        if src_route is not None: bgp_state[dst].available_routes.add(src_route)
        dst_route = bgp_state[dst].outbox
        if dst_route is not None: bgp_state[src].available_routes.add(dst_route)

def update_node(node_id, node, dist_to_destination):
    available_routes = set(list(node.available_routes) + ([node.locRib] if node.locRib is not None else []))
    best_route: BgpRoute = bgp_select(node_id, available_routes, dist_to_destination)
    
    # continue propagation of best_route if it is better than the current node.adjRibOut
    outbox = None
    if best_route != node.adjRibOut and (best_route.is_ebgp_learned or node.type == "route_reflector"):
        outbox = best_route.copy()
        outbox.next_hop = node_id
        outbox.is_ebgp_learned = False
    return BgpNodeState(node.type, best_route, outbox, outbox or node.adjRibOut, set())

def update(g, bgp_state, dist_to_destination):
    num_updates = 0
    
    for n in g.nodes(): 
        s = bgp_state[n]
        if s.type == "network" or s.type == "external": continue
        before_speaker_id = (bgp_state[n].locRib.bgp_speaker_id if bgp_state[n].locRib is not None else -1)

        bgp_state[n] = update_node(n, s, dist_to_destination)
        
        updated_speaker_id = (bgp_state[n].locRib.bgp_speaker_id if bgp_state[n].locRib is not None else -1)
        if before_speaker_id != updated_speaker_id: num_updates += 1
    return num_updates

from dataclasses import dataclass
@dataclass
class BgpNodeState:
    type: str
    locRib: BgpRoute
    outbox: BgpRoute
    adjRibOut: BgpRoute
    available_routes: set

def compute_forwarding_state(g):
    external_nodes = [n for n in g.nodes() if g.nodes[n]["type"] == "external"]
    networks = [n for n in g.nodes() if g.nodes[n]["type"] == "network"]
    
    ospf_edges = [(src,dst) for src,dst in g.edges() if "weight" in g[src][dst].keys()]
    ebgp_edges = [(src,dst) for src,dst in g.edges() if g[src][dst]["type"] == "ebgp"]

    edge_routers = []
    for e in external_nodes:
        edge_routers += [n for n in g.neighbors(e) if g.nodes[n]["type"] == "router"]

    # initialise forwarding edge attribute
    for src, dst in ospf_edges + ebgp_edges: g[src][dst]["is_forwarding"] = {}
    
    # compute shortest paths
    dist_to_destination = {} # dist_to_destination[D][n] := dist from n to D
    next_router_to_destination = {} # next_router_to_destination[D][n] := next router on shortest path from n to D

    g_ospf = nx.DiGraph()
    for src,dst in ospf_edges:
        g_ospf.add_edge(src, dst, weight=g[src][dst]["weight"])
        g_ospf.add_edge(dst, src, weight=g[src][dst]["weight"])
    for src,dst in ebgp_edges:
        g_ospf.add_edge(src, dst, weight=1)
        g_ospf.add_edge(dst, src, weight=1)

    igp_destinations = external_nodes + edge_routers + [n for n in g.nodes() if g.nodes[n]["type"] == "route_reflector"]

    for destination in igp_destinations:
        preds_per_node, dists = nx.algorithms.dijkstra_predecessor_and_distance(g_ospf, destination) # calculate the OSPF
        dist_to_destination[destination] = dists
        next_router_to_destination[destination] = preds_per_node

    for net in networks:
        # initialise BGP state
        bgp_state = {}
        for n in g.nodes():
            type = g.nodes[n]["type"]
            bgp_route = prop(g.nodes[n], "bgp_route")
            if bgp_route is not None: 
                if bgp_route.destination != net: 
                    bgp_route = None
                else:
                    bgp_route.next_hop = n
            bgp_state[n] = BgpNodeState(type, bgp_route, bgp_route, None, set())

        bgp_edges = [(src,dst) for src,dst in g.edges() if g[src][dst]["type"] == "ibgp" or g[src][dst]["type"] == "ebgp"]
        max_iteration = 10

        def print_state(pseudonymised=False):
            P = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S"]
            pseudo_idx = 0
            pseudos = {}

            for n in sorted(g.nodes()):
                s = bgp_state[n]
                if s.type == "router" or s.type == "route_reflector" or s.type == "external": 
                    route_str = str(s.locRib)
                    if s.locRib is not None and pseudonymised:
                        route_str = s.locRib.bgp_speaker_id
                        if route_str not in pseudos: 
                            pseudos[route_str] = P[pseudo_idx]
                            pseudo_idx += 1
                            assert pseudo_idx < len(P), "not enough pseudonyms"
                        route_str = pseudos[route_str]

        num_iterations = 0
        num_changes = 1

        while num_changes > 0:
            if max_iteration <= num_iterations: break
            num_iterations += 1

            propagate(g, bgp_state, bgp_edges) # BGP routing
            num_changes = update(g, bgp_state, dist_to_destination) # updates of BGP routing
        # print(f"BGP simulation finished after {num_iterations} iterations")

        for n in g.nodes():
            s = bgp_state[n]
            type = g.nodes[n]["type"]
            if type != "router" and type != "route_reflector": continue

            if s.locRib is None:
                #print(f"warning: network {net} is not reachable from node {n}")    
                continue
            #assert s.locRib is not None, f"node {n} has not received any of the advertised BGP routes for network {net}"
            
            # store info on next hop per node
            if "next_hop" not in g.nodes[n].keys(): g.nodes[n]["next_hop"] = {}
            g.nodes[n]["next_hop"][net] = s.locRib.next_hop

            if "peer" not in g.nodes[n].keys(): g.nodes[n]["peer"] = {}
            g.nodes[n]["peer"][net] = s.locRib.bgp_speaker_id

            if s.locRib.next_hop != n:
                next_router = next_router_to_destination[s.locRib.next_hop][n][0]
                g[n][next_router]["is_forwarding"][net] = 1
            else:
                next_router = n
    
    network_label_mapping = {}
    for n in networks:
        if not "net_label" in g.nodes[n].keys(): 
            # not using labeled_networks
            return
        net_label = g.nodes[n]["net_label"]
        network_label_mapping[n] = net_label
    
    for src, dst in g.edges():
        if "is_forwarding" in g[src][dst]:
            g[src][dst]["is_forwarding"] = dict([(network_label_mapping[n],holds) for n,holds in g[src][dst]["is_forwarding"].items()])



class BgpSemantics(Semantics):
    def __init__(self, with_groundtruth=False, with_real_world_topologies_p=0.0, labeled_networks=False):
        self.with_groundtruth = with_groundtruth
        self.with_real_world_topologies_p = with_real_world_topologies_p

        self.predicate_semantics = [ # these THREE are the "specifications" that we will check the consistency here (all in "predicate_semantics.py")
            ForwardingPathPredicateSemantics(), 
            #FullForwardingPlanePredicateSemantics(),
            TrafficIsolationPredicateSemantics(),
            ReachablePredicateSemantics()
        ]
        self.predicate_semantics_sample_config = {
            "fwd":              { "n": 32 },
            "reachable":        { "n": 12 },
            "trafficIsolation": { "n": 12 }
        }
        self.labeled_networks = labeled_networks

    def decls(self):
        d = {
            "router": PredicateDeclaration("router", [Constant]),
            "network": PredicateDeclaration("network", [Constant]),
            # bgp
            "external": PredicateDeclaration("external", [Constant]),
            "route_reflector": PredicateDeclaration("route_reflector", [Constant]),
            "ibgp": PredicateDeclaration("ibgp", [Constant, Constant]),
            "ebgp": PredicateDeclaration("ebgp", [Constant, Constant]),
            # gateway, network, LP, AS, OT, MED, IS_EBGP, SPEAKER_ID
            "bgp_route": PredicateDeclaration("bgp_route", [Constant, Constant, int, int, int, int, int, int]),
            # ospf
            "connected": PredicateDeclaration("connected", [Constant, Constant, int]),
            # forwarding plane
            "fwd": PredicateDeclaration("fwd", [Constant, Constant, Constant]),
            "reachable": PredicateDeclaration("reachable", [Constant, Constant, Constant]),
            "trafficIsolation": PredicateDeclaration("trafficIsolation", [Constant, Constant, Constant, Constant]),
        }

        if self.with_groundtruth:
            d["connected_gt"] = PredicateDeclaration("connected_gt", [Constant, Constant, int])
            d["fwd_gt"] = PredicateDeclaration("fwd_gt", [Constant, Constant, Constant])

        if self.labeled_networks:
            d["network"] = PredicateDeclaration("network", [Constant, int])
            d["fwd"] = PredicateDeclaration("fwd", [Constant, int, Constant])
            d["reachable"] = PredicateDeclaration("reachable", [Constant, int, Constant])
            d["trafficIsolation"] = PredicateDeclaration("trafficIsolation", [Constant, Constant, int, int])
            d["bgp_route"] = PredicateDeclaration("bgp_route", [Constant, int, int, int, int, int, int, int])

        return d

    def sample(self,
               num_nodes=None,
               num_networks=None,
               NUM_GATEWAY_NODES=None,
               seed=None,
               real_world_topology=False,
               skip_fwd_facts_p=0.0,
               predicate_semantics_sample_config_overrides={},
               basedOnGraph=None,
               FULLY_MESHED=False,
               use_topology_file=None):
        s = np.random.RandomState(seed=seed)

        if basedOnGraph is not None:
            graph = basedOnGraph
        elif not real_world_topology: # randomly generate a NetworkX graph
            graph = generate_graph(seed=seed, NUM_NODES=num_nodes, NUM_NETWORKS=num_networks, NUM_GATEWAY_NODES=NUM_GATEWAY_NODES, FULLY_MESHED=FULLY_MESHED)
        else:
            #num_nodes = 9999
            #while num_nodes > 80: # avoid very large topologies
            topology_file = choose_random(topologies.all_topology_files, s=s)
            if use_topology_file is not None: topology_file = use_topology_file
            graph = generate_graph_with_topology(topology_file, seed=seed, NUM_NETWORKS=num_networks, NUM_GATEWAY_NODES=NUM_GATEWAY_NODES, FULLY_MESHED=FULLY_MESHED)
            num_nodes = len(graph.nodes())

        router_nodes = [n for n in graph.nodes() if graph.nodes[n]["type"] == "router"]
        network_nodes = [n for n in graph.nodes() if graph.nodes[n]["type"] == "network"]
        external_nodes = [n for n in graph.nodes() if graph.nodes[n]["type"] == "external"]
        route_reflector_nodes = [n for n in graph.nodes() if graph.nodes[n]["type"] == "route_reflector"]        

        def c(r): return Constant(f"c{r}")
        
        # mapping of traffic classes / networks
        network_mapping = {}
        for n in network_nodes:
            network_mapping[c(n).name] = len(network_mapping)

        p = FactBase(self.decls()) # !!! This is the place where NetworkX is converted into a FactBase
        
        def router(r): p.add_fact("router", False, c(r))
        def network(n): 
            if self.labeled_networks: p.add_fact("network", False, c(n), network_mapping[c(n).name])
            else: p.add_fact("network", False, c(n))
        def external(n): p.add_fact("external", False, c(n))
        def route_reflector(n): p.add_fact("route_reflector", False, c(n))
        def connected(r1, r2, weight): 
            p.add_fact("connected", False, c(r1), c(r2), weight)
            if self.with_groundtruth:
                p.add_fact("connected_gt", False, c(r1), c(r2), weight)
        def ibgp(r1, r2): 
            p.add_fact("ibgp", False, c(r1), c(r2))
        def ebgp(r1, r2): 
            p.add_fact("ebgp", False, c(r1), c(r2))
        def bgp_route(gateway, network, route: BgpRoute): 
            network = network_mapping[c(network).name] if self.labeled_networks else c(network)
            # gateway, network, LP, AS, OT, MED, IS_EBGP, SPEAKER_ID
            p.add_fact("bgp_route", False, c(gateway), network, route.local_preference, 
                    route.as_path_length, route.origin_type, route.med, 
                    1 if route.is_ebgp_learned else 0, route.bgp_speaker_id)
        
        for r in router_nodes: router(r) # generate the configuration of "router()";
        for n in network_nodes: network(n)
        for n in external_nodes: external(n)
        for n in route_reflector_nodes: route_reflector(n)

        # set names in network graph
        for n in graph.nodes(): graph.nodes[n]["label"] = c(n).name

        # prevent double linkage
        is_connected = np.zeros([len(graph.nodes()), len(graph.nodes())])
        is_ibgp = np.zeros([len(graph.nodes()), len(graph.nodes())])
        is_ebgp = np.zeros([len(graph.nodes()), len(graph.nodes())])
        external_node_networks = {}
        
        ospf_edges = [(src,dst) for src,dst in graph.edges() if "weight" in graph[src][dst].keys()]
        ibgp_edges = [(src,dst) for src,dst in graph.edges() if graph[src][dst]["type"] == "ibgp"]
        ebgp_edges = [(src,dst) for src,dst in graph.edges() if graph[src][dst]["type"] == "ebgp"]
        network_edges = [(src,dst) for src,dst in graph.edges() if graph[src][dst]["type"] == "network"]

        for src, dst in ospf_edges:
            if is_connected[src][dst] != 0 or is_connected[dst][src] != 0: continue
            is_connected[src][dst] = graph[src][dst]["weight"]
            connected(src, dst, graph[src][dst]["weight"])
        
        for src, dst in ibgp_edges:
            if is_ibgp[src][dst] != 0 or is_ibgp[dst][src] != 0: continue
            is_ibgp[src][dst] = 1
            ibgp(src, dst)
        
        for src, dst in ebgp_edges:
            if is_ebgp[src][dst] != 0 or is_ebgp[dst][src] != 0: continue
            is_ebgp[src][dst] = 1
            ebgp(src, dst)
        
        for src, dst in network_edges:
            external_node = src if src in external_nodes else dst
            network_node = dst if src in external_nodes else src
            assert graph.nodes[external_node]["type"] == "external", f"expected node {external_node} to be external"
            assert graph.nodes[network_node]["type"] == "network", f"expected node {network_node} to be network"
            external_node_networks[external_node] = network_node
        
        # add bgp routes
        for n in external_nodes:
            route = graph.nodes[n]["bgp_route"]
            network = external_node_networks[n]
            bgp_route(n, network, route)
        
        # generating the ground truth for constructing the dataset
        compute_forwarding_state(graph) # generate the specifications

        def get_overrides(pred_s):
            if pred_s.predicate_name in predicate_semantics_sample_config_overrides.keys():
                return predicate_semantics_sample_config_overrides[pred_s.predicate_name]
            return {}

        # derive specification predicates from the computed forwarding plane
        for pred_s in self.predicate_semantics:
            config = self.sampling_config(pred_s, overrides=get_overrides(pred_s))
            derived = pred_s.sample(graph, random=s, **config)

            if self.labeled_networks:
                for f in derived:
                    def network_constants_to_label(a):
                        if type(a) is Constant and a.name in network_mapping.keys(): return network_mapping[a.name]
                        return a
                    f.args = [network_constants_to_label(a) for a in f.args]
                        
            for f in derived: p.add_fact(f.name, f.is_negated, *f.args)        

        return p

    def sampling_config(self, predicate_semantics, overrides={}):
        if predicate_semantics.predicate_name in self.predicate_semantics_sample_config.keys():
            config = dict(self.predicate_semantics_sample_config[predicate_semantics.predicate_name])
        else:
            config = {}
        config.update(overrides)
        return config

    def check(self, p, return_summary=False, logging=False, ignore_missing_fwd_facts=False, return_corrected_program=False, return_counter_facts=False):
        network, _ = self.network(p) # generate NetworkX, based on FactBase
        compute_forwarding_state(network) # re-calculate the FWD

        predicate_semantics = dict([(s.predicate_name, s) for s in self.predicate_semantics])

        consistency_results = []
        checked_fact_names = []

        program_correct = FactBase(p.predicate_declarations)
        is_spec = []

        # add corrected fact if predicate semantics check fails
        for fact in p.get_all_facts():
            if not fact.name in predicate_semantics.keys():
                if return_corrected_program: 
                    program_correct.add_fact(fact.name, fact.is_negated, *fact.args)
                    is_spec.append(False)
                continue # do not check the consistency of the configurations
            s = predicate_semantics[fact.name]

            if s.check(network, fact):
                consistency_results.append(1.0)
                checked_fact_names.append(fact.name)
                
                if return_corrected_program: 
                    program_correct.add_fact(fact.name, fact.is_negated, *fact.args)
                    is_spec.append(True)
            else:
                consistency_results.append(0.0)
                checked_fact_names.append(fact.name)
                
                if return_corrected_program: 
                    program_correct.add_fact(fact.name, not fact.is_negated, *fact.args)
                    is_spec.append(True)

        consistency_values = np.array(consistency_results)
        checked_fact_names = np.array(checked_fact_names)

        return_value = [consistency_values.mean()]
    
        if return_corrected_program:
            program_correct.is_spec = is_spec
            return_value.append(program_correct)
        
        if return_summary:
            summary = {}
            for n in checked_fact_names:
                summary[n] = consistency_values[checked_fact_names == n].mean()
            summary["overall"] = consistency_values.mean()
            return_value.append(summary)

        return return_value if len(return_value) > 1 else return_value[0]

    def network(self, p: FactBase): # construct the NetworkX graph from the FactBase
        routers = p.constants("router")
        network_nodes = p.constants("network")
        route_reflector_nodes = p.constants("route_reflector")
        external_nodes = p.constants("external")
        net_label_to_nodes = {}

        g = nx.DiGraph()
        
        for n in network_nodes: 
            g.add_node(n, type="network")
            if self.labeled_networks:
                _, net_label, _ = p.query("network", n,-1)[0]
                g.nodes[n]["net_label"] = net_label
                net_label_to_nodes[net_label] = n
        for n in routers: g.add_node(n, type="router")
        for n in route_reflector_nodes: g.add_node(n, type="route_reflector")
        for n in external_nodes: g.add_node(n, type="external")

        # query opsf links from factbase
        edges = p.query("connected", -1, -1, -1)
        for src, dst, weight, holds in edges:
            g.add_edge(src, dst, weight=weight, type="ospf", is_forwarding={})
            g.add_edge(dst, src, weight=weight, type="ospf", is_forwarding={})
        
        # query ibgp edges from factbase
        edges = p.query("ibgp", -1, -1)
        for src, dst, holds in edges:
            g.add_edge(src, dst, type="ibgp")
            g.add_edge(dst, src, type="ibgp")
            
        # query ebgp edges from factbase
        edges = p.query("ebgp", -1, -1)
        for src, dst, holds in edges:
            g.add_edge(src, dst, type="ebgp", is_forwarding={})
            g.add_edge(dst, src, type="ebgp", is_forwarding={})
        
        # query bgp routes from factbase
        # gateway, network, LP, AS, OT, MED, IS_EBGP, SPEAKER_ID
        routes = p.query("bgp_route", -1, -1, -1, -1, -1, -1, -1, -1)
        for gateway, network, LP, AS, OT, MED, IS_EBGP, SPEAKER_ID, holds in routes:
            if self.labeled_networks: network = net_label_to_nodes[network]
            g.nodes[gateway]["bgp_route"] = BgpRoute(network, LP, AS, OT, MED, IS_EBGP, SPEAKER_ID, 0)
            g.add_edge(gateway, network, type="network")
            g.add_edge(network, gateway, type="network")
        
        facts, constant_name_mapping = p.get_all_facts(return_constant_name_mapping=True)
        fwd_facts = [f for f in facts if f.name == "fwd"]
        for fwd in fwd_facts:
            src, net, dst = [constant_name_mapping[a.name] if type(a) is Constant else a for a in fwd.args]
            if not fwd.is_negated and not fwd.is_query:
                g[src][dst]["is_forwarding"][net] = 1

        node_id_to_name = dict([(n.id, n.name) for n in p.nodes.values()])
        for n in g.nodes():
            if n in node_id_to_name.keys():
                g.nodes[n]["label"] = node_id_to_name[n]

        return g, network_nodes

def slice(x, feature):
    num_networks = 0

    network_label_features = ["predicate_network_arg1", "predicate_bgp_route_arg1", "predicate_fwd_arg1", 
        "predicate_reachable_arg1", "predicate_trafficIsolation_arg2", "predicate_trafficIsolation_arg3"]

    is_network_specific_node_masks = [
        x[:,:,feature(f).idx] > -1 for f in network_label_features
    ]
    node_network_indices = [ 
        x[mask][:,feature(f).idx] for f,mask in zip(network_label_features, is_network_specific_node_masks)
    ]
    non_network_specific_mask = torch.stack([mask for mask in is_network_specific_node_masks]).max(axis=0).values.bool().logical_not()
    num_networks = int(torch.tensor([index.max() if len(index) > 0 else 0 for index in node_network_indices]).max().item())
    
    
    slices = torch.zeros(num_networks+1, x.size(0), dtype=torch.bool, device=x.device)
    for i in range(num_networks+1):
        slices[i] = non_network_specific_mask.view(-1).clone().detach()

        for mask, index in zip(is_network_specific_node_masks, node_network_indices):
            slices[i][mask.view(-1)] = torch.logical_or(slices[i][mask.view(-1)], index == i)

    x_sliced = torch.zeros(x.size(0), num_networks+1, x.size(-1), dtype=x.dtype, device=x.device)
    x_mask = torch.zeros(x.size(0), num_networks+1, dtype=x.dtype, device=x.device)

    for i,mask in enumerate(slices): 
        slice = mask.view(-1,1) * x[:,0] + mask.view(-1,1).logical_not() * torch.tensor(-1, device=x.device)
        x_sliced[:,i] = slice
        x_mask[:,i] = mask.view(-1)

    return x_sliced, x_mask