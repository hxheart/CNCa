import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../model"))

import numpy as np

from semantics import FactBaseSemanticsDataset  # 1) FRAMEWORK:  how to create datasets
from bgp_semantics import BgpSemantics          # 3) CORE LOGIC: actual BGP/OSPF simulation
from nutils import choose_random

class ConfiguredBgpSemantics:                   # 2) CONFIGURATION: what parameters to use
    def __init__(self):
        self.s = BgpSemantics(labeled_networks=False)   # BgpSemantics does the protocol simulation. Explore this to find out how BGP/OSPF simulation works

    def sample(self, seed):
        s = np.random.RandomState(seed=seed) # randomly generate some network parameters, such as number of nodes, number of ASes, ...
        real_world_topology = False # np.random.random() < 0.2 # {True:0.3, False:0.7}
        num_networks = choose_random(list(range(4,8)), s)
        num_gateway_nodes = 3
        num_nodes = choose_random(range(16,24), s)

        sample_config_overrides = {
            "fwd":              {"n": choose_random([8, 10, 12], s)},   # num paths of fwd predicates
            "reachable":        {"n": choose_random([4, 5, 6, 7], s)},  # num reachable predicates
            "trafficIsolation": {"n": choose_random(range(10, 30), s)},     # num trafficIsolation predicates
        }

        seed = s.randint(0,1024*1024*1024)

        # Here, this is using BgpSematics.sample()
        return self.s.sample(num_nodes=num_nodes,
                             real_world_topology=real_world_topology,
                             num_networks=num_networks,
                             predicate_semantics_sample_config_overrides=sample_config_overrides,
                             seed=seed,
                             NUM_GATEWAY_NODES=num_gateway_nodes)

if __name__ == "__main__":
    dataset = FactBaseSemanticsDataset(  #
        ConfiguredBgpSemantics(),        # here will create a BgpSemantics instance: "self.s = BgpSemantics(labeled_networks=False)", which is not empty;
        "bgp-ospf-dataset-sub",     # go to "root", just creating a path/folder;
        num_samples=10*1024,
        tmp_directory="tmp-bgp-dataset"
    )
    # print(len(dataset))
    # print('\n The first sample is:', dataset[0])
    # print('\n', dataset[0].keys())

    # Accessing the underlying semantic objects
    semantics_visualise = ConfiguredBgpSemantics()
    sample_visualise = semantics_visualise.sample(seed=42)
    print("\n 3) ===> Sample from semantics:\n", sample_visualise)
    print("\n 4) ===> Type of sample:\n", type(sample_visualise))

    data = dataset[0]
    print('\n ===> data is:\n', data) # PyTorch Geometric (PyG) Data object: torch_geometric.data.Data
    print('\n ===> data.x is:\n', data.x)           # x represents: node feature matrix -- semantic information is encoded here!
    print('\n ===> data.x[0] is:\n', data.x[0])
    print('\n ===> data.x[100] is:\n', data.x[100])
    print('\n ===> data.x[:,0] is (Feature 0: type (idx=0)):\n', data.x[:,0])
    print('\n ===> data.x[:,1] is (Feature 1: id (idx=1)):\n', data.x[:, 1])
    print('\n ===> data.x[:,2] is (Feature 2: predicate (idx=2)):\n', data.x[:, 2])
    print('\n ===> data.x[:,3] is (Feature 3: holds (idx=3)):\n', data.x[:, 3])
    print('\n ===> data.x[:,11] is (Feature 11: predicate_external (idx=11)):\n', data.x[:, 11])
    print('\n ===> data.x[:,12] is (Feature 12: predicate_network (idx=12)):\n', data.x[:, 12])
    print('\n ===> data.x[:,13] is (Feature 13: predicate_route_reflector (idx=13)):\n', data.x[:, 13])
    print('\n ===> data.x[:,14] is (Feature 14: predicate_router (idx=14)):\n', data.x[:, 14])
    print('\n ===> data.edge_type is:\n', data.edge_type)
    print('\n ===> data.edge_index.shape is:\n', data.edge_index.shape)
    print('\n ===> data.edge_index[0] is:\n', data.edge_index[0])  # Edge connectivity matrix
    print('\n ===> data.edge_index[1] is:\n', data.edge_index[1])  # Edge connectivity matrix
    print('\n ===> data.edge_attr is:\n', data.edge_attr)   # Edge attribute matrix

    # View all properties of a data object
    print("\n 1) ===> Data attributes:\n", dir(data))
    print("\n 2) ===> Data keys:\n", data.keys())



# configuration facts: router, network, external, connected, ibgp, ebgp, bgp_route
# specification facts: fwd, reachable, trafficIsolation