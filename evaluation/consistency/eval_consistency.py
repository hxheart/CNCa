import sys

sys.path.append("../../dataset")
sys.path.append("../../model")
sys.path.append("../../")

import pandas as pd
import numpy as np
from sampling import *
from beam import beam_search

from train_bgp import Model as GraphBgpModel
from bgp_semantics import BgpSemantics
from factbase import *
from tqdm import tqdm
import time
import torch
import argparse
from snapshot import ModelSnapshot
import os

class SampleDescriptor:
    def __init__(self, num_nodes, num_networks, program):
        self.num_nodes = num_nodes
        self.num_networks = num_networks
        self.program = program

parser = argparse.ArgumentParser(description='Evaluate models.')
parser.add_argument('modelpath', metavar='path', type=str, nargs=1,
                    help='the path of the model')
parser.add_argument('--num-shots', dest='num_shots', type=int, default=1,
                    help='the number of sampling shots')
parser.add_argument('--num-iterations', dest='num_iterations', type=int, default=-1,
                    help='the number of decoder iterations')
parser.add_argument('--inverted', dest='inverted', type=bool, default=False,
                    help='whether to invert the sampling order')
parser.add_argument('--random', dest='random', type=bool, default=False,
                    help='whether to randomise the sampling order')
parser.add_argument('--cpu', dest='cpu', type=bool, default=False,
                    help='whether to use the cpu as device')
parser.add_argument('--dataset', dest='dataset', type=str, default="random",
                    help='which dataset to use (random|zoo)')
parser.add_argument('--sampling-mode', dest='sampling_mode', type=str, default="argmax",
                    help='which sampling mode to use (argmax|topk|regular)')
parser.add_argument('--num-samples', dest='samples', type=int, default=10,
                    help='number of samples per example')
parser.add_argument('--evolutionary', dest='evolutionary', type=bool, default=False,
                    help='whether to use evolutionary optimisation')
parser.add_argument('--beam', dest='beam', action="store_true", default=False)
parser.add_argument('--protocol', dest='protocol', type=str, default="bgp")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.cpu:
    device = torch.device("cpu")

def get_id(filename):
    str_n = filename.split("-n", 1)[1].split(".", 1)[0]
    if "-unsatsample" in str_n:
        str_n = str_n.split("-unsatsample", 1)[0]
    return int(str_n)

if args.dataset.startswith("."):
    import os
    files = [f for f in os.listdir(args.dataset) if f.endswith(".logic")]
    files = sorted(files, key=lambda x: get_id(x))
    print(files)
    programs = [torch.load(os.path.join(args.dataset, f)) for f in files]
else:
    print(f"unknown dataset {args.dataset}")
    sys.exit(1)

results = None
consistency_metrics = None

model_path = args.modelpath[0]
snapshot = ModelSnapshot(__file__)

if "EXP_ID" in os.environ.keys():
    exp_id = os.environ["EXP_ID"]
    exp_id = os.environ["EXP_ID"]
    rand = str(np.random.randint(10000,20000))
    uid = f"{exp_id}-{rand}"
    model_name = f"results-{uid}.pkl"
    print("using exp_id as name", uid)
else: 
    print("Please specify a EXP_ID as environment variable.")
    sys.exit(1)

NO_STATIC_ROUTES = True

print("inverted", args.inverted)
print("random", args.random)
print("device", device)
print("NO_STATIC_ROUTES", NO_STATIC_ROUTES)

s = BgpSemantics()

predicate_declarations = s.decls()
print(predicate_declarations)
prog = FactBase(predicate_declarations)
feature = prog.feature_registry.feature

excluded_feature_indices = set([1])
features = prog.feature_registry.get_all_features()

model = None

writer = snapshot.writer()

if model_path != "random":
    state_dict, HIDDEN_DIM, NUM_EDGE_TYPES, excluded_feature_indices = torch.load(model_path, map_location=device)
    model = GraphBgpModel(features, HIDDEN_DIM, NUM_EDGE_TYPES, excluded_feature_indices).to(device)

    state_dict = convert_old_gat_conv_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.feature = feature
    print("using model at", model_path, "with hidden dim", HIDDEN_DIM, "and num edge types", NUM_EDGE_TYPES)

    if args.num_iterations > 0:
        model.num_iterations = args.num_iterations
    print("model iterations", model.num_iterations)
else:
    pass

def mask_parameters(x, decls, with_prob_static_route=True, without_static_routes=NO_STATIC_ROUTES):
    mask = torch.zeros_like(x)
    
    # predicate_connected_arg2 [weight]
    mask[:,:,feature("predicate_connected_arg2").idx] = (x[:,:,feature("predicate_connected_arg2").idx] > -1)
    
    # bgp_route: gateway, network, LP, AS, OT, MED, IS_EBGP, SPEAKER_ID
    # predicate_bgp_route_arg2 [LP]
    # predicate_bgp_route_arg3 [AS]
    # predicate_bgp_route_arg4 [OT]
    # predicate_bgp_route_arg5 [MED], 
    # predicate_bgp_route_arg6 [IS_EBGP]
    # predicate_bgp_route_arg7 [SPEAKER_ID]
    if args.protocol == "bgp":
        masked_bgp_route_args = [2,3,5]
        for i in masked_bgp_route_args:
            idx = feature("predicate_bgp_route_arg"+str(i)).idx
            mask[:,:,idx] = (x[:,:,idx] > -1)

    return mask.bool()

def sample_random_prediction(model, prediction_features, batched_data, mask):
    r = torch.zeros_like(batched_data.x)
    for f in prediction_features:
        r[:,:,f.idx] = torch.randint(0, 32, size=[data.x.size(0), 1]).to(device)
    r[:,:,feature("predicate_bgp_route_arg6").idx] = torch.randint(0, 2, size=[data.x.size(0),1]).to(device)
    
    return mask * r + mask.logical_not() * batched_data.x

#descriptor = programs[0]
pbar = tqdm(programs)
consistency_metrics = ["fwd", "reachable", "trafficIsolation", "overall"]
for i, descriptor in enumerate(pbar):
    if type(descriptor) is Data or type(descriptor) is dict:
        descriptor = SampleDescriptor(0, 0, FactBase.from_data(descriptor))
    data, names = descriptor.program.to_torch_data(return_node_names=True)
        
    prediction_features = [
        feature("predicate_connected_arg2"),  # OSPF weights
        # bgp_route: LP x AS x -OT x MED x -IS_EBGP x -SPEAKER_ID
        feature("predicate_bgp_route_arg2"),  # BGP LP
        feature("predicate_bgp_route_arg3"), # BGP AS
        #feature("predicate_bgp_route_arg4"), # BGP ORIGIN_TYPE
        feature("predicate_bgp_route_arg5"), # BGP MED
        #feature("predicate_bgp_route_arg6"), # BGP IS_EBGP
        #feature("predicate_bgp_route_arg7") # SPEAKER_ID
    ]

    batched_data = data.clone().to(device)
    batched_data.x = batched_data.x.unsqueeze(1)
    batched_data.edge_index = reflexive(bidirectional(batched_data.edge_index), num_nodes=batched_data.x.size(0))
    batched_data.edge_type = reflexive_bidirectional_edge_type(batched_data.edge_type, batched_data.x.size(0))
    mask = mask_parameters(batched_data.x, predicate_declarations).to(device)
    
    best_consistency = 0

    for j in range(args.samples):
        tstart = time.time()
        if model is not None:
            if args.beam:
                data.x = beam_search(model, prediction_features, batched_data, mask, iterative=True, 
                    number_of_shots=args.num_shots, inverted=False, mode=args.sampling_mode, beam_n=128, beam_k=8)[:,0]
            elif args.random:
                data.x = sample_random_order(model, prediction_features, batched_data, mask, iterative=True, 
                    number_of_shots=args.num_shots, inverted=args.inverted, mode=args.sampling_mode)[:,0]
            elif args.evolutionary:
                def check_fct(data):
                    predicted_program = FactBase.from_data(data, decls=predicate_declarations, names=names)
                    consistency, summary = s.check(predicted_program, return_summary=True)
                    return consistency
                data.x = sample_evolutionary(model, prediction_features, batched_data, data, mask,
                    check_fct, device, mode=args.sampling_mode)[:,0]
            else:
                data.x = sample_by_entropy(model, prediction_features, batched_data, mask, iterative=True, number_of_shots=args.num_shots, inverted=args.inverted)[:,0]
        else:
            data.x = sample_random_prediction(model, prediction_features, batched_data, mask)[:,0]
        timeelapsed = time.time() - tstart
        predicted_program = FactBase.from_data(data, decls=predicate_declarations, names=names)
        consistency, summary = s.check(predicted_program, return_summary=True)
        best_consistency = max(consistency, best_consistency)

        if results is None:
            columns = ["num_nodes", "num_networks", "sample_id"] + consistency_metrics + ["time"]
            results = pd.DataFrame([], columns=columns)
        
        def get_value(k):
            if k in summary.keys(): return summary[k]
            else: return 1.0

        num_nodes = len(descriptor.program.constants("router")) + len(descriptor.program.constants("route_reflector"))
        num_networks = len(descriptor.program.constants("network"))

        results = results.append(pd.DataFrame([[num_nodes, num_networks, i] + [get_value(m) for m in consistency_metrics] + [timeelapsed]], columns=columns))
        pbar.set_description("Consistency %0.2f (best %0.2f) (Nodes %d, Sample %d)" % (consistency, best_consistency, num_nodes, j))

        writer.add_scalar("Time", timeelapsed, global_step = num_nodes)
        writer.add_scalar("Consistency/Mean", consistency, global_step = num_nodes)
        writer.flush()

        if best_consistency == 1.0: break
        #if args.num_shots == 1: break
    
    print(best_consistency)
    writer.add_scalar("Consistency/BestOf10", best_consistency, global_step = i)
    writer.flush()
#results.groupby(["num_nodes", "num_networks"]).mean()
results.to_pickle(f"{model_name}-results.pkl")
