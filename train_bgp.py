import sys
import os

from torch.nn.modules.transformer import TransformerDecoderLayer
from torch_geometric.nn.conv.message_passing import MessagePassing

sys.path.append(os.path.join(os.path.dirname(__file__), "dataset"))
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))

import numpy as np
from multiprocessing import Pool
from functools import reduce
from tqdm import tqdm

from torch_geometric.nn import GATConv, GCNConv, GatedGraphConv
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from feature import *
from coders import *
from sampling import sample_random_order

from factbase import FactBase, Constant
from nutils import *
from semantics import FactBaseSemanticsDataset

from snapshot import ModelSnapshot
from bgp_semantics import BgpSemantics

import argparse

NUM_EDGE_TYPES = 4

# determines factbase consistency
def check_factbase(semantics, data): 
    program = FactBase.from_data(data)
    _, summary = semantics.check(program, ignore_missing_fwd_facts=True, return_summary=True)
    return summary

def mask_parameters(x, decls):
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
    masked_bgp_route_args = [2,3,5]
    for i in masked_bgp_route_args:
        idx = feature("predicate_bgp_route_arg"+str(i)).idx
        mask[:,:,idx] = (x[:,:,idx] > -1)

    return mask.bool()

def mask_parameters_or_fw(x, decls):
    fwd_predicate_feature_value = decls["fwd"].predicate_feature_value
    mask = mask_parameters(x, decls)
    mask[:,:,feature("holds").idx] = mask[:,:,feature("holds").idx].logical_or(
        (x[:,:,feature("holds").idx] > -1).logical_and(
            x[:,:,feature("predicate").idx] == fwd_predicate_feature_value
        )
    )
    return mask.bool()

def combine_dict(dicts):
    flatten = lambda seq: reduce(lambda a,b: a.union(b), seq, set())
    keys = flatten([set(d.keys()) for d in dicts])
    values_for_key = lambda k: [d[k] for d in dicts if k in d.keys()]
    return dict([(k,torch.tensor(values_for_key(k))) for k in keys])

class AsyncFactBaseChecker:
    def __init__(self, pool):
        self.checking_tasks = []
        self.current_step = -1
        self.on_step_finish = None
        
        self.pool = pool

    def wait_for_step_results(self, global_step):
        if self.current_step == global_step:
            return
        else:
            if any([not task.ready() for task in self.checking_tasks]):
                print("info: waiting for program checker in evaluation (consider using more processes)")

            self._get_results_blocking()
            
            self.current_step = global_step
            self.checking_tasks = []

    def _get_results_blocking(self):
        summaries = [task.get() for task in self.checking_tasks]
        results = combine_dict(summaries)
        if self.current_step != -1: 
            assert self.on_step_finish is not None, "must set self.on_step_finish callback function"
            self.on_step_finish(self.current_step, results)

    def check_last_step_results(self):
        if any([not task.ready() for task in self.checking_tasks]):
            return
        self._get_results_blocking()

    def queue(self, semantics, data, global_step):
        self.wait_for_step_results(global_step)
        self.checking_tasks.append(self.pool.apply_async(check_factbase, (semantics, data)))

class MaxGraphLayer(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr='max')
        
        self.lin = torch.nn.Linear(hidden_dim, hidden_dim)
        self.message_func = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.lin(x)
        x = torch.relu(x)

        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return torch.relu(self.message_func.forward(x_j))

class EdgeTypeTransformerLayer(torch.nn.Module):
    def __init__(self, hidden_dim, d_inner, num_edge_types, n_heads=8, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_inner = d_inner

        # initialise layers
        self.layers = []
        for j in range(num_edge_types):
            l = GATConv(self.hidden_dim, self.hidden_dim, n_heads, False)
            #l = GCNConv(self.hidden_dim, self.hidden_dim, aggr="max")
            #l = MaxGraphLayer(self.hidden_dim)
            self.layers.append(l)
            self.add_module("layer_edge_type_" + str(j), l)
        
        # initialise norm
        self.drop = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.BatchNorm1d(hidden_dim)
        self.drop1 = torch.nn.Dropout(dropout)
        self.norm2 = torch.nn.BatchNorm1d(hidden_dim)
        self.drop2 = torch.nn.Dropout(dropout)
        
        self.linear1 = torch.nn.Linear(self.hidden_dim, self.d_inner)
        self.linear2 = torch.nn.Linear(self.d_inner, self.hidden_dim)
    
    def forward(self, x, edge_index, edge_type):
        def edge_index_for_type(type):
            indices = (edge_type == type)
            index = edge_index[:,indices]
            return index

        def layer(l, x, ei):
            return l(x, ei)

        # edge type attention
        x2 = torch.stack([layer(l, x.view(-1, self.hidden_dim), edge_index_for_type(t)) for t,l in enumerate(self.layers)], axis=0).sum(axis=0)
        x = x + self.drop1(x2)
        x = self.norm1.forward(x.view(-1, self.hidden_dim)).view(x.shape)
        x = self.linear2(self.drop(torch.relu(self.linear1(x))))
        x = self.norm2(x.view(-1, self.hidden_dim)).view(x.shape)
        
        return x

class EdgeTypeGraphTransformer(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers, num_edge_types):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.propagation_layers = [EdgeTypeTransformerLayer(hidden_dim, 4*hidden_dim, num_edge_types, 8) for i in range(num_layers)]
        for i, l in enumerate(self.propagation_layers): 
            self.add_module("prop_layer_" + str(i), l)

    def forward(self, x, edge_index, edge_type):
        for l in self.propagation_layers:
            x = l.forward(x, edge_index, edge_type)
            x = torch.relu(x)
        return x

class PredicateGraphEmbedding(torch.nn.Module):
    def __init__(self, features, hidden_dim, num_edge_types, excluded_feature_indices=set(), num_layers=6):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types

        self.encoder = NodeFeatureEmbedding(hidden_dim, features, excluded_feature_indices=excluded_feature_indices)
        self.decoder = NodeFeatureDecoder(hidden_dim, features, excluded_feature_indices=excluded_feature_indices)
        
    def forward(self, x, mask, edge_index, edge_type, reliable_masking):
        return self.encoder.forward(x, mask, reliable_masking, positional_encoding=False)

class Model(torch.nn.Module):
    def __init__(self, features, hidden_dim, num_edge_types, excluded_feature_indices):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types

        self.embedding = PredicateGraphEmbedding(features, hidden_dim, self.num_edge_types, excluded_feature_indices)
        self.decoder = self.embedding.decoder

        self.transformer_encoder = EdgeTypeGraphTransformer(self.hidden_dim, 2, num_edge_types)
        self.transformer_decoder = EdgeTypeGraphTransformer(self.hidden_dim, 6, num_edge_types)

        self.num_iterations = 4
    
    def add_noise(self, x):
        noise = torch.randn(x.shape, device=x.device)
        noise_mask = torch.zeros([self.hidden_dim], device=x.device)
        noise_mask[0:int(self.hidden_dim/2)] = 1
        
        return x + noise_mask.unsqueeze(0) * noise

    def forward(self, x, mask, edge_index, edge_type, reliable_masking):
        # adjust shape when called from eval/serve script
        if edge_index.dim() == 3:
            assert edge_index.size(0) == 1
            assert edge_type.size(0) == 1
            edge_index = edge_index[0]
            edge_type = edge_type[0]

        assert x.size(1) == 1

        x = self.embedding.forward(x, mask, edge_index, edge_type, reliable_masking)[:,0]
        x = self.transformer_encoder.forward(x, edge_index, edge_type)
        
        # decode from noise
        x = self.add_noise(x)

        for i in range(self.num_iterations):
            x = self.transformer_decoder.forward(x, edge_index, edge_type) + x

        return x

def eval(model, dataset, features, decls, program_checker, step, num_samples, writer, prefix, with_consistency=False, p=1.0):
    batch_size = 2
    loader = DataLoader(dataset, batch_size=batch_size)
    
    program_checker.wait_for_step_results(step)
    model.eval()

    for batch in loader:
        original_batch = batch
        original_batch.edge_type = batch.edge_type.clone()

        batch = batch.to(device)
        batch.x = batch.x.unsqueeze(1)
        batch.edge_index = reflexive(bidirectional(batch.edge_index), num_nodes=batch.x.size(0))
        batch.edge_type = reflexive_bidirectional_edge_type(batch.edge_type, batch.x.size(0))
        
        mask = mask_parameters(batch.x, decls)
        x_emb = model.forward(batch.x, mask, batch.edge_index, batch.edge_type, True)
        target = mask_node_features(batch.x, mask.logical_not())
        
        weight_accuracy = model.decoder.accuracy(x_emb, target, "predicate_connected_arg2")
        writer.add_scalar(prefix+"WeightSynthesis/GroundtruthAccuracy", weight_accuracy)

        tasks = [None for j in range(batch_size * num_samples)]

        if with_consistency:
            for i in range(num_samples):
                x_completed = sample_random_order(model, features, batch, mask, iterative=False, number_of_shots=4)
            
                for j, original_data in enumerate(original_batch.to_data_list()):
                    data = original_data
                    data.x = x_completed[(batch.batch == j),0].clone()

                    tasks[j*num_samples + i] = [sem, data.to(torch.device("cpu")), step]
        
        for t in tasks: program_checker.queue(*t)

if __name__ == '__main__':
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=128)
        parser.add_argument("--epochs",     dest="epochs",     type=int, default=2700)
        parser.add_argument("--checkpoint", dest="checkpoint", type=str, default=None)
        return parser.parse_args()

    args = get_args()
    HIDDEN_DIM = args.hidden_dim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on", device)
    snapshot = ModelSnapshot(__file__)

    sem = BgpSemantics() # sem: topology/config → run Prot → produce spec facts
    dataset = FactBaseSemanticsDataset(sem, "bgp-ospf-dataset-sub", num_samples=128) # generate (or load) the dataset
    print("Dataset Size", len(dataset))
    
    num_validation_samples = 16
    training_dataset, validation_dataset = dataset[num_validation_samples:], dataset[:num_validation_samples]
    training_eval_dataset = dataset[num_validation_samples:2*num_validation_samples]
    
    print("Validation Dataset Size", len(validation_dataset))

    predicate_declarations = sem.decls()
    for decl in predicate_declarations.values():
        constant_types = [at for at in decl.arg_types if at is Constant]
        assert len(constant_types) <= NUM_EDGE_TYPES, f"declaration {decl} requires more than {NUM_EDGE_TYPES} edge types"
    prog = FactBase(predicate_declarations)
    feature = prog.feature_registry.feature

    excluded_feature_indices = set([1])
    features = prog.feature_registry.get_all_features()
    print(prog.predicate_declarations)
    print(features)
    model = Model(features, HIDDEN_DIM, NUM_EDGE_TYPES, excluded_feature_indices).to(device)
    model.feature = feature
    
    if args.checkpoint is not None:    
        state_dict, hidden_dim, _, _ = torch.load(args.checkpoint, map_location=device)
        assert HIDDEN_DIM == hidden_dim, f"dimension mismatch configured {HIDDEN_DIM} vs. state dict {hidden_dim}"
        model.load_state_dict(state_dict)
        print("restored checkpoint from ", args.checkpoint)

    writer = snapshot.writer()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) # get_std_opt(model)
    print(len(list(model.parameters())))

    pool = Pool(processes=8)
    program_checker_training = AsyncFactBaseChecker(pool)
    program_checker_validation = AsyncFactBaseChecker(pool)
    num_eval_samples = 10
    
    def best_sample_mean(res):
        if res.numel() == 0: return 0
        res = res.view(-1, num_eval_samples)
        res = res.max(axis=1).values
        return res.mean()

    def on_evaluation_step_finish(prefix):
        def handler(step, res):
            for key in res.keys():
                writer.add_scalar(f"{prefix}/WeightSynthesis/Consistency/{key}", best_sample_mean(res[key]), global_step=step)
        return handler
    program_checker_training.on_step_finish = on_evaluation_step_finish("Training")
    program_checker_validation.on_step_finish = on_evaluation_step_finish("Validation")

    synthesised_features = [
        feature("predicate_connected_arg2"),  # OSPF weights
        # bgp_route: LP x AS x -OT x MED x -IS_EBGP x -SPEAKER_ID
        feature("predicate_bgp_route_arg2"),  # BGP LP
        feature("predicate_bgp_route_arg3"), # BGP AS
        #feature("predicate_bgp_route_arg4"), # BGP ORIGIN_TYPE
        feature("predicate_bgp_route_arg5"), # BGP MED
        #feature("predicate_bgp_route_arg6"), # BGP IS_EBGP
        #feature("predicate_bgp_route_arg7") # SPEAKER_ID
    ]

    p = 0.85
    batch_size = 8
    num_samples_per_epoch = 1024
    num_batches_per_epoch = int(num_samples_per_epoch / batch_size)

    for epoch in tqdm(range(args.epochs), leave=False):
        training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        step_writer = StepWriter(writer, epoch)

        model.train()
        for i, batch in tqdm(enumerate(training_loader), leave=False, total=num_batches_per_epoch, desc=f"Epoch {epoch}"):
            if i > num_batches_per_epoch: break

            optimiser.zero_grad()
            batch = batch.to(device)
            batch.x = batch.x.unsqueeze(1)
            batch.edge_index = reflexive(bidirectional(batch.edge_index), num_nodes=batch.x.size(0))
            batch.edge_type = reflexive_bidirectional_edge_type(batch.edge_type, batch.x.size(0))

            # mask = mask_like(batch.x, p=p).to(device) * mask_parameters_or_fw(batch.x, prog.predicate_declarations)
            # mask = mask_like(batch.x, p=p).to(device) * mask_parameters(batch.x, prog.predicate_declarations)
            mask = mask_parameters(batch.x, prog.predicate_declarations)
            target = mask_node_features(batch.x, mask.logical_not())

            x_emb = model.forward(batch.x, mask, batch.edge_index, batch.edge_type, False)

            loss = torch.tensor(0.0, device=device)
            for f in synthesised_features: loss += model.decoder.loss(x_emb, target, f.name)

            if torch.any(torch.isnan(loss)): print("isnan")

            loss.backward()
            optimiser.step()

        step_writer.add_scalar("Loss", loss)

        step_writer.add_scalar("Training/MaskingFactor", mask.float().mean())
        
        #p = min(0.8, p + 0.0001)
        # step_writer.add_scalar("Training/lr", optimiser._rate)
        
        program_checker_training.check_last_step_results()
        program_checker_validation.check_last_step_results()

        if epoch % 100 == 0:
            eval(model, training_eval_dataset, synthesised_features, prog.predicate_declarations, program_checker_training, epoch, num_eval_samples, step_writer, "Training/", True)
            eval(model, validation_dataset, synthesised_features, prog.predicate_declarations, program_checker_validation, epoch, num_eval_samples, step_writer, "Validation/", True)
        if epoch % 100 == 0:
            uid = os.environ["EXP_ID"] if "EXP_ID" in os.environ.keys() else "unnamed"
            torch.save([model.state_dict(), HIDDEN_DIM, NUM_EDGE_TYPES, excluded_feature_indices], f"models/{uid}-model-epoch{epoch}.pt")
        
        writer.flush()

    # always save and eval last state
    eval(model, training_eval_dataset, synthesised_features, prog.predicate_declarations, program_checker_training, epoch, num_eval_samples, step_writer, "Training/", True)
    eval(model, validation_dataset, synthesised_features, prog.predicate_declarations, program_checker_validation, epoch, num_eval_samples, step_writer, "Validation/", True)
    
    uid = os.environ["EXP_ID"] if "EXP_ID" in os.environ.keys() else "unnamed"
    torch.save([model.state_dict(), HIDDEN_DIM, NUM_EDGE_TYPES, excluded_feature_indices], f"models/{uid}-model-epoch{epoch}.pt")
