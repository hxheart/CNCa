import torch
from torch_geometric.data import Data

from nutils import *

import torch.nn.functional as F

def _sample(model, features, prediction_mask, x, mask, edge_index, edge_type):
    x_emb = model.forward(x, mask, edge_index, edge_type, True)
        
    x_completed = x.clone().detach()
    for feature in features:
        x_completed[:,:,feature.idx][prediction_mask[:,:,feature.idx]] = model.decoder.decode(x_emb, feature.name)[prediction_mask[:,:,feature.idx]]
    
    return x_completed

def sample_prediction(model, features, data, mask, iterative=True, number_of_shots=None):# TODO: use order to sample one parameter per graph in batch via (order == idx)
    x = data.x
    edge_index = data.edge_index
    edge_type = data.edge_type

    model.eval()

    order = torch.zeros_like(mask, dtype=torch.long)

    for b in range(x.size(1)):
        num_masked_features_in_sample = mask[:,b].sum()
        order[:,b,][mask[:,b]] = torch.arange(1, num_masked_features_in_sample + 1, device=x.device)

    if number_of_shots is not None:
        order = order % number_of_shots

    num_samples = order.max()
    for s in range(1, num_samples + 1):
        prediction_mask = (order == s)
        if not iterative: prediction_mask = (order != 0)
        x = _sample(model, features, prediction_mask, x, mask, edge_index, edge_type).clone().detach()
        # remove predicted feature value from mask
        mask = mask.clone().detach()
        mask[prediction_mask] = 0.0

        if not iterative: break
    
    return x

def to_batch(data):
    has_target = "target" in dir(data)
    has_bug_mask = "bug_mask" in dir(data)

    x = data.x
    edge_type = data.edge_type
    if has_target:
        target = data.target

    device = data.x.device
    samples = data.to_data_list()
    edge_index_dim = 0
    x_dim = 0
    num_samples = len(samples)

    for i, data in enumerate(samples):
        edge_index = reflexive(bidirectional(data.edge_index), num_nodes=data.x.size(0))
        x_dim = max(data.x.size(0), x_dim)
        edge_index_dim = max(edge_index.size(1), edge_index_dim)

    x_batched = torch.ones([x_dim, num_samples, x.size(1)], dtype=torch.long, device=device) * -1
    if has_bug_mask:
        bug_mask_batched = torch.zeros([x_dim, num_samples, x.size(1)], dtype=torch.long, device=device)
    edge_index_batched = torch.ones([num_samples, 2, edge_index_dim], dtype=torch.long, device=device) * -1
    edge_type_batched = torch.ones([num_samples, edge_index_dim], dtype=torch.long, device=device) * -1
    edge_type_offset = 0

    for i, data in enumerate(samples):
        x_batched[:len(data.x),i] = data.x
        if has_bug_mask:
            bug_mask_batched[:len(data.x),i] = data.bug_mask
        
        edge_index = reflexive(bidirectional(data.edge_index), num_nodes=data.x.size(0))
        edge_index_batched[i,:,:edge_index.size(1)] = edge_index
        
        num_edges_in_original = data.edge_index.size(1)
        sample_edge_type = edge_type[edge_type_offset:edge_type_offset+num_edges_in_original]
        edge_type_batched[i,:edge_index.size(1)] = reflexive_bidirectional_edge_type(sample_edge_type, num_nodes=len(data.x))
        edge_type_offset += num_edges_in_original
    
    assert edge_type_offset == len(edge_type)

    return Data(x=x_batched, edge_index=edge_index_batched, edge_type=edge_type_batched, 
        bug_mask=bug_mask_batched if has_bug_mask else None, target=target if has_target else None)


def _sample_with_entropies(model, features, x, mask, edge_index, edge_type, num_parameters, inverted, random_order=False, mode="argmax"):
    x_emb = model.forward(x, mask, edge_index, edge_type, True)

    if mask.sum() == 0: return x, mask

    x_completed = x.clone().detach()
    x_entropies = torch.zeros_like(x, dtype=torch.float)
    x_predicted = torch.zeros_like(x)
    
    for feature in features:
        y_pred = model.decoder.forward(x_emb, feature.name)
        if y_pred.dim() < 3: y_pred = y_pred.unsqueeze(1)
        
        if mode == "argmax":
            values = y_pred.argmax(axis=-1)
        elif mode == "topk":
            probs = torch.exp(y_pred)
            k = min(list(probs.shape)[-1],4)
            topk_probs = torch.topk(probs, k, dim=-1).values[:,:,-1] # determine top 4 probs
            probs = probs * (probs >= topk_probs.unsqueeze(-1)) # set non-topk probs t 0
            probs = probs / probs.sum(axis=-1).unsqueeze(-1) # re-normalise
            values = torch.distributions.Categorical(probs=probs).sample()
        elif mode == "regular":
            probs = torch.exp(y_pred)
            values = torch.distributions.Categorical(probs=probs).sample()
        elif mode == "lowtemp":
            assert False, "lowtemp sampling not supported with sliced models"
            probs = torch.softmax(y_pred / 0.5, axis=-1)
            values = torch.distributions.Categorical(probs=probs).sample()
        elif mode == "hightemp":
            assert False, "hightemp sampling not supported with sliced models"
            probs = torch.softmax(y_pred / 1.5, axis=-1)
            values = torch.distributions.Categorical(probs=probs).sample()
        else:
            assert False, f"unknown sampling mode '{mode}'"

        entropies = torch.distributions.Categorical(logits=y_pred).entropy()

        #if values.dim() > 1: values = values.view(values.size(0))
        #print(entropies.shape)

        x_predicted[:,:,feature.idx] = values
        x_entropies[:,:,feature.idx] = entropies

    sign = -1 if inverted else 1
    entropy_values_to_sort = (x_entropies * mask + mask.logical_not() * sign * 5000).view(-1)
    if random_order:
        entropy_values_to_sort = (mask * torch.rand_like(x_entropies) + mask.logical_not() * sign * 5000).view(-1)
    permutation = torch.argsort(sign * entropy_values_to_sort, dim=0)
    order_by_entropy = torch.zeros_like(x_entropies.view(-1), dtype=torch.long)
    order_by_entropy[permutation] = torch.arange(1,permutation.numel()+1, device=order_by_entropy.device)
    order_by_entropy = order_by_entropy.view(x_entropies.shape)
    permutation = permutation.view(x_entropies.shape)

    last_index = min(num_parameters.long().item(), mask.sum().item())
    
    order_by_entropy = (order_by_entropy <= last_index) * order_by_entropy
    prediction_mask = (order_by_entropy != 0)
    
    assert torch.all(mask[prediction_mask]), "invalid prediction mask"

    """
    # sanity checking and logging
    m = order_by_entropy != 0
    s = torch.zeros_like(x_entropies[m])
    for pos,v in zip(order_by_entropy[m],x_entropies[m]):
        s[pos-1] = v
    low_val = s[0]
    hi_val = s[-1]
    assert low_val == x_entropies[prediction_mask].min()
    assert hi_val == x_entropies[prediction_mask].max()
    
    print("lowest entropy", x_entropies[prediction_mask].min())
    print("highest entropy", x_entropies[prediction_mask].max())
    """
    
    order_by_entropy = (order_by_entropy <= last_index) * order_by_entropy
    prediction_mask = order_by_entropy != 0
    
    assert prediction_mask.sum() <= mask.sum()
    #assert x_entropies[prediction_mask * mask].mean() <= x_entropies[prediction_mask.logical_not() * mask].mean() or (prediction_mask.logical_not() * mask).sum() == 0

    for feature in features:
        x_completed[:,:,feature.idx][prediction_mask[:,:,feature.idx]] = x_predicted[:,:,feature.idx][prediction_mask[:,:,feature.idx]]

    return x_completed, prediction_mask

def sample_by_entropy(model, features, data, mask, iterative=True, number_of_shots=1, inverted=False):
    x = data.x
    edge_index = data.edge_index
    edge_type = data.edge_type

    model.eval()

    num_parameters_per_shot = torch.ceil(mask.sum() / number_of_shots)

    x[mask] = -2
    x_before = x.clone().detach()
    original_mask = mask

    for s in range(1, number_of_shots + 1):
        x, prediction_mask = _sample_with_entropies(model, features, x.clone().detach(), mask, edge_index, edge_type, num_parameters_per_shot, inverted)
        x = x.clone().detach()
        # remove predicted feature value from mask
        mask = mask.clone().detach()
        mask[prediction_mask] = 0.0
    
    assert torch.all(original_mask[x_before != x]), "model completed non-masked parameters"
    assert torch.all((x_before != x)[original_mask]), "model did not complete all masked parameters"
    #print((x_before != x).sum(), original_mask.sum())

    return x

def sample_random_order(model, features, data, mask, iterative=True, number_of_shots=1, inverted=False, mode="argmax"):
    x = data.x
    edge_index = data.edge_index
    edge_type = data.edge_type

    model.eval()

    num_parameters_per_shot = torch.ceil(mask.sum() / number_of_shots)

    x[mask] = -2
    x_before = x.clone().detach()
    original_mask = mask

    for s in range(1, number_of_shots + 1):
        x, prediction_mask = _sample_with_entropies(model, features, x.clone().detach(), mask, edge_index, edge_type, num_parameters_per_shot, inverted, random_order=True, mode=mode)
        x = x.clone().detach()
        # remove predicted feature value from mask
        mask = mask.clone().detach()
        mask[prediction_mask] = 0.0

    for f in features:
        assert torch.all(x[:,:,f.idx] != -2), f"model did not complete all masked parameters for feature {f}"
        if not torch.all(x[:,:,f.idx] != -2):
            print(f)
            print(x_before[:,:,f.idx], x[:,:,f.idx])

    assert torch.all(original_mask[x_before != x]), "model completed non-masked parameters"
    assert torch.all((x_before != x)[original_mask]), "model did not complete all masked parameters"
    #print((x_before != x).sum(), original_mask.sum())

    return x

def sample_evolutionary(model, features, batched_data, data, mask, check_fct, device, mode="argmax"):
    x = batched_data.x
    edge_index = batched_data.edge_index
    edge_type = batched_data.edge_type

    model.eval()

    x[mask] = -2
    x_before = x.clone().detach()
    original_mask = mask

    x = _sample_evolutionary(model, features, x.clone().detach(), mask, edge_index, edge_type, check_fct, data, mode=mode)
    x = x.clone().detach()

    for f in features:
        assert torch.all(x[:,:,f.idx] != -2), f"model did not complete all masked parameters for feature {f}"
        if not torch.all(x[:,:,f.idx] != -2):
            print(f)
            print(x_before[:,:,f.idx], x[:,:,f.idx])

    assert torch.all(original_mask[x_before != x]), "model completed non-masked parameters"
    assert torch.all((x_before != x)[original_mask]), "model did not complete all masked parameters"
    #print((x_before != x).sum(), original_mask.sum())

    return x

def _sample_evolutionary(model, features, x, mask, edge_index, edge_type, check_fct, regular_data, mode="argmax"):
    x_emb = model.forward(x, mask, edge_index, edge_type, True)
    
    if mask.sum() == 0: return x, mask

    feature_distributions = [model.decoder.forward(x_emb, feature.name).unsqueeze(1) for feature in features]
    discounts = [torch.zeros_like(d) for d in feature_distributions]
    
    def check(x_candidate):
        data = regular_data.clone()
        data.x = x_candidate[:,0]
        return check_fct(data)

    def complete(synthesis_mask, x_candidate=None, sampling_mode="topk"):
        if x_candidate is None:
            x_candidate = x.clone().detach()

        x_predicted = torch.zeros_like(x)
        for feature, y_pred, feature_discount in zip(features, feature_distributions, discounts):
            feature_discount = torch.softmax(feature_discount, axis=-1)

            if sampling_mode == "argmax":
                values = y_pred.argmax(axis=-1)
            elif sampling_mode == "topk":
                probs = torch.exp(y_pred) * feature_discount
                k = min(list(probs.shape)[-1],4)
                topk_probs = torch.topk(probs, k, dim=-1).values[:,:,-1] # determine top 4 probs
                probs = probs * (probs >= topk_probs.unsqueeze(-1)) # set non-topk probs t 0
                probs = probs / probs.sum(axis=-1).unsqueeze(-1) # re-normalise
                values = torch.distributions.Categorical(probs=probs).sample()
            elif sampling_mode == "lowtemp":
                #assert False, "lowtemp sampling is disabled"
                probs = torch.softmax(y_pred / 0.5, axis=-1) * feature_discount
                values = torch.distributions.Categorical(probs=probs).sample()
            elif sampling_mode == "hightemp":
                assert False, "hightemp sampling is disabled"
                probs = torch.softmax(y_pred / 1.5, axis=-1) * feature_discount
                values = torch.distributions.Categorical(probs=probs).sample()
            elif sampling_mode == "sample":
                probs = torch.softmax(y_pred, -1) * feature_discount
                values = torch.distributions.Categorical(probs=probs).sample()
            elif sampling_mode == "uniform":
                values = torch.randint(0, 64, size=list(y_pred.shape)[0:-1])
            else:
                assert False, f"unknown sampling sampling_mode '{sampling_mode}'"

            x_predicted[:,:,feature.idx] = values.view(x_predicted[:,:,feature.idx].shape)

        for feature in features:
            x_candidate[:,:,feature.idx][synthesis_mask[:,:,feature.idx]] = x_predicted[:,:,feature.idx][synthesis_mask[:,:,feature.idx]]

        return x_candidate

    # determine mutation entropies
    def feature_entropy():
        feature_probs = [torch.softmax(y_pred, axis=-1) * torch.softmax(feature_discount, axis=-1) for y_pred,feature_discount in zip(feature_distributions, discounts)]
        feature_entropies = [torch.distributions.Categorical(probs=probs).entropy() for probs in feature_probs]
        mutation_entropy = torch.zeros_like(mask, dtype=torch.float)
        for feature, e in zip(features, feature_entropies): 
            mutation_entropy[:,:,feature.idx] = mask[:,:,feature.idx] * e.view(mask[:,:,feature.idx].shape)
        mutation_entropy = mutation_entropy / mutation_entropy.sum()
        return mutation_entropy
    
    global mutation_entropy
    mutation_entropy = feature_entropy()

    def discount(score, x_candidate):
        discount_factor = (1.0 - score) * 1
        for feature, d in zip(features, discounts):
            discount = F.one_hot(torch.maximum(torch.tensor(0),x_candidate)[:,:,feature.idx], num_classes=64) * discount_factor
            d -= discount * (x_candidate[:,:,feature.idx] != -1).unsqueeze(-1)
        
        global mutation_entropy
        mutation_entropy = feature_entropy()


    evolution_steps = 10
    def num_mutations(score):
        return max(1, 4 * int(score * mask.sum()))
        #if score < 0.8: return 15
        #elif score < 0.9: return 10
        #elif score < 0.95: return 5
        #else: return 4

    pool = CandidatePool(n=20)
    for i in range(pool.n):
        x_candidate = complete(mask, sampling_mode="lowtemp")
        pool.add(check(x_candidate), x_candidate)

    for i in range(evolution_steps):
        print(f"Evolution Step {i}: Best ", pool.candidates[0][0])
        if pool.candidates[0][0] == 1.0: break
        
        for score, candidate in pool.current_candidates:
            mutation_mask = torch.zeros_like(mask)

            for j in range(num_mutations(score)):
                mutation_index = torch.distributions.Categorical(mutation_entropy.view(-1)).sample()
                mutation_mask.view(-1)[mutation_index] = 1.0
            
            mutated = complete(mutation_mask, x_candidate=candidate.clone().detach(), sampling_mode="sample")

            # discount feature distributions            
            score = check(mutated)
            #discount(score, mutated)

            pool.add(score, mutated)
    
    return pool.candidates[0][1]

class CandidatePool:
    def __init__(self, n=4):
        self.candidates = []
        self.n = n
    
    def add(self, score, candidate):
        self.candidates.append((score,candidate))
        self.candidates.sort(key=lambda e: e[0], reverse=True)
        self.candidates = self.candidates[0:self.n]

    @property
    def current_candidates(self):
        return [e for e in self.candidates]