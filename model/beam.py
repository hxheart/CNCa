import torch

def get_prediction_mask(x_predicted, n, mask):
    entropy_values_to_sort = (mask * torch.rand(x_predicted.shape, device=x_predicted.device) + mask.logical_not() * 5000).view(-1)
    permutation = torch.argsort(entropy_values_to_sort, dim=0)
    order_by_entropy = torch.zeros_like(x_predicted.view(-1), dtype=torch.long)
    order_by_entropy[permutation] = torch.arange(1,permutation.numel()+1, device=order_by_entropy.device)
    order_by_entropy = order_by_entropy.view(x_predicted.shape)
    
    last_index = min(n, mask.sum().item())
    
    order_by_entropy = (order_by_entropy <= last_index) * order_by_entropy
    prediction_mask = (order_by_entropy != 0)

    assert torch.all(mask[prediction_mask]), "invalid prediction mask"
    assert prediction_mask.sum() <= mask.sum()

    return prediction_mask


def beam_step(model, features, x, mask, edge_index, edge_type, num_parameters, beam_k=4, topk_k=8, mode="argmax"):
    x_emb = model.forward(x, mask, edge_index, edge_type, True)

    if mask.sum() == 0: 
        yield x.clone().detach(), torch.zeros_like(mask), 0.0
        return

    original_mask = mask

    for i in range(beam_k):
        mask = original_mask.clone().detach()

        x_completed = x.clone().detach()
        x_logprobs = torch.zeros_like(x, dtype=torch.float)
        x_predicted = torch.zeros_like(x)
        
        for feature in features:
            y_pred = model.decoder.forward(x_emb, feature.name)
            if y_pred.dim() < 3: y_pred = y_pred.unsqueeze(1)
            
            if mode == "topk":
                values = y_pred.argmax(axis=-1)
                
                probs = torch.exp(y_pred)
                k = min(list(probs.shape)[-1],topk_k)
                topk_probs = torch.topk(probs, k, dim=-1).values[:,:,-1] # determine top  probs
                probs = probs * (probs >= topk_probs.unsqueeze(-1)) # set non-topk probs t 0
                probs = probs / probs.sum(axis=-1).unsqueeze(-1) # re-normalise
                distribution = torch.distributions.Categorical(probs=probs)
                values = distribution.sample()
            elif mode == "argmax":
                distribution = torch.distributions.Categorical(logits=y_pred)
                values = distribution.sample()
            else:
                raise ValueError(f"mode {mode} not supported")

            x_predicted[:,:,feature.idx] = values
            x_logprobs[:,:,feature.idx] = distribution.log_prob(values)

        prediction_mask = get_prediction_mask(x_predicted, num_parameters.long().item(), mask)
        
        for feature in features:
            x_completed[:,:,feature.idx][prediction_mask[:,:,feature.idx]] = x_predicted[:,:,feature.idx][prediction_mask[:,:,feature.idx]]

            # print(x_completed[:,:,feature.idx][prediction_mask[:,:,feature.idx]])

        sample_logprob = (x_logprobs * prediction_mask).sum() 

        yield x_completed, prediction_mask, sample_logprob

class BeamPool:
    def __init__(self, n):
        self.n = n
        self.pool = []
    
    def add(self, x, logprob):
        if len(self.pool) < self.n:
            self.pool.append((x, logprob))
        else:
            if logprob > self.pool[-1][1]:
                self.pool[-1] = (x, logprob)
                self.pool.sort(key=lambda x: x[1], reverse=True)
        self.pool = self.pool[:self.n]

    def __len__(self):
        return len(self.pool)
    
    def __iter__(self):
        return iter(self.pool)

def beam_search(model, features, data, original_mask, iterative=True, number_of_shots=1, inverted=False, mode="argmax", beam_n=8, beam_k=16):
    x = data.x
    edge_index = data.edge_index
    edge_type = data.edge_type

    model.eval()

    num_parameters_per_shot = torch.ceil(original_mask.sum() / number_of_shots)

    x[original_mask] = -2
    x_before = x.clone().detach()

    pool = BeamPool(beam_n)
    pool.add((original_mask, x), 0)

    for s in range(1, number_of_shots + 1):
        next_pool = BeamPool(n=pool.n)

        for (branch_mask, x), sample_logprob in pool:
            for x, prediction_mask, step_logprob in list(beam_step(model, features, x.clone().detach(), branch_mask, edge_index, edge_type, num_parameters_per_shot, beam_k=beam_k, mode=mode)):
                x = x.clone().detach()
                # remove predicted feature value from mask
                mask = branch_mask.clone().detach()
                mask[prediction_mask] = 0.0

                assert torch.all(x[mask.logical_not()] != -2)

                next_pool.add((mask, x), sample_logprob + step_logprob)

        print("Candidates in pool", len(next_pool))
        pool = next_pool

    assert len(pool.pool) != 0, "beam pool is empty"

    (mask, x), _ = pool.pool[0]

    for f in features:
        if not torch.all(x[:,:,f.idx] != -2):
            print(f)
            print(x_before[:,:,f.idx], x[:,:,f.idx])
        assert torch.all(x[:,:,f.idx] != -2), f"model did not complete all masked parameters for feature {f}"

    assert torch.all(original_mask[x_before != x]), "model completed non-masked parameters"
    assert torch.all((x_before != x)[original_mask]), "model did not complete all masked parameters"
    #print((x_before != x).sum(), original_mask.sum())

    return x