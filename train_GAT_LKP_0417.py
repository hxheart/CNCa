import sys
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

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
from coders_LKP import *
from sampling import sample_random_order

from factbase import FactBase, Constant
from nutils import *
from semantics import FactBaseSemanticsDataset

from snapshot import ModelSnapshot
from bgp_semantics import BgpSemantics

import argparse

from inject_anomaly import *

# ————————————————————————————————————————————————————————————————————————————————


NUM_EDGE_TYPES = 4

NUM_ITERATION_MESSAGE = 3  # message passing 的时候，信息跳了几次
NUM_TRANSFORMER_ENCODER = 2
NUM_TRANSFORMER_DECODER = 2

NUM_LEN_DATASET = 1024*4
NUM_LEN_DATASET_EVAL = 100  # max(100, int(NUM_LEN_DATASET * 0.25))

LEARNING_RATE_DECAY_EPOCHS = 10

LEARNING_RATE_INIT = 5e-4
LEARNING_RATE_END  = 1e-4

WEIGHT_DECAY = 1e-5

batch_size_train = 4
num_samples_per_epoch_train = 16


# ————————————————————————————————————————————————————————————————————————————————

def make_noisy_reference(x, noise_rate=0.02):
    noisy = x.clone()
    valid_mask = (x[:, 0, :] != -1)
    valid_positions = valid_mask.nonzero()

    num_to_corrupt = int(len(valid_positions) * noise_rate)
    if num_to_corrupt == 0:
        return noisy

    chosen = valid_positions[torch.randperm(len(valid_positions))[:num_to_corrupt]]

    for node_idx, feat_idx in chosen:
        col_vals = x[:, 0, feat_idx]
        legal_vals = col_vals[col_vals != -1]
        if len(legal_vals) > 1:
            pick = legal_vals[torch.randint(len(legal_vals), (1,)).item()]
            noisy[node_idx, 0, feat_idx] = pick

    return noisy

# ============================================================
#  Feature injection configuration
# ============================================================
#
#  Strategy summary:
#    swap  — swap values between two nodes. Used for predicate and
#            holds: values are always in-distribution, anomaly is
#            only detectable via neighbourhood consistency.
#    vocab — replace with a different value from the known legal set.
#            Used for small discrete features (ORIGIN_TYPE, IS_EBGP).
#    bias  — add a small signed offset. Used for all numerical
#            bgp_route and connected parameters, because:
#              - value distributions may not be rich enough for swap
#                to reliably produce genuine changes (e.g. MED is
#                mostly 0, SPEAKER_ID may have repeated values)
#              - bias magnitude is tuned per feature to stay within
#                a plausible range, so the anomaly cannot be spotted
#                from the node value alone and requires message
#                passing to detect.



# ============================================================
#  Per-strategy inject functions
# ============================================================

def _inject_swap(x, corrupted_x, target_binary, f, nodes_with_feature, error_rate):
    """
    Swap values between randomly chosen pairs of nodes.

    The injected value is always drawn from the real distribution,
    so the anomaly is invisible from the node's own value alone.
    Detection requires comparing the node against its neighbours
    via message passing.

    A swap is only committed when the two chosen nodes carry
    distinct values, guaranteeing a genuine change.
    num_to_attempt is halved because one swap corrupts two nodes,
    so we still corrupt ~(error_rate * N) nodes in expectation.
    """
    num_to_attempt = int(len(nodes_with_feature) * error_rate / 2)
    if num_to_attempt == 0:
        return

    for _ in range(num_to_attempt):
        idx_a, idx_b = nodes_with_feature[torch.randperm(len(nodes_with_feature))[:2]]
        val_a = x[idx_a, 0, f.idx]
        val_b = x[idx_b, 0, f.idx]
        if val_a != val_b:
            corrupted_x[idx_a, 0, f.idx] = val_b
            corrupted_x[idx_b, 0, f.idx] = val_a
            target_binary[idx_a, 0, f.idx] = 1
            target_binary[idx_b, 0, f.idx] = 1


def _inject_vocab(x, corrupted_x, target_binary, f, nodes_with_feature, error_rate):
    """
    Replace each selected node's value with a different value drawn
    uniformly from the known legal vocabulary.

    Used for ORIGIN_TYPE and IS_EBGP whose value sets are so small
    that swap would too often pick the same value and do nothing.
    """
    vocab = FEATURE_VOCAB[f.name]
    num_to_corrupt = int(len(nodes_with_feature) * error_rate)
    if num_to_corrupt == 0:
        return

    corrupt_indices = nodes_with_feature[
        torch.randperm(len(nodes_with_feature))[:num_to_corrupt]
    ]
    original_values = x[corrupt_indices, 0, f.idx]

    new_vals = original_values.clone()
    for i, orig in enumerate(original_values):
        candidates = [v for v in vocab if v != orig.item()]
        if candidates:
            pick = torch.randint(0, len(candidates), (1,)).item()
            new_vals[i] = candidates[pick]

    corrupted_x[corrupt_indices, 0, f.idx] = new_vals
    target_binary[corrupt_indices, 0, f.idx] = 1


def _inject_bias(x, corrupted_x, target_binary, f, nodes_with_feature, error_rate):
    """
    Add a small signed offset to the selected nodes' values.

    Design choices:
      - Per-feature magnitude range from FEATURE_BIAS_RANGE, tuned
        to each parameter's realistic value distribution.
      - Random sign {-1, +1}: prevents the model learning the
        shortcut "value increased → anomaly".
      - clamp(min=0): all target parameters are non-negative.
      - Final equality check: if clamping cancelled the offset
        (e.g. original=0, sign=-1), nudge by +1 to guarantee a
        genuine change.
    """
    num_to_corrupt = int(len(nodes_with_feature) * error_rate)
    if num_to_corrupt == 0:
        return

    corrupt_indices = nodes_with_feature[
        torch.randperm(len(nodes_with_feature))[:num_to_corrupt]
    ]
    original_values = x[corrupt_indices, 0, f.idx]

    bias_min, bias_max = FEATURE_BIAS_RANGE.get(f.name, (1, 3))
    bias = torch.randint(bias_min, bias_max + 1,
                         (num_to_corrupt,), device=x.device).long()
    sign = torch.randint(0, 2, (num_to_corrupt,), device=x.device) * 2 - 1  # {-1, +1}

    new_vals = (original_values + sign * bias).clamp(min=0)

    # guarantee a genuine change after clamping
    still_equal = (new_vals == original_values)
    new_vals[still_equal] += 1

    corrupted_x[corrupt_indices, 0, f.idx] = new_vals
    target_binary[corrupt_indices, 0, f.idx] = 1


# ============================================================
#  Dispatcher
# ============================================================

_INJECT_FN = {
    "swap":  _inject_swap,
    "vocab": _inject_vocab,
    "bias":  _inject_bias,
}


def inject_anomaly(x, synthesised_features, error_rate):
    """
    Inject misconfigurations into a batch of node features.

    Args:
        x                   : node feature tensor, shape (N, 1, D)
        synthesised_features : list of feature descriptors (each has
                               .name and .idx)
        error_rate           : fraction of eligible nodes to corrupt,
                               in (0, 1]

    Returns:
        corrupted_x    : corrupted copy of x, same shape
        target_binary  : long tensor, same shape as x
                           1  → node/feature was corrupted
                           0  → node/feature is normal
                          -1  → feature does not apply to this node
    """
    corrupted_x = x.clone()
    target_binary = torch.full_like(x, -1).long()

    for f in synthesised_features:
        mask = (x[:, 0, f.idx] != -1)
        nodes_with_feature = torch.where(mask)[0]
        if len(nodes_with_feature) == 0:
            continue

        # all eligible nodes start as normal (label 0)
        target_binary[nodes_with_feature, 0, f.idx] = 0

        mode = FEATURE_INJECTION.get(f.name, "bias")
        inject_fn = _INJECT_FN[mode]
        inject_fn(x, corrupted_x, target_binary, f, nodes_with_feature, error_rate)

    return corrupted_x, target_binary


# def inject_anomaly(x, synthesised_features, error_rate):  # 换位置
#     corrupted_x = x.clone()
#     target_binary = torch.full_like(x, -1).long()
#
#     for f in synthesised_features:
#         mask = (x[:, 0, f.idx] != -1)
#         nodes = torch.where(mask)[0]
#         if len(nodes) < 2: continue
#
#         target_binary[nodes, 0, f.idx] = 0  # 默认全为正常
#
#         num_to_attempt = int(len(nodes) * error_rate / 2)  # 尝试交换的对数
#
#         for _ in range(num_to_attempt):
#             # 随机抽两个点
#             idx_a, idx_b = nodes[torch.randperm(len(nodes))[:2]]
#
#             val_a = x[idx_a, 0, f.idx]
#             val_b = x[idx_b, 0, f.idx]
#
#             # 只有真的不一样才换，且标记为 1
#             if val_a != val_b:
#                 corrupted_x[idx_a, 0, f.idx] = val_b
#                 corrupted_x[idx_b, 0, f.idx] = val_a
#                 target_binary[idx_a, 0, f.idx] = 1
#                 target_binary[idx_b, 0, f.idx] = 1
#             # 如果一样，就什么都不做，target 保持为 0
#
#     return corrupted_x, target_binary


# def inject_anomaly(x, synthesised_features, error_rate): # 改成其他值
#     corrupted_x = x.clone()
#     target_binary = torch.full_like(x, -1).long()
#
#     for f in synthesised_features:
#         mask = (x[:, 0, f.idx] != -1)
#         nodes_with_feature = torch.where(mask)[0]
#         if len(nodes_with_feature) == 0: continue
#
#         target_binary[nodes_with_feature, 0, f.idx] = 0
#         num_to_corrupt = int(len(nodes_with_feature) * error_rate)
#
#         if num_to_corrupt > 0:
#             corrupt_indices = nodes_with_feature[torch.randperm(len(nodes_with_feature))[:num_to_corrupt]]
#
#             # --- 关键改变：错误值也在 [10, 20] 之间 ---
#             # 这样模型就不能通过数值大小来判断了
#             original_values = x[corrupt_indices, 0, f.idx]
#
#             # 生成一个 10-20 之间的随机数
#             random_vals = torch.randint(1, 5, (num_to_corrupt,), device=x.device).long()
#             # random_vals = torch.randint(10, 21, (num_to_corrupt,), device=x.device).long()  # 去掉 .float()
#
#             # 确保随机生成的值不等于原始值
#             equal_mask = (random_vals == original_values)
#             random_vals[equal_mask] += 1
#             random_vals[random_vals > 50] = 50  # 越界处理
#
#             corrupted_x[corrupt_indices, 0, f.idx] = random_vals
#             target_binary[corrupt_indices, 0, f.idx] = 1
#
#     return corrupted_x, target_binary



# ————————————————————————————————————————————————————————————————————————————————

def mask_parameters(x, decls):  # x is the PyG node features tensor;
    mask = torch.zeros_like(x)  # create tensor

    # mask OSPF
    mask[:, :, feature("predicate_connected_arg2").idx] = (x[:, :, feature("predicate_connected_arg2").idx] > -1)

    # mask BGP
    masked_bgp_route_args = [2, 3, 5]
    for i in masked_bgp_route_args:
        idx = feature("predicate_bgp_route_arg" + str(i)).idx
        mask[:, :, idx] = (x[:, :, idx] > -1)

    return mask.bool()  # return tensor (NN learns to predict the complete tensor from the masked tensor)


def combine_dict(dicts):
    flatten = lambda seq: reduce(lambda a, b: a.union(b), seq, set())
    keys = flatten([set(d.keys()) for d in dicts])
    values_for_key = lambda k: [d[k] for d in dicts if k in d.keys()]
    return dict([(k, torch.tensor(values_for_key(k))) for k in keys])


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


class EdgeTypeTransformerLayer(torch.nn.Module):  #
    def __init__(self, hidden_dim, d_inner, num_edge_types, n_heads=8, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_inner = d_inner

        # initialise layers
        self.layers = []
        for j in range(num_edge_types):
            l = GATConv(self.hidden_dim, self.hidden_dim, n_heads, False)
            # l = GCNConv(self.hidden_dim, self.hidden_dim, aggr="max")
            # l = MaxGraphLayer(self.hidden_dim)
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
            index = edge_index[:, indices]
            return index

        def layer(l, x, ei):
            return l(x, ei)

        # edge type attention
        x2 = torch.stack(
            [layer(l, x.view(-1, self.hidden_dim), edge_index_for_type(t)) for t, l in enumerate(self.layers)],
            axis=0).sum(axis=0)
        x = x + self.drop1(x2)
        x = self.norm1.forward(x.view(-1, self.hidden_dim)).view(x.shape)
        x = self.linear2(self.drop(torch.relu(self.linear1(x))))
        x = self.norm2(x.view(-1, self.hidden_dim)).view(x.shape)

        return x


class EdgeTypeGraphTransformer(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers, num_edge_types):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.propagation_layers = [EdgeTypeTransformerLayer(hidden_dim, 4 * hidden_dim, num_edge_types, 8) for i in
                                   range(num_layers)]
        for i, l in enumerate(self.propagation_layers):
            self.add_module("prop_layer_" + str(i), l)

    def forward(self, x, edge_index, edge_type):
        for l in self.propagation_layers:
            x = l.forward(x, edge_index, edge_type)
            x = torch.relu(x)
        return x


class PredicateGraphEmbedding(torch.nn.Module):  # 它负责初始的特征处理
    def __init__(self, features, hidden_dim, num_edge_types, excluded_feature_indices=set(), num_layers=6):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types

        self.encoder = NodeFeatureEmbedding(hidden_dim, features, excluded_feature_indices=excluded_feature_indices)
        self.decoder = NodeFeatureDecoder(hidden_dim, features, excluded_feature_indices=excluded_feature_indices)
        # 它里面有一个 self.decoder，类型是 NodeFeatureDecoder。这就是我们要找的 Readout 逻辑的家。

    def forward(self, x, mask, edge_index, edge_type, reliable_masking, x_clean=None):
        return self.encoder.forward(x, mask, reliable_masking, positional_encoding=False, x_clean=x_clean)


# 模型的调用链路是：Model → EdgeTypeGraphTransformer → EdgeTypeTransformerLayer
class Model(torch.nn.Module):
    def __init__(self, features, hidden_dim, num_edge_types, excluded_feature_indices):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types

        self.embedding = PredicateGraphEmbedding(features, hidden_dim, self.num_edge_types, excluded_feature_indices)
        self.decoder = self.embedding.decoder

        self.transformer_encoder = EdgeTypeGraphTransformer(self.hidden_dim, NUM_TRANSFORMER_ENCODER, num_edge_types)
        self.transformer_decoder = EdgeTypeGraphTransformer(self.hidden_dim, NUM_TRANSFORMER_DECODER, num_edge_types)

        self.num_iterations = NUM_ITERATION_MESSAGE

    def add_noise(self, x):
        noise = torch.randn(x.shape, device=x.device)
        noise_mask = torch.zeros([self.hidden_dim], device=x.device)
        noise_mask[0:int(self.hidden_dim / 2)] = 1

        return x + noise_mask.unsqueeze(0) * noise

    def forward(self, x, mask, edge_index, edge_type, reliable_masking, x_clean=None):
        # adjust shape when called from eval/serve script
        if edge_index.dim() == 3:
            assert edge_index.size(0) == 1
            assert edge_type.size(0) == 1
            edge_index = edge_index[0]
            edge_type = edge_type[0]

        assert x.size(1) == 1

        # 1. 把 15 维变成高维向量
        x = self.embedding.forward(x, mask, edge_index, edge_type, reliable_masking, x_clean=x_clean)[:, 0]

        # 2. 进 Transformer 跑 2 层
        x = self.transformer_encoder.forward(x, edge_index, edge_type)

        # 3. 加点噪声
        x = self.add_noise(x)

        # 4. 再跑 6 层 Transformer 迭代
        for i in range(self.num_iterations):
            x = self.transformer_decoder.forward(x, edge_index, edge_type) + x

        # 5. 返回每个点的最终向量
        return x  # 它的 forward 函数最后一行返回的是 x。这个 x 是经过多轮迭代后的 Hidden State（隐藏表示）


if __name__ == '__main__':
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=128)
        parser.add_argument("--epochs", dest="epochs", type=int, default=NUM_TRAINING_EPOCH)
        parser.add_argument("--checkpoint", dest="checkpoint", type=str, default=None)
        parser.add_argument("--run-id", dest="run_id", type=int, default=None)
        return parser.parse_args()


    args = get_args()  # Parse hyperparameters
    HIDDEN_DIM = args.hidden_dim

    # ---- device selection ----

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("\n Running on", device)
    snapshot = ModelSnapshot(__file__)

    # ---- end of device selection ----

    sem = BgpSemantics()  # sem: topology/config → run Prot → produce spec facts
    dataset = FactBaseSemanticsDataset(sem, "bgp-ospf-dataset-sub",
                                       num_samples=NUM_LEN_DATASET)  # !!!! generate (or load) the dataset
    # print("Dataset Size", len(dataset))

    data = dataset[0]
    # print('\n ===> data.x[xxx] is:\n', data.x[108])
    # print("\n The 0 column (index 0):")
    # print(data.x[:, 0])
    # # 打印第2列（即 idx=2 的特征：predicate）
    # print("\n The 2nd column (index 2):")
    # print(data.x[:, 2])
    #
    # print("\n The 5 column (index 5):")
    # print(data.x[:, 5])

    training_dataset, validation_dataset = dataset[NUM_LEN_DATASET_EVAL:], dataset[:NUM_LEN_DATASET_EVAL]
    training_eval_dataset = dataset[NUM_LEN_DATASET_EVAL:2 * NUM_LEN_DATASET_EVAL]

    print("\n Validation Dataset Size", len(validation_dataset))

    predicate_declarations = sem.decls()
    for decl in predicate_declarations.values():
        constant_types = [at for at in decl.arg_types if at is Constant]
        assert len(
            constant_types) <= NUM_EDGE_TYPES, f"declaration {decl} requires more than {NUM_EDGE_TYPES} edge types"
    prog = FactBase(predicate_declarations)  # build a new blank FactBase
    feature = prog.feature_registry.feature

    excluded_feature_indices = set([])
    features = prog.feature_registry.get_all_features()
    print(prog.predicate_declarations)
    print("\nfeatures:", features)

    # 添加这些打印
    print("\n===== All Features =====")
    for i, f in enumerate(features):
        print(f"Feature {i}: {f.name} (idx={f.idx})")
    print("========================\n")

    model = Model(features, HIDDEN_DIM, NUM_EDGE_TYPES, excluded_feature_indices).to(device)
    model.feature = feature

    if args.checkpoint is not None:
        state_dict, hidden_dim, _, _ = torch.load(args.checkpoint, map_location=device)
        assert HIDDEN_DIM == hidden_dim, f"dimension mismatch configured {HIDDEN_DIM} vs. state dict {hidden_dim}"
        model.load_state_dict(state_dict)
        print("restored checkpoint from ", args.checkpoint)

    writer = snapshot.writer()
    # optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_INIT, weight_decay=WEIGHT_DECAY)  # get_std_opt(model)
    # print(len(list(model.parameters())))
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE_INIT,
        weight_decay=WEIGHT_DECAY
    )

    # 第一段：前10个epoch，从2e-4线性降到1e-4
    scheduler_decay = torch.optim.lr_scheduler.LinearLR(
        optimiser,
        start_factor=1.0,
        end_factor=LEARNING_RATE_END / LEARNING_RATE_INIT,  # = 0.5
        total_iters=LEARNING_RATE_DECAY_EPOCHS
    )

    # 第二段：之后所有epoch，lr保持不变（乘以1.0）
    scheduler_flat = torch.optim.lr_scheduler.ConstantLR(
        optimiser,
        factor=LEARNING_RATE_END / LEARNING_RATE_INIT,  # 维持在0.5倍初始值
        last_epoch=-1
    )

    # 拼起来：前10个epoch用第一段，第10个epoch之后切换到第二段
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimiser,
        schedulers=[scheduler_decay, scheduler_flat],
        milestones=[LEARNING_RATE_DECAY_EPOCHS]  # 第10个epoch切换
    )

    pool = Pool(processes=8)
    # program_checker_training = AsyncFactBaseChecker(pool)
    # program_checker_validation = AsyncFactBaseChecker(pool)
    num_eval_samples = 10


    def best_sample_mean(res):
        if res.numel() == 0: return 0
        res = res.view(-1, num_eval_samples)
        res = res.max(axis=1).values
        return res.mean()


    def on_evaluation_step_finish(prefix):
        def handler(step, res):
            for key in res.keys():
                writer.add_scalar(f"{prefix}/WeightSynthesis/Consistency/{key}", best_sample_mean(res[key]),
                                  global_step=step)

        return handler


    # program_checker_training.on_step_finish = on_evaluation_step_finish("Training")
    # program_checker_validation.on_step_finish = on_evaluation_step_finish("Validation")

    # List of features to predict/synthesize # 告诉模型：虽然我有 15 个特征，但我只打算训练这 4 个。
    synthesised_features = [
        # feature("type"),
        # feature("id"),
        # feature("predicate"),
        # feature("holds"),
        # # # -----
        # # bgp_route: LP x AS x -OT x MED x -IS_EBGP x -SPEAKER_ID
        feature("predicate_bgp_route_arg2"),  # BGP LP
        feature("predicate_bgp_route_arg3"),  # BGP AS
        # feature("predicate_bgp_route_arg4"),  # BGP ORIGIN_TYPE
        feature("predicate_bgp_route_arg5"),  # BGP MED
        # feature("predicate_bgp_route_arg6"),  # BGP IS_EBGP
        # feature("predicate_bgp_route_arg7"),  # SPEAKER_ID
        feature("predicate_connected_arg2"),  # OSPF weights
        # -----
        # feature("predicate_external"),
        # feature("predicate_network"),
        # feature("predicate_route_reflector"),
        # feature("predicate_router"),
    ]
    # if args.feature is not None:
    #     synthesised_features = [feature(args.feature)]
    # else:
    #     synthesised_features = [feature("predicate_bgp_route_arg2")]  # 默认值

    # p = 0.85
    # batch_size = 8
    # num_samples_per_epoch = 32
    num_batches_per_epoch = int(num_samples_per_epoch_train / batch_size_train)

    # 初始化全局历史记录（总账本）
    loss_history, accuracy_history, f1_history = [], [], []
    # 初始化特征历史记录（分账本）
    feature_loss_history = {f.name: [] for f in synthesised_features}
    feature_accuracy_history = {f.name: [] for f in synthesised_features}
    feature_f1_history = {f.name: [] for f in synthesised_features}

    # --- 新增: DWA 记录器 ---
    # 记录每个特征在上两个 Epoch 的平均 Loss，初始化为 1.0
    last_epoch_losses = np.ones(len(synthesised_features))
    last_last_epoch_losses = np.ones(len(synthesised_features))
    # 初始化权重为 1.0
    task_weights = np.ones(len(synthesised_features))
    T = 2.0  # 温度系数，T 越大权重越平滑，T 越小权重越倾向于难的任务

    for epoch in tqdm(range(args.epochs), leave=False):
        training_loader = DataLoader(training_dataset, batch_size=batch_size_train, shuffle=True)
        step_writer = StepWriter(writer, epoch)

        model.train()  # 设置模型为训练模式; # PyTorch的模式切换，不是执行训练
        epoch_loss, epoch_acc, epoch_f1 = 0.0, 0.0, 0.0
        num_batches = 0

        # ===== 这个循环才是真正的训练！=====
        for i, batch in tqdm(enumerate(training_loader), leave=False, total=num_batches_per_epoch,
                             desc=f"Epoch {epoch}"):
            if i > num_batches_per_epoch: break

            # 1. 准备数据
            optimiser.zero_grad()  # 清空梯度
            batch = batch.to(device)
            batch.x = batch.x.unsqueeze(1)
            batch.edge_index = reflexive(bidirectional(batch.edge_index), num_nodes=batch.x.size(0))
            batch.edge_type = reflexive_bidirectional_edge_type(batch.edge_type, batch.x.size(0))

            # ===== 新增: 注入错误 =====
            corrupted_x, target_binary = inject_anomaly(batch.x, synthesised_features, error_rate=ERROR_RATE)
            mask = mask_parameters(batch.x, prog.predicate_declarations)
            # target = mask_node_features(batch.x, mask.logical_not())

            # ===== 新增: 创建假mask (全False,表示不mask任何东西) =====
            fake_mask = torch.zeros_like(corrupted_x, dtype=torch.bool)

            # 3. 前向传播，一次前向传播处理整个图。这个图里面有所有的节点，比如，307个。
            # x_emb = model.forward(batch.x, mask, batch.edge_index, batch.edge_type, False)

            noisy_x_clean = make_noisy_reference(batch.x, noise_rate=0.02)

            # ===== 修改: 用corrupted_x和fake_mask =====
            x_emb = model.forward(
                corrupted_x,        # 注入异常后的图（模型主输入）
                fake_mask,          # 不 mask 任何东西
                batch.edge_index,
                batch.edge_type,
                False,              # reliable_masking=False
                x_clean=noisy_x_clean,    # 干净图作为参考信号（选项 B 双路 encoder）
            )

            batch_total_loss = torch.tensor(0.0, device=device)
            batch_acc_sum, batch_f1_sum = 0.0, 0.0

            for idx, f in enumerate(synthesised_features):  # f 是要预测的特征，比如 "predicate_connected_arg2"
                f_loss = model.decoder.loss(x_emb, target_binary, f.name)
                # batch_total_loss += f_loss
                batch_total_loss += f_loss * task_weights[idx]

                # 如果当前特征是 holds，则放大其 Loss
                # if f.name == "holds":
                #     batch_total_loss += f_loss * 1
                # else:
                #     batch_total_loss += f_loss

                # 新增：计算每个特征的准确率
                with torch.no_grad():  # 计算准确率时不需要梯度
                    f_acc = model.decoder.accuracy(x_emb, target_binary, f.name)
                    f_f1 = model.decoder.decoders[f.name].f1(x_emb, target_binary[:, :, f.idx])

                    # 记录到“分账本”
                    feature_loss_history[f.name].append(f_loss.item())
                    feature_accuracy_history[f.name].append(f_acc.item())
                    feature_f1_history[f.name].append(f_f1.item())

                    batch_acc_sum += f_acc
                    batch_f1_sum += f_f1

            if torch.any(torch.isnan(f_loss)):
                print("isnan")  # 检测训练是否出错 --> 具体是检查损失值是否变成了 NaN (Not a Number)

            # 5. 反向传播 + 更新参数
            batch_total_loss.backward()
            optimiser.step()  # 更新参数（真正的"训练"）

            # 4. 记录到“总账本” (Batch 级别)
            loss_history.append(batch_total_loss.item())
            accuracy_history.append((batch_acc_sum / len(synthesised_features)).item())
            f1_history.append((batch_f1_sum / len(synthesised_features)).item())

            # 累加用于 Epoch 打印
            epoch_loss += batch_total_loss.item()
            epoch_acc += accuracy_history[-1]
            epoch_f1 += f1_history[-1]
            num_batches += 1

            # 计算当前 Epoch 每个特征的平均 Loss
            current_avg_losses = []
            for f in synthesised_features:
                # 从你的 feature_loss_history 中取当前 Epoch 的最后 N 个 batch 的平均值
                avg_l = np.mean(feature_loss_history[f.name][-num_batches:])
                current_avg_losses.append(avg_l)
            current_avg_losses = np.array(current_avg_losses)

            # --- DWA 核心逻辑 ---
            if epoch >= 2:  # 至少有三个数据点才能计算变化率
                # 计算下降率 r: 越接近 1 说明降得越慢 (甚至上升)
                r = current_avg_losses / last_epoch_losses

                # 使用 Softmax 逻辑分配权重
                exp_r = np.exp(r / T)
                task_weights = (exp_r / np.sum(exp_r)) * len(synthesised_features)

                # 打印一下，让你知道模型现在在重点关注谁
                # print(f"Next Epoch Weights: {dict(zip([f.name for f in synthesised_features], task_weights))}")

            # 滚动更新历史记录
            last_last_epoch_losses = last_epoch_losses.copy()
            last_epoch_losses = current_avg_losses.copy()

            scheduler.step()

        # =============================== 每个 Epoch 打印汇总 ============================
        print(f"\n" + "=" * 70)
        print(f"📊 Epoch {epoch} Overall Summary:")
        print(
            f"Total Loss: {epoch_loss / num_batches:.4f} | Total Acc: {epoch_acc / num_batches:.4f} | Total F1: {epoch_f1 / num_batches:.4f}")
        print("-" * 70)

        # 打印“分账本”表头
        print(f"{'Feature Name':<35} | {'Loss':<8} | {'Acc':<8} | {'F1':<8}")
        print("-" * 70)

        for f in synthesised_features:
            # 从你已经记录好的历史记录中，取出当前 Epoch 的最后 num_batches 个点求平均
            f_name = f.name
            avg_f_loss = sum(feature_loss_history[f_name][-num_batches:]) / num_batches
            avg_f_acc = sum(feature_accuracy_history[f_name][-num_batches:]) / num_batches
            avg_f_f1 = sum(feature_f1_history[f_name][-num_batches:]) / num_batches

            # 逻辑判断：如果 F1 仍然在 0.5 左右徘徊，打个警示灯
            status = "❌" if avg_f_f1 < 0.8 else "✅"

            # 针对你关注的重点特征，加个标记
            highlight = "⬅️ Target!" if "connected_arg2" in f_name or "bgp_route_arg5" in f_name else ""

            print(f"{f_name:<35} | {avg_f_loss:<8.4f} | {avg_f_acc:<8.4f} | {avg_f_f1:<8.4f} {status} {highlight}")

        print("=" * 70 + "\n")

    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    # ——————————————————————————————————————————————————
    # 1. 创建输出目录
    # ——————————————————————————————————————————————————
    ALGO_NAME = "GAT_LKP"

    base_dir = f"results_train/{ALGO_NAME}"
    os.makedirs(base_dir, exist_ok=True)

    if args.run_id is not None:
        out_dir = f"{base_dir}/{ALGO_NAME}_{args.run_id}"
    else:
        existing = [d for d in os.listdir(base_dir) if d.startswith(ALGO_NAME + "_")]
        nums = [int(d.split("_")[-1]) for d in existing if d.split("_")[-1].isdigit()]
        next_run = max(nums, default=0) + 1
        out_dir = f"{base_dir}/{ALGO_NAME}_{next_run}"

    os.makedirs(out_dir, exist_ok=True)

    # ——————————————————————————————————————————————————
    # 2. 保存 Loss / Accuracy / F1 到 CSV
    # ——————————————————————————————————————————————————
    df = pd.DataFrame({
        'step': range(len(loss_history)),
        'loss': loss_history,
        'accuracy': accuracy_history,
        'f1': f1_history,
    })
    csv_path = f"{out_dir}/metrics_{ALGO_NAME}.csv"
    df.to_csv(csv_path, index=False)

    # ——————————————————————————————————————————————————
    # 2b. 保存每个 feature 的细分 metrics 到 CSV
    # ——————————————————————————————————————————————————
    rows = []
    for f in synthesised_features:
        n = len(feature_loss_history[f.name])
        for i in range(n):
            rows.append({
                'feature': f.name,
                'step': i,
                'loss': feature_loss_history[f.name][i],
                'accuracy': feature_accuracy_history[f.name][i],
                'f1': feature_f1_history[f.name][i],
            })
    df_feat = pd.DataFrame(rows)
    feat_csv_path = f"{out_dir}/metrics_{ALGO_NAME}_per_feature.csv"
    df_feat.to_csv(feat_csv_path, index=False)
    print(f">>> Per-feature CSV 已保存: {feat_csv_path}")


    print(f">>> CSV 已保存: {csv_path}")

    # ——————————————————————————————————————————————————
    # 3. 保存模型参数
    # ——————————————————————————————————————————————————
    import torch

    ckpt_path = f"{out_dir}/model_{ALGO_NAME}.pt"
    torch.save({
        'algo': ALGO_NAME,
        'features': [f.name for f in synthesised_features],
        'state_dict': model.state_dict(),
        'hidden_dim': HIDDEN_DIM,
    }, ckpt_path)
    print(f">>> 模型参数已保存: {ckpt_path}")

    # ——————————————————————————————————————————————————
    # 4. 画图并保存
    # ——————————————————————————————————————————————————
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    ax1.plot(df['loss'], color='blue')
    ax1.set_title(f'Loss — {ALGO_NAME}')
    ax1.set_xlabel('Batch Step');
    ax1.set_ylabel('Loss');
    ax1.grid(True)

    ax2.plot(df['accuracy'], color='orange')
    ax2.set_title(f'Accuracy — {ALGO_NAME}')
    ax2.set_xlabel('Batch Step');
    ax2.set_ylabel('Accuracy');
    ax2.grid(True)

    ax3.plot(df['f1'], color='green')
    ax3.set_title(f'F1 Score — {ALGO_NAME}')
    ax3.set_xlabel('Batch Step');
    ax3.set_ylabel('F1');
    ax3.grid(True)

    plt.tight_layout()
    fig_path  = f"{out_dir}/fig_{ALGO_NAME}.pdf"
    plt.savefig(fig_path)
    print(f">>> 图已保存: {fig_path}")

    plt.close()

    # ——————————————————————————————————————————————————
    # 5. 每个 feature 单独一张图
    # ——————————————————————————————————————————————————
    for f in synthesised_features:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"{ALGO_NAME} — {f.name}", fontsize=13)

        ax1.plot(feature_loss_history[f.name], color='blue');
        ax1.set_title('Loss');
        ax1.set_xlabel('Batch Step');
        ax1.grid(True)
        ax2.plot(feature_accuracy_history[f.name], color='orange');
        ax2.set_title('Accuracy');
        ax2.set_xlabel('Batch Step');
        ax2.grid(True)
        ax3.plot(feature_f1_history[f.name], color='green');
        ax3.set_title('F1 Score');
        ax3.set_xlabel('Batch Step');
        ax3.grid(True)

        plt.tight_layout()
        feat_fig_path = f"{out_dir}/fig_{ALGO_NAME}_{f.name}.pdf"
        plt.savefig(feat_fig_path)
        plt.close()
        print(f">>> Feature 图已保存: {feat_fig_path}")

