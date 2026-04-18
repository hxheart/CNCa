import sys
from typing import List, Set

sys.path.append("../")

import torch
import math
from nutils import *


# ###############################################################

# class NumericalEncoder(torch.nn.Module):
#     def __init__(self, feature, hidden_dim):
#         super().__init__()
#         # 现在输入变成 3 维：原值, 对数值, 平方值
#         self.proj = torch.nn.Linear(2, hidden_dim)

#     # 修改 NumericalEncoder 的 forward
#     def forward(self, x):
#         x_f = x.float()
#         # 针对 0-31，我们不需要除以 1000，直接除以 32.0 甚至不除
#         # 加上一个强力的缩放系数，比如 10
#         f1 = (x_f / 32.0) * 10.0

#         # 甚至可以加入一个平方项，放大高权重的惩罚感
#         f2 = torch.pow(x_f / 32.0, 2) * 0.1

#         combined = torch.stack([f1, f2], dim=-1)
#         return self.proj(combined)

# ###############################################################

def decoder(num_values):  # 它不是一个普通的函数，它是用来“生产” PyTorch 类的函数

    # 强制改成二分类
    num_values = 2

    class Decoder(torch.nn.Module):
        def __init__(self, hidden_dim, sliced):
            super().__init__()
            self.num_values = num_values
            self.sliced = sliced

            self.decoder_net = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, self.num_values),
            )

            # 新的decoder：输入256维（自身128 + 邻居128）
            self.decoder_net_with_context = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 2, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(hidden_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, self.num_values),
            )

        def forward(self, x):
            x = x
            x = self.decoder_net(x)  # 通过三层 MLP
            x = x.log_softmax(dim=-1)  # 转成 log 概率

            if self.sliced:
                return x.sum(axis=1)
            else:
                return x

        def forward_with_neighbors(self, x_with_context):
            """
            x_with_context: [num_nodes, hidden_dim * 2]
            """
            x = self.decoder_net_with_context(x_with_context)
            x = x.log_softmax(dim=-1)

            if self.sliced:
                return x.sum(axis=1)
            else:
                return x

        def decode(self, x, return_entropies=False):
            x = self.forward(x)
            res = x.argmax(axis=-1)  # 这里是得到最终 configuration 的值

            if return_entropies:
                return res, torch.distributions.Categorical(logits=x).entropy()

            return res

        def loss(self, x, target):  # 传入这里的，只有15维特征当中的1维，这是在mainloop里面实现的。
            x = self.forward(x)  # 得到 log_softmax 后的概率分布
            config = (target != -1).view(-1)  # 过滤无效特征（只关心不是 -1 的部分）
            x_configed = x.view(-1, self.num_values)[config, :]  # 只取有意义的预测
            target_configed = target.view(-1)[config]  # 只取有意义的标签
            # # ———————— equal loss ————————
            return torch.nn.functional.nll_loss(x_configed, target_configed)  # 正确和错误样本一样重要

        # def loss(self, x, target):
        #     x = self.forward(x)
        #     mask = (target != -1).view(-1)
        #     x_masked = x.view(-1,self.num_values)[mask,:]
        #     target_masked = target.view(-1)[mask]
        #     return torch.nn.functional.nll_loss(x_masked, target_masked)

        def accuracy(self, x, target):
            x = self.forward(x).argmax(axis=-1)  # 得到的是整数。因为是二分类，所以是0或者1;
            config = (target != -1).view(-1)  #
            x_configed = x.view(-1)[config]
            target_configed = target.view(-1)[config]
            return (x_configed == target_configed).sum().float() / config.sum()

        # Precision (精确率)：在你指出有错的地方里，有多少是真的错了？（防止乱报）。
        # Recall (召回率/抓取率)：在所有真的改错的地方里，你抓住了多少个？（防止漏报）。
        # F1 Score 是 Precision 和 Recall 的“调和平均数”。
        # 如果模型全猜 0（正常），它的 Recall 是 0，F1 Score 也会直接变成 0。
        # 只有当模型既能抓准、又不漏抓时，F1 才会高。这才是评价“找茬”模型真正的指标。
        def f1(self, x, target):
            x = self.forward(x).argmax(axis=-1)
            config = (target != -1).view(-1)
            x_configed = x.view(-1)[config]
            target_configed = target.view(-1)[config]
            return f1_loss(target_configed, x_configed)

    return Decoder


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1


def boolean_encoding():
    return onehot_encoding(2)


def binary_encoding(num_values):
    class BinaryEncoding(torch.nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()

            bitwidth = torch.ceil(torch.log2(torch.tensor(num_values).float())).long().item()
            self.embedding = torch.nn.Linear(bitwidth, hidden_dim, bias=False)

            mask = 2 ** torch.arange(bitwidth)
            self.register_buffer('mask', mask)

        def forward(self, x):
            mask = self.mask.clone().detach().requires_grad_(False)
            x = x.unsqueeze(-1).bitwise_and(mask).ne(0).float()
            x = self.embedding(x)
            return x

    return BinaryEncoding, decoder(num_values)


def onehot_encoding(num_values):
    class OneHotEncoding(torch.nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.embedding = torch.nn.Embedding(num_values + 1, hidden_dim)

        def forward(self, x):
            return self.embedding.forward(x.to(torch.long))

    return OneHotEncoding, decoder(num_values)


class MaskedFeatureEmbedding(torch.nn.Module):
    def __init__(self, embeddings: List[torch.nn.Module], hidden_dim, excluded_feature_indices: Set[int]):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embeddings = embeddings
        for i, e in enumerate(self.embeddings):
            self.add_module("embedding_" + str(i), e)
        self.num_embeddings = len(self.embeddings)
        self.excluded_feature_indices = list(excluded_feature_indices)
        self.mask_embedding = torch.nn.Parameter(torch.randn([len(self.embeddings), hidden_dim]))

    def forward(self, x, mask, reliable_masking):
        assert x.size(
            2) == self.num_embeddings, f"cannot embed input with shape {x.shape} using {self.num_embeddings} embeddings"

        x_emb = torch.zeros([x.size(0), x.size(1), self.hidden_dim], device=x.device)

        modules = dict(self.named_modules())

        for i in range(self.num_embeddings):
            embedding = modules['embedding_' + str(i)]
            if i in self.excluded_feature_indices: continue

            mask_emb = self.mask_embedding[i].unsqueeze(0).unsqueeze(0)
            feature_emb = embedding.forward(torch.maximum(torch.tensor(0), x[:, :, i] + 1))
            random_emb = feature_emb[torch.randint(0, feature_emb.size(0), device=x.device, size=[x.size(0)])]

            if not reliable_masking:
                masked_emb = categorical([feature_emb, random_emb, mask_emb], [0.1, 0.1, 0.8])
            else:
                masked_emb = mask_emb
            non_masked_emb = feature_emb

            feature_is_applicable = (x[:, :, i] != -1)
            feature_is_queried = (x[:, :, i] == -2)
            is_masked = mask[:, :, i].logical_or(feature_is_queried).unsqueeze(-1)

            x_emb = x_emb + feature_is_applicable.unsqueeze(-1) * (
                        is_masked * masked_emb + is_masked.logical_not() * non_masked_emb)
        return x_emb


def mask_like(x, p=0.5):
    return ((torch.rand_like(x, dtype=torch.float) - (1 - p)) >= 0)


class PositionalEncoding(torch.nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=4 * 30000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pos = self.pe[0, :x.size(0)].clone().detach().requires_grad_(False).unsqueeze(1)
        x = x + pos
        return self.dropout(x)


# ========================================================================

class NodeFeatureEmbedding(torch.nn.Module):
    def __init__(self, hidden_dim, features, excluded_feature_indices=set()):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.positional_encoding = PositionalEncoding(hidden_dim, 0.0)

        # Baseline (LKP): 两路都使用 lookup encoder，不使用 NumericalEncoder
        # 对比对象: coders.py 使用 NumericalEncoder 处理数值型 feature（C-Ca 创新点）
        # 结构完全对称，唯一变量是 encoder 类型
        self.embedding = MaskedFeatureEmbedding(
            [f.encoder(hidden_dim) for f in features],
            self.hidden_dim,
            excluded_feature_indices
        )
        self.ref_embedding = MaskedFeatureEmbedding(
            [f.encoder(hidden_dim) for f in features],
            self.hidden_dim,
            excluded_feature_indices
        )

        # 融合层：结构与 coders.py 完全一致
        self.fusion = torch.nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, mask, reliable_masking, positional_encoding=True, x_clean=None):
        x_emb = self.embedding.forward(x, mask, reliable_masking)

        if x_clean is not None:
            clean_mask = torch.zeros_like(mask)
            ref_emb = self.ref_embedding.forward(x_clean, clean_mask, reliable_masking=False)
            x_emb = self.fusion(torch.cat([x_emb, ref_emb], dim=-1))

        return x_emb

# ========================================================================

# class NodeFeatureEmbedding(torch.nn.Module):
#     def __init__(self, hidden_dim, features, excluded_feature_indices=set()):
#         super().__init__()

#         self.hidden_dim = hidden_dim
#         self.positional_encoding = PositionalEncoding(hidden_dim, 0.0)

#         numerical_feature_names = {
#             "predicate_connected_arg2",
#             "predicate_bgp_route_arg2",
#             "predicate_bgp_route_arg3",
#             "predicate_bgp_route_arg4",
#             "predicate_bgp_route_arg5",
#             "predicate_bgp_route_arg6",
#             "predicate_bgp_route_arg7",
#         }

#         def build_encoders(features, prefix):
#             encoders = []
#             for f in features:
#                 if f.name in numerical_feature_names:
#                     # print(f"[{prefix}] NumericalEncoder for: {f.name}")
#                     encoders.append(NumericalEncoder(f, hidden_dim))
#                 else:
#                     encoders.append(f.encoder(hidden_dim))
#             return encoders

#         # corrupted_embedding: 处理注入了异常的 corrupted_x（模型的主输入）
#         # ref_embedding:       处理干净的 x_clean（正常状态参考信号）
#         #                      两套权重独立，让模型自己学会如何利用两者的差异
#         self.embedding = MaskedFeatureEmbedding(
#             build_encoders(features, "corrupted"),
#             self.hidden_dim,
#             excluded_feature_indices
#         )
#         self.ref_embedding = MaskedFeatureEmbedding(
#             build_encoders(features, "ref"),
#             self.hidden_dim,
#             excluded_feature_indices
#         )

#         # 融合层：把两路 hidden_dim concat 后 project 回 hidden_dim
#         # 用 Linear 而不是直接相加，让模型自己学融合方式
#         self.fusion = torch.nn.Linear(hidden_dim * 2, hidden_dim)

#     def forward(self, x, mask, reliable_masking, positional_encoding=True, x_clean=None):
#         # 主路：处理 corrupted_x
#         x_emb = self.embedding.forward(x, mask, reliable_masking)

#         if x_clean is not None:
#             # 参考路：处理干净的 x_clean，不需要 masking
#             clean_mask = torch.zeros_like(mask)
#             ref_emb = self.ref_embedding.forward(x_clean, clean_mask, reliable_masking=False)
#             # concat → Linear → 还原为 hidden_dim
#             x_emb = self.fusion(torch.cat([x_emb, ref_emb], dim=-1))

#         return x_emb


# ========================================================================


class NodeFeatureDecoder(
    torch.nn.Module):  # This is GNN's "readout" function; 这个类负责将 GNN 产生的隐藏特征向量（Hidden Dim）变回我们能看懂的配置数值（如 OSPF 权重）。
    def __init__(self, hidden_dim, features, excluded_feature_indices=set(), sliced=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.excluded_feature_indices = excluded_feature_indices
        self.features_by_name = {}
        self.decoders = {}  # 一个 self.decoders 字典。它为每个特征（比如 OSPF 权重）都生产了一个专属的小 MLP，并存在字典里，Key 就是特征的名字（如 predicate_connected_arg2）。

        for i, f in enumerate(features):
            self.features_by_name[f.name] = f
            self.decoders[f.name] = f.decoder(hidden_dim, sliced=sliced)  #
            self.add_module(f"decoder_{f.name}_{i}", self.decoders[f.name])

    def forward(self, x, feature_name):
        assert feature_name in self.features_by_name.keys(), f"unknown node feature {feature_name}, available {str(self.features_by_name.keys())}"
        feature: Feature = self.features_by_name[feature_name]
        decoder: Decoder = self.decoders[feature_name]

        return decoder.forward(x)

    def decode(self, x, feature_name, return_entropies=False):
        assert feature_name in self.features_by_name.keys(), f"unknown node feature {feature_name}, available {str(self.features_by_name.keys())}"
        feature: Feature = self.features_by_name[feature_name]
        decoder: Decoder = self.decoders[feature_name]

        return decoder.decode(x, return_entropies)

    def loss(self, x, target, feature_name):
        assert feature_name in self.features_by_name.keys(), f"unknown node feature {feature_name}, available {str(self.features_by_name.keys())}"
        feature: Feature = self.features_by_name[feature_name]  # 找到这个特征的配置信息（从字典里取出对应的 Feature 对象）
        decoder: Decoder = self.decoders[feature_name]  # 把init的decoders变成decoder；[找到这个特征专属的decoder(小MLP)]
        # 上面两行代码里面的冒号，就是增加一个注释。分别告诉，变量是 Feature 类型，是 Decoder 类型;

        assert feature.idx not in self.excluded_feature_indices, f"cannot apply loss for excluded feature {feature.name}"

        # 下面这个 loss 只计算，一个 synthesised_features 的 loss，从 target[:,:,feature.idx] 这里也能看出来，只取了一个
        loss = decoder.loss(x, target[:, :, feature.idx])  # 调用上面找到的专属decoder的loss方法
        if torch.isnan(loss):  # 检查 loss 值是不是 NaN (Not a Number)
            print("nan loss for", feature_name)
            return torch.tensor(0)
        return loss

    def accuracy(self, x, target, feature_name):
        assert feature_name in self.features_by_name.keys(), f"unknown node feature {feature_name}, available {str(self.features_by_name.keys())}"
        feature: Feature = self.features_by_name[feature_name]
        decoder: Decoder = self.decoders[feature_name]

        assert feature.idx not in self.excluded_feature_indices, f"cannot evaluate accuracy for excluded feature {feature.name}"
        return decoder.accuracy(x, target[:, :, feature.idx])