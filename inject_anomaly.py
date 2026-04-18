import torch

ERROR_RATE = 0.2
NUM_TRAINING_EPOCH = 250

FEATURE_INJECTION = {
    # --- relational / structural: swap ---
    # "predicate":                "swap",
    # "holds":                    "swap",
    # # --- bgp_route vocab ---
    "predicate_bgp_route_arg7": "swap",  # SPEAKER_ID
    "predicate"               : "vocab",
    "holds"                   : "vocab",
    # --- bgp_route vocab ---
    "predicate_bgp_route_arg7": "vocab",  # SPEAKER_ID
    #
    "predicate_bgp_route_arg2": "vocab",  # LP
    "predicate_bgp_route_arg3": "vocab",  # AS-path length
    "predicate_bgp_route_arg4": "vocab",  # ORIGIN_TYPE {0,1,2}
    "predicate_bgp_route_arg5": "vocab",  # MED
    "predicate_bgp_route_arg6": "vocab",  # IS_EBGP     {0,1}
    # --- OSPF: bias ---
    "predicate_connected_arg2": "vocab",  # OSPF weight
}

FEATURE_VOCAB = {
    "predicate"               : [0, 1, 2, 4, 5, 7, 10],
    "holds"                   : [0, 1],
    # "predicate_bgp_route_arg7": ,
    #
    "predicate_bgp_route_arg2": list(range(0, 10)), # local preference
    "predicate_bgp_route_arg3": list(range(1, 10)), # AS-path length    # [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "predicate_bgp_route_arg4": [0, 1, 2],          # Lookup encoder
    "predicate_bgp_route_arg5": list(range(0, 30)), # MED
    "predicate_bgp_route_arg6": [0, 1],             # Lookup encoder
    #
    "predicate_connected_arg2": list(range(1, 32)),
}


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

