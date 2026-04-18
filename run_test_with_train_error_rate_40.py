import os
import re
import sys
import argparse
import glob

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

sys.path.append(os.path.join(os.path.dirname(__file__), "dataset"))
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))

from feature import *
import coders as coders_NUM
import coders_LKP as coders_LKP
from factbase import FactBase, Constant
from nutils import *
from semantics import FactBaseSemanticsDataset
from bgp_semantics import BgpSemantics
from inject_anomaly import *
from torch_geometric.nn import GATv2Conv, GATConv, GCNConv
from torch_geometric.nn.conv.message_passing import MessagePassing

# ── 常量（与训练脚本保持一致） ────────────────────────────────────────────────
NUM_EDGE_TYPES          = 4
NUM_ITERATION_MESSAGE   = 3
NUM_TRANSFORMER_ENCODER = 2
NUM_TRANSFORMER_DECODER = 2
NUM_LEN_DATASET_EVAL    = 100   # test set 大小
NUM_LEN_DATASET         = 1024 * 4


# ══════════════════════════════════════════════════════════════════════════════
#  Model 定义（三个算法各自的 class，唯一区别是 conv 层）
# ══════════════════════════════════════════════════════════════════════════════

def _make_edge_type_transformer_layer(hidden_dim, num_edge_types, conv_cls):
    """工厂函数：生成一个 EdgeTypeTransformerLayer，conv 层类型由 conv_cls 决定。"""

    class EdgeTypeTransformerLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.layers = []
            for j in range(num_edge_types):
                l = conv_cls(hidden_dim, hidden_dim, 8, False)
                self.layers.append(l)
                self.add_module(f"layer_edge_type_{j}", l)
            self.drop  = torch.nn.Dropout(0.2)
            self.norm1 = torch.nn.BatchNorm1d(hidden_dim)
            self.drop1 = torch.nn.Dropout(0.2)
            self.norm2 = torch.nn.BatchNorm1d(hidden_dim)
            self.linear1 = torch.nn.Linear(hidden_dim, 4 * hidden_dim)
            self.linear2 = torch.nn.Linear(4 * hidden_dim, hidden_dim)

        def forward(self, x, edge_index, edge_type):
            x2 = torch.stack(
                [l(x.view(-1, self.hidden_dim), edge_index[:, edge_type == t])
                 for t, l in enumerate(self.layers)],
                dim=0
            ).sum(dim=0)
            x = x + self.drop1(x2)
            x = self.norm1(x.view(-1, self.hidden_dim)).view(x.shape)
            x = self.linear2(self.drop(torch.relu(self.linear1(x))))
            x = self.norm2(x.view(-1, self.hidden_dim)).view(x.shape)
            return x

    return EdgeTypeTransformerLayer()


def _make_graph_transformer(hidden_dim, num_layers, num_edge_types, conv_cls):

    class EdgeTypeGraphTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.propagation_layers = []
            for i in range(num_layers):
                l = _make_edge_type_transformer_layer(hidden_dim, num_edge_types, conv_cls)
                self.propagation_layers.append(l)
                self.add_module(f"prop_layer_{i}", l)

        def forward(self, x, edge_index, edge_type):
            for l in self.propagation_layers:
                x = torch.relu(l(x, edge_index, edge_type))
            return x

    return EdgeTypeGraphTransformer()


class PredicateGraphEmbedding(torch.nn.Module):
    def __init__(self, features, hidden_dim, num_edge_types,
                 excluded_feature_indices=set(), coders_mod=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        self.encoder = coders_mod.NodeFeatureEmbedding(
            hidden_dim, features,
            excluded_feature_indices=excluded_feature_indices)
        self.decoder = coders_mod.NodeFeatureDecoder(
            hidden_dim, features,
            excluded_feature_indices=excluded_feature_indices)

    def forward(self, x, mask, edge_index, edge_type,
                reliable_masking, x_clean=None):
        return self.encoder.forward(x, mask, reliable_masking,
                                    positional_encoding=False, x_clean=x_clean)


def build_model(features, hidden_dim, num_edge_types,
                excluded_feature_indices, conv_cls, coders_mod):
    """根据 conv_cls 和 coders_mod 构建对应算法的 Model。"""

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_edge_types = num_edge_types
            self.embedding = PredicateGraphEmbedding(
                features, hidden_dim, num_edge_types, excluded_feature_indices,
                coders_mod=coders_mod)
            self.decoder = self.embedding.decoder
            self.transformer_encoder = _make_graph_transformer(
                hidden_dim, NUM_TRANSFORMER_ENCODER, num_edge_types, conv_cls)
            self.transformer_decoder = _make_graph_transformer(
                hidden_dim, NUM_TRANSFORMER_DECODER, num_edge_types, conv_cls)
            self.num_iterations = NUM_ITERATION_MESSAGE

        def add_noise(self, x):
            noise = torch.randn(x.shape, device=x.device)
            noise_mask = torch.zeros([self.hidden_dim], device=x.device)
            noise_mask[:self.hidden_dim // 2] = 1
            return x + noise_mask.unsqueeze(0) * noise

        def forward(self, x, mask, edge_index, edge_type,
                    reliable_masking, x_clean=None):
            if edge_index.dim() == 3:
                edge_index = edge_index[0]
                edge_type  = edge_type[0]
            assert x.size(1) == 1
            x = self.embedding.forward(
                x, mask, edge_index, edge_type,
                reliable_masking, x_clean=x_clean)[:, 0]
            x = self.transformer_encoder.forward(x, edge_index, edge_type)
            x = self.add_noise(x)
            for _ in range(self.num_iterations):
                x = self.transformer_decoder.forward(x, edge_index, edge_type) + x
            return x

    return Model()


# ── 算法配置 ──────────────────────────────────────────────────────────────────
ALGO_CONFIGS = {
    "GATv2-NUM": {"folder": "GATv2_NUM", "conv_cls": GATv2Conv, "coders_mod": coders_NUM},
    "GATv2-LKP": {"folder": "GATv2_LKP", "conv_cls": GATv2Conv, "coders_mod": coders_LKP},
    "GAT-LKP":   {"folder": "GAT_LKP",   "conv_cls": GATConv,   "coders_mod": coders_LKP},  # ← 新增
    "GAT-NUM":   {"folder": "GAT_NUM",   "conv_cls": GATConv,   "coders_mod": coders_NUM},
    "GCN-NUM":   {"folder": "GCN_NUM",   "conv_cls": GCNConv,   "coders_mod": coders_NUM},
}

# ── 三个 test dataset 的配置 ───────────────────────────────────────────────────
DATASET_CONFIGS = [
    {"name": "bgp-ospf-dataset-test-origin", "label": "Baseline"},
    {"name": "bgp-ospf-dataset-test-larger", "label": "Larger-Scale"},
    {"name": "bgp-ospf-dataset-test-real",   "label": "Real-World"},
]

# 论文里更好看的 feature 名字
FEATURE_LABELS = {
    "predicate_bgp_route_arg2": "BGP Local Pref",
    "predicate_bgp_route_arg3": "BGP AS-Path",
    "predicate_bgp_route_arg5": "BGP MED",
    "predicate_connected_arg2": "OSPF Weight",
}

ALGO_COLORS = {
    "GATv2-NUM": "#E07B54",   # 橙红，对应论文里 Ours 的颜色
    "GATv2-LKP": "#5B8DB8",   # 蓝
    "GAT-LKP":   "#E67E22",   # 橙色  ← 新增
    "GAT-NUM":   "#4C9E6B",   # 绿
    "GCN-NUM":   "#8E44AD",   # 紫
}


# ══════════════════════════════════════════════════════════════════════════════
#  辅助函数（与训练脚本保持一致）
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  单次 run 评估
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_one_run(ckpt_path, conv_cls, coders_mod, test_dataset, synthesised_features,
                     features_all, hidden_dim, device, error_rate, seed):
    """加载一个 .pt，在 test set 上跑 inference，返回每个 feature 的 acc 和 f1。"""

    # 固定 seed，保证三个模型的 inject anomaly 完全一致
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(features_all, hidden_dim, NUM_EDGE_TYPES,
                        set(), conv_cls, coders_mod).to(device)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    # 兼容两种存储格式
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)

    loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    acc_sums  = {f.name: 0.0 for f in synthesised_features}
    f1_sums   = {f.name: 0.0 for f in synthesised_features}
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        batch.x = batch.x.unsqueeze(1)
        batch.edge_index = reflexive(bidirectional(batch.edge_index),
                                     num_nodes=batch.x.size(0))
        batch.edge_type = reflexive_bidirectional_edge_type(
            batch.edge_type, batch.x.size(0))

        corrupted_x, target_binary = inject_anomaly(
            batch.x, synthesised_features, error_rate=error_rate)

        fake_mask     = torch.zeros_like(corrupted_x, dtype=torch.bool)
        noisy_x_clean = make_noisy_reference(batch.x, noise_rate=0.02)

        x_emb = model.forward(
            corrupted_x, fake_mask,
            batch.edge_index, batch.edge_type,
            False,
            x_clean=noisy_x_clean,
        )

        for f in synthesised_features:
            acc = model.decoder.accuracy(x_emb, target_binary, f.name)
            f1  = model.decoder.decoders[f.name].f1(
                x_emb, target_binary[:, :, f.idx])
            acc_sums[f.name] += acc.item()
            f1_sums[f.name]  += f1.item()

        n_batches += 1

    return {
        f.name: {
            "acc": acc_sums[f.name] / n_batches,
            "f1":  f1_sums[f.name]  / n_batches,
        }
        for f in synthesised_features
    }


# ══════════════════════════════════════════════════════════════════════════════
#  扫描所有 run，汇总 mean ± std
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_algo(algo_name, folder, conv_cls, coders_mod, results_dir,
                  test_dataset, synthesised_features, features_all,
                  hidden_dim, device, error_rate, seed):

    pattern = os.path.join(results_dir, folder,
                           f"{folder}_*", f"model_{folder}.pt")
    ckpt_paths = sorted(glob.glob(pattern))

    if len(ckpt_paths) == 0:
        print(f"  [!] 没有找到 {algo_name} 的 checkpoint，跳过。\n      路径模式: {pattern}")
        return None

    print(f"\n{'='*60}")
    print(f"  {algo_name}: 找到 {len(ckpt_paths)} 个 run")
    print(f"{'='*60}")

    all_runs = []
    for ckpt_path in ckpt_paths:
        print(f"  >> 评估: {ckpt_path}")
        run_result = evaluate_one_run(
            ckpt_path, conv_cls, coders_mod, test_dataset, synthesised_features,
            features_all, hidden_dim, device, error_rate, seed)
        all_runs.append(run_result)
        for f_name, vals in run_result.items():
            label = FEATURE_LABELS.get(f_name, f_name)
            print(f"     {label:<20}  acc={vals['acc']*100:.2f}%  f1={vals['f1']*100:.2f}%")

    # mean ± std
    summary = {}
    for f in synthesised_features:
        accs = [r[f.name]["acc"] for r in all_runs]
        f1s  = [r[f.name]["f1"]  for r in all_runs]
        summary[f.name] = {
            "acc_mean": np.mean(accs), "acc_std": np.std(accs),
            "f1_mean":  np.mean(f1s),  "f1_std":  np.std(f1s),
        }

    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  生成 Table（CSV）和 Bar Chart
# ══════════════════════════════════════════════════════════════════════════════

def save_table(all_summaries, synthesised_features, out_dir):
    csv_path = os.path.join(out_dir, "test_results_table.csv")

    # 读已有 CSV 作为基础，保留之前跑过的算法列
    if os.path.exists(csv_path):
        df_base = pd.read_csv(csv_path).set_index("Feature")
    else:
        df_base = pd.DataFrame()

    # 只写这次 summary 不为 None 的算法（跳过的算法保持 CSV 原值）
    feature_labels = [FEATURE_LABELS.get(f.name, f.name) for f in synthesised_features]
    index = feature_labels + ["Overall"]
    df_new = pd.DataFrame(index=index)
    df_new.index.name = "Feature"

    for algo_name, summary in all_summaries.items():
        if summary is None:
            continue  # 跳过，CSV 里原有的值自动保留
        acc_col, f1_col = f"{algo_name} Acc", f"{algo_name} F1"
        for f, label in zip(synthesised_features, feature_labels):
            s = summary[f.name]
            df_new.loc[label, acc_col] = f"{s['acc_mean']*100:.1f}±{s['acc_std']*100:.1f}"
            df_new.loc[label, f1_col]  = f"{s['f1_mean']*100:.1f}±{s['f1_std']*100:.1f}"
        # Overall 行
        acc_m = np.mean([summary[f.name]["acc_mean"] for f in synthesised_features])
        acc_s = np.mean([summary[f.name]["acc_std"]  for f in synthesised_features])
        f1_m  = np.mean([summary[f.name]["f1_mean"]  for f in synthesised_features])
        f1_s  = np.mean([summary[f.name]["f1_std"]   for f in synthesised_features])
        df_new.loc["Overall", acc_col] = f"{acc_m*100:.1f}±{acc_s*100:.1f}"
        df_new.loc["Overall", f1_col]  = f"{f1_m*100:.1f}±{f1_s*100:.1f}"

    # merge：旧列保留，新列覆盖（新数据优先）
    if not df_base.empty:
        df_merged = df_base.copy()
        for col in df_new.columns:
            df_merged[col] = df_new[col]   # 新算法列直接写入/覆盖
    else:
        df_merged = df_new

    df_merged = df_merged.reset_index()
    df_merged.to_csv(csv_path, index=False)
    print(f"\n>>> Table 已更新: {csv_path}")
    print(df_merged.to_string(index=False))
    return df_merged


def save_bar_chart(synthesised_features, out_dir, metric="f1"):
    csv_path = os.path.join(out_dir, "test_results_table.csv")
    if not os.path.exists(csv_path):
        print(f"  [!] 没有找到 CSV，跳过画图: {csv_path}")
        return

    df = pd.read_csv(csv_path).set_index("Feature")
    feature_labels = [FEATURE_LABELS.get(f.name, f.name) for f in synthesised_features]

    # 从列名里找出所有算法
    algo_names = [col[:-4] for col in df.columns if col.endswith(" Acc")]

    n_features = len(feature_labels)
    n_algos    = len(algo_names)
    x          = np.arange(n_features)
    bar_width  = 0.18   # ← 从 0.22 缩小，为第5根柱留出空间
    offsets    = np.linspace(-(n_algos - 1) / 2,
                              (n_algos - 1) / 2, n_algos) * bar_width

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, algo_name in enumerate(algo_names):
        col = f"{algo_name} {'Acc' if metric == 'acc' else 'F1'}"
        if col not in df.columns:
            continue
        # CSV 里存的格式是 "85.3±2.1"，拆成 mean 和 std
        means, stds = [], []
        for label in feature_labels:
            val = df.loc[label, col] if label in df.index else None
            if val and val != "N/A":
                parts = str(val).split("±")
                means.append(float(parts[0]))
                stds.append(float(parts[1]) if len(parts) > 1 else 0.0)
            else:
                means.append(0.0)
                stds.append(0.0)

        ax.bar(x + offsets[i], means, bar_width,
               yerr=stds, capsize=4,
               label=algo_name,
               color=ALGO_COLORS.get(algo_name),
               alpha=0.85,
               error_kw={"elinewidth": 1.2, "ecolor": "black"})

    metric_label = "F1 Score (%)" if metric == "f1" else "Accuracy (%)"
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"test_results_{metric}.pdf")
    plt.savefig(fig_path)
    plt.close()
    print(f">>> Bar chart ({metric_label}) 已保存: {fig_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认从 inject_anomaly.py 读 ERROR_RATE，命令行可覆盖
    parser.add_argument("--error-rate",   type=float, default=ERROR_RATE,
                        help=f"默认读 inject_anomaly.ERROR_RATE (当前={ERROR_RATE})")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--hidden-dim",   type=int,   default=128)
    parser.add_argument("--results-dir",  type=str,   default="./results_train_40_error_rate")
    parser.add_argument("--out-dir",      type=str,   default=None,
                        help="不指定时会根据 --error-rate 和 --results-dir 自动生成，"
                             "例如 ./results_test_20_by_train_40")
    parser.add_argument("--force", nargs="*", default=[],
                        help="强制重新评估指定算法，例如: --force GATv2-NUM GAT-NUM")
    args = parser.parse_args()

    # ── 自动推断输出目录 ──────────────────────────────────────────────────────
    # 测试 error rate: 0.2 -> "20", 0.4 -> "40"
    # 训练 error rate: 从 --results-dir 里抠出数字，例如
    #   ./results_train_40_error_rate -> "40"
    if args.out_dir is None:
        test_tag = f"{int(round(args.error_rate * 100)):02d}"
        m = re.search(r"(\d+)", os.path.basename(args.results_dir.rstrip("/")))
        train_tag = m.group(1) if m else "unknown"
        args.out_dir = f"./results_test_{test_tag}_by_train_{train_tag}"
        print(f"[auto] --out-dir 未指定，自动使用: {args.out_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    # device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Running on {device}")

    # feature 注册（只需做一次，三个 dataset 共用）
    sem          = BgpSemantics()
    prog         = FactBase(sem.decls())
    feature      = prog.feature_registry.feature
    features_all = prog.feature_registry.get_all_features()

    synthesised_features = [
        feature("predicate_bgp_route_arg2"),
        feature("predicate_bgp_route_arg3"),
        feature("predicate_bgp_route_arg5"),
        feature("predicate_connected_arg2"),
    ]

    # 三个 test dataset 配置
    DATASET_CONFIGS = [
        {"name": "bgp-ospf-dataset-test-origin", "label": "Baseline"},
        {"name": "bgp-ospf-dataset-test-larger", "label": "Larger-Scale"},
        {"name": "bgp-ospf-dataset-test-real",   "label": "Real-World"},
    ]

    # ── 外层循环：逐 dataset 评估 ──────────────────────────────────────────────
    for ds_cfg in DATASET_CONFIGS:
        ds_name  = ds_cfg["name"]
        ds_label = ds_cfg["label"]

        print(f"\n{'#'*60}")
        print(f"  Dataset: {ds_label}  ({ds_name})")
        print(f"{'#'*60}")

        dataset      = FactBaseSemanticsDataset(sem, ds_name,
                                                num_samples=NUM_LEN_DATASET_EVAL)
        test_dataset = dataset[:NUM_LEN_DATASET_EVAL]
        print(f"  Test set size: {len(test_dataset)}")

        # 这个 dataset 的输出子目录
        ds_out_dir = os.path.join(args.out_dir, ds_label)
        os.makedirs(ds_out_dir, exist_ok=True)

        # 读已有 CSV，找出已跑过的算法，直接跳过
        existing_csv = os.path.join(ds_out_dir, "test_results_table.csv")
        already_done = set()
        if os.path.exists(existing_csv):
            df_existing = pd.read_csv(existing_csv)
            for col in df_existing.columns:
                if col.endswith(" Acc"):
                    already_done.add(col[:-4])
            if already_done:
                print(f"  [skip] CSV 中已有结果，跳过: {already_done}")

        # 逐算法评估
        all_summaries = {}
        for algo_name, cfg in ALGO_CONFIGS.items():
            # if algo_name in already_done:
            if algo_name in already_done and algo_name not in ALGO_CONFIGS:
                print(f"  [skip] {algo_name} 已有结果，跳过。")
                all_summaries[algo_name] = None  # None = 保留 CSV 原值
                continue

            summary = evaluate_algo(
                algo_name            = algo_name,
                folder               = cfg["folder"],
                conv_cls             = cfg["conv_cls"],
                coders_mod           = cfg["coders_mod"],
                results_dir          = args.results_dir,
                test_dataset         = test_dataset,
                synthesised_features = synthesised_features,
                features_all         = features_all,
                hidden_dim           = args.hidden_dim,
                device               = device,
                error_rate           = args.error_rate,
                seed                 = args.seed,
            )
            all_summaries[algo_name] = summary

            # 每跑完一个立刻更新 CSV，中途崩了也不丢数据
            save_table(all_summaries, synthesised_features, ds_out_dir)

        # 输出这个 dataset 的结果（确保最终状态写入）
        save_table(all_summaries, synthesised_features, ds_out_dir)
        save_bar_chart(synthesised_features, ds_out_dir, metric="f1")
        save_bar_chart(synthesised_features, ds_out_dir, metric="acc")

    print(f"\n全部完成！输出文件在: {args.out_dir}")