"""
A_plot_fig.py  —  所有图的绘制脚本

图的顺序：
  Fig 1 — fig_GATv2_NUM_per_feature          (GATv2_NUM 单算法，4 feature 同一子图)
  Fig 2 — fig_training_curves_comparison     (多算法训练曲线，4行×3列)
  Fig 3 — fig_test_different_network_scale   (test@40%, 暖色)
  Fig 4 — fig_test_different_network_scale   (test@20% by train@40%, 冷色)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from pathlib import Path


# ============================================================
#  全局配置
# ============================================================

MODELS = [
    ("GATv2_NUM", "CNCa (Ours)"),
    ("GAT_LKP",   "LKP-GAT [ref]"),      # ← 新增
    ("GATv2_LKP", "LKP-NCa"),
    ("GAT_NUM",   "CCa-GAT"),
    ("GCN_NUM",   "CCa-GCN"),
]

NUM_RUNS   = 3
METRICS      = ["f1",       "accuracy", "loss"]
METRIC_NAMES = ["F1 Score", "Accuracy", "Loss"]
SMOOTH_WIN   = 20          # rolling 窗口，设 1 关闭平滑

# ✅ 在这里统一配置，切换只改这一行
RESULTS_DIR_TEMPLATE = "results_train_20_error_rate/{algo_name}"
RESULTS_DIR_TEMPLATE = "results_train_40_error_rate/{algo_name}"

# feature 内部名 → 物理世界名称的映射
FEATURE_NAME_MAP = {
    "predicate_connected_arg2": "OSPF Weight",
    "predicate_bgp_route_arg5": "MED",
    "predicate_bgp_route_arg3": "AS-Path Length",
    "predicate_bgp_route_arg2": "Local Preference",
}

# Fig 1 per-feature 颜色（按 feature 区分，暖色系）
FEATURE_COLORS = ["#9B59B6", "#1D9E75", "#378ADD", "deeppink", ]  

# Fig 2 算法对比颜色（区别于 Fig 1，避免误解）
ALGO_COLORS     = ["orangered", "#8E44AD", "darkgoldenrod", "#27AE60", "#2C7BB6"]  # 橘色 / 暗紫 / 橙色 / 墨绿 / 钢蓝
ALGO_LINESTYLES = ["-", "--", (0, (3, 1, 1, 1)), "-.", ":"]                  # 实线 / 长虚 / 点划点 / 点划 / 点线

# 自动推断输出路径后缀
if "train_20_error_rate" in RESULTS_DIR_TEMPLATE:
    _SUFFIX = "20_error_rate"
elif "train_40_error_rate" in RESULTS_DIR_TEMPLATE:
    _SUFFIX = "40_error_rate"
else:
    _SUFFIX = "unknown"

OUTPUT_FIG1 = f"./figs/fig_GATv2_NUM_per_feature_{_SUFFIX}.pdf"
OUTPUT_FIG2 = f"./figs/fig_training_curves_comparison_{_SUFFIX}.pdf"

# ylim 配置（按 error rate 区分）
YLIM_MAP = {
    "20_error_rate": {"f1": (-0.02, 1.0), "accuracy": (0.80, 1.0), "loss": (0.03, 0.52)},
    "40_error_rate": {"f1": (-0.02, 1.0), "accuracy": (0.55, 1.0), "loss": (0.03, 0.70)},
}.get(_SUFFIX, {"f1": (0.0, 1.0), "accuracy": (0.0, 1.0), "loss": (0.0, 1.0)})


# ============================================================
#  共用工具函数
# ============================================================

def smooth(arr: np.ndarray, w: int) -> np.ndarray:
    """按列做 rolling mean，不产生边缘效应（min_periods=1）。"""
    if w <= 1:
        return arr
    return (pd.Series(arr)
              .rolling(w, min_periods=1, center=False)
              .mean()
              .values)


def apply_ax_style(ax):
    """统一子图样式：去掉上/右边框，加虚线网格。"""
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def apply_legend_style(leg):
    """统一图例边框样式。"""
    leg.get_frame().set_edgecolor("#888888")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.5)


def feature_display_name(feat: str) -> str:
    """将 CSV 内部 feature 名转换为物理世界名称。"""
    return FEATURE_NAME_MAP.get(feat, feat)


# ============================================================
#  加载训练曲线数据
# ============================================================

def load_per_feature(algo_name: str, num_runs: int):
    """返回 {feature: {metric: ndarray(runs, steps)}}"""
    results_dir = Path(RESULTS_DIR_TEMPLATE.format(algo_name=algo_name))
    dfs = []
    for run in range(1, num_runs + 1):
        csv_path = (results_dir / f"{algo_name}_{run}"
                    / f"metrics_{algo_name}_per_feature.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["run"] = run
            dfs.append(df)
        else:
            print(f"  [WARN] 找不到: {csv_path}")

    if not dfs:
        return {}

    combined = pd.concat(dfs, ignore_index=True)
    out = {}
    for feat in combined["feature"].unique():
        sub = combined[combined["feature"] == feat]
        min_steps = sub.groupby("run")["step"].count().min()
        feat_data = {}
        for metric in METRICS:
            runs_arr = [
                sub[sub["run"] == r][metric].values[:min_steps]
                for r in range(1, num_runs + 1)
                if len(sub[sub["run"] == r][metric].values[:min_steps]) == min_steps
            ]
            if runs_arr:
                feat_data[metric] = np.stack(runs_arr)  # (runs, steps)
        out[feat] = feat_data
    return out


# 收集所有算法的训练曲线数据
all_data: dict = {}          # {algo_name: {feature: {metric: ndarray}}}
all_features_ordered: list = []

for algo_name, _ in MODELS:
    print(f"加载 {algo_name} ...")
    d = load_per_feature(algo_name, NUM_RUNS)
    all_data[algo_name] = d
    for f in d:
        if f not in all_features_ordered:
            all_features_ordered.append(f)

n_features = len(all_features_ordered)
n_cols     = len(METRICS)


# ============================================================
#  Fig 1 — GATv2_NUM per-feature（4 个 feature 同一子图）
# ============================================================

TARGET_ALGO = "GATv2_NUM"

fig1, axes1 = plt.subplots(1, n_cols, figsize=(15, 2.5), squeeze=False)
fig1.subplots_adjust(wspace=0.32)

algo_data = all_data.get(TARGET_ALGO, {})

for col, (metric, metric_name) in enumerate(zip(METRICS, METRIC_NAMES)):
    ax = axes1[0][col]

    for feat_idx, feat in enumerate(all_features_ordered):
        feat_data = algo_data.get(feat)
        if feat_data is None or metric not in feat_data:
            continue

        color = FEATURE_COLORS[feat_idx % len(FEATURE_COLORS)]
        arr   = feat_data[metric]
        steps = np.arange(arr.shape[1])
        mean  = smooth(arr.mean(axis=0), SMOOTH_WIN)
        std   = smooth(arr.std(axis=0),  SMOOTH_WIN)

        ax.plot(steps, mean, color=color, linewidth=1.2,
                label=feature_display_name(feat))
        ax.fill_between(steps, mean - std, mean + std,
                        alpha=0.15, color=color, linewidth=0)

    ax.set_title(metric_name, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Training step", fontsize=9)
    if col == 0:
        ax.set_ylabel("Value", fontsize=9)
    ax.set_ylim(*YLIM_MAP[metric])
    apply_ax_style(ax)

legend_handles1 = [
    mlines.Line2D([], [], color=FEATURE_COLORS[i % len(FEATURE_COLORS)],
                  linewidth=1.5, label=feature_display_name(feat))
    for i, feat in enumerate(all_features_ordered)
]
leg1 = fig1.legend(handles=legend_handles1, loc="upper center",
                   ncol=len(all_features_ordered), fontsize=10,
                   frameon=True, bbox_to_anchor=(0.5, 1.15))
apply_legend_style(leg1)

fig1.savefig(OUTPUT_FIG1, bbox_inches="tight")
print(f">>> 已保存: {OUTPUT_FIG1}")


# ============================================================
#  Fig 2 — 多算法训练曲线对比（n_features 行 × 3 列）
# ============================================================

fig2, axes2 = plt.subplots(n_features, n_cols,
                            figsize=(5 * n_cols, 3 * n_features),
                            squeeze=False)
fig2.subplots_adjust(hspace=0.3, wspace=0.2)

for row, feat in enumerate(all_features_ordered):
    for col, (metric, metric_name) in enumerate(zip(METRICS, METRIC_NAMES)):
        ax = axes2[row][col]

        for (algo_name, algo_label), color, ls in zip(MODELS, ALGO_COLORS, ALGO_LINESTYLES):
            feat_data = all_data.get(algo_name, {}).get(feat)
            if feat_data is None or metric not in feat_data:
                continue

            arr   = feat_data[metric]
            steps = np.arange(arr.shape[1])
            mean  = smooth(arr.mean(axis=0), SMOOTH_WIN)
            std   = smooth(arr.std(axis=0),  SMOOTH_WIN)

            ax.plot(steps, mean, color=color, linestyle=ls, linewidth=1.2, label=algo_label)
            ax.fill_between(steps, mean - std, mean + std,
                            alpha=0.15, color=color, linewidth=0)

        if row == 0:
            ax.set_title(metric_name, fontsize=11, fontweight="bold", pad=6)
        if col == 0:
            ax.set_ylabel(feature_display_name(feat), fontsize=10, labelpad=8)

        ax.set_ylim(*YLIM_MAP[metric])
        ax.set_xlabel("Training step", fontsize=8)
        apply_ax_style(ax)

legend_handles2 = [
    mlines.Line2D([], [], color=c, linestyle=ls, linewidth=1.5, label=label)
    for (_, label), c, ls in zip(MODELS, ALGO_COLORS, ALGO_LINESTYLES)
]
leg2 = fig2.legend(handles=legend_handles2, loc="upper center",
                   ncol=len(MODELS), fontsize=12,
                   frameon=True, bbox_to_anchor=(0.5, 0.95))
apply_legend_style(leg2)

fig2.savefig(OUTPUT_FIG2, bbox_inches="tight")
print(f">>> 已保存: {OUTPUT_FIG2}")

# plt.show()


# ============================================================
#  Fig 3 & Fig 4 — Test results（不同网络规模，bar chart）
#  共用 parse_mean_std / load_csv / plot_test_results
# ============================================================

# ALGO_NAMES_TEST    = ["GATv2-NUM",          "GATv2-LKP", "GAT-NUM", "GCN-NUM"]  
# DISPLAY_NAMES_TEST = ["CNCa (Ours)",  "LKP-NCa",   "CCa-GAT", "CCa-GCN"]  
# ALGO_HATCHES_TEST  = ["", "//", "\\\\", ".."]    

ALGO_NAMES_TEST    = ["GATv2-NUM",   "GAT-LKP",       "GATv2-LKP", "GAT-NUM", "GCN-NUM"]  # ← 新增 GAT-LKP
DISPLAY_NAMES_TEST = ["CNCa (Ours)", "LKP-GAT [ref]", "LKP-NCa",   "CCa-GAT", "CCa-GCN"]  # ← 新增 GAT-LKP
ALGO_HATCHES_TEST  = ["", "//", "\\\\", "..", "xx"]                                    # ← 新增 "xx"


# CSV 里 Feature 列的原始名称（用于 key 查找）
FEATURES_TEST  = ["BGP Local Pref", "BGP AS-Path", "BGP MED", "OSPF Weight"]
# 对应的物理世界显示名称（用于 x 轴 label）
FEATURES_TEST_DISPLAY = ["Local Preference", "AS-Path Length", "MED", "OSPF Weight"]
METRICS_TEST   = ["F1", "Acc"]
COL_LABELS_TEST = ["F1 Score (%)", "Accuracy (%)"]


def parse_mean_std(s):
    if s == "N/A":
        return (np.nan, np.nan)
    parts = s.split("±")
    return float(parts[0]), float(parts[1])


def load_csv_test(csv_path, metric, algo_names, features):
    df = pd.read_csv(csv_path)
    df = df[df["Feature"] != "Overall"]
    result = {}
    for algo in algo_names:
        col = f"{algo} {metric}"
        result[algo] = {row["Feature"]: parse_mean_std(row[col])
                        for _, row in df.iterrows()}
    return result


def plot_test_results(dataset_configs, algo_colors, out_path):
    n_rows, n_cols   = len(dataset_configs), len(METRICS_TEST)
    n_features       = len(FEATURES_TEST)
    n_algos          = len(ALGO_NAMES_TEST)

    bar_width = 0.14  # ← 从 0.22 缩小，为第5根柱留出空间
    x       = np.arange(n_features)
    offsets = np.linspace(-(n_algos - 1) / 2,
                           (n_algos - 1) / 2, n_algos) * bar_width * 1

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7.16, 8.5),
                             sharey=True, sharex=False)

    for row_idx, ds_cfg in enumerate(dataset_configs):
        csv_path   = ds_cfg["csv"]
        data_cache = {}

        for col_idx, metric in enumerate(METRICS_TEST):
            ax = axes[row_idx][col_idx]

            if row_idx == 0:
                ax.set_title(COL_LABELS_TEST[col_idx], fontsize=14,
                             fontweight="bold", pad=6)
            if col_idx == 0:
                ax.set_ylabel(f"{ds_cfg['label']}\n\nScore (%)", fontsize=12)

            if not os.path.exists(csv_path):
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="gray")
                ax.set_xticks([])
                continue

            if metric not in data_cache:
                data_cache[metric] = load_csv_test(csv_path, metric,
                                                   ALGO_NAMES_TEST, FEATURES_TEST)
            data = data_cache[metric]

            for i, (algo, color, hatch) in enumerate(
                    zip(ALGO_NAMES_TEST, algo_colors, ALGO_HATCHES_TEST)):
                means = [data[algo][f][0] for f in FEATURES_TEST]
                stds  = [data[algo][f][1] for f in FEATURES_TEST]
                ax.bar(x + offsets[i], means, bar_width,
                       yerr=stds, capsize=3,
                       color=color, hatch=hatch, alpha=0.85,
                       label=algo,
                       error_kw={"elinewidth": 1.0, "ecolor": "black", "capthick": 1.0})

            ax.set_xticks(x)
            if row_idx == n_rows - 1:
                ax.set_xticklabels(FEATURES_TEST_DISPLAY, fontsize=12, rotation=20, ha="right")
            else:
                ax.set_xticklabels([""] * n_features)

            ax.set_ylim(0, 108)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # handles = [
    #     mpatches.Patch(facecolor=c, hatch=h, label=a, alpha=0.85)
    #     for a, c, h in zip(DISPLAY_NAMES_TEST, algo_colors, ALGO_HATCHES_TEST)
    # ]

    # handles_ordered = [handles[0], handles[1], handles[2],   # 第一行：CNCa, LKP-GAT, LKP-NCa
    #                handles[3], handles[4]] 

    # fig.legend(handles=handles_ordered, loc="upper center", ncol=3,  #ncol=n_algos,
    #            fontsize=12, frameon=True, framealpha=0.9,
    #            edgecolor="#cccccc", bbox_to_anchor=(0.5, 1.08))

    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(facecolor=c, hatch=h, label=a, alpha=0.85)
        for a, c, h in zip(DISPLAY_NAMES_TEST, algo_colors, ALGO_HATCHES_TEST)
    ]

    # 透明占位 handle（完全看不见）
    invisible = mpatches.Patch(facecolor='none', edgecolor='none', label=' ')

    # 第二行（3个）：带框，作为"外框"——先画，位置偏下
    leg2 = fig.legend(handles=[invisible]*6,
                      loc="upper center", ncol=3,
                      fontsize=12, frameon=True, framealpha=0.9,
                      edgecolor="#cccccc",
                      borderpad=1.,
                      labelspacing= -0.5,       # 行间距，决定框的高度
                      columnspacing=8.5,      # 列间距，决定框的宽度
                      handletextpad=0.8,
                      bbox_to_anchor=(0.5, 1.1))
    fig.add_artist(leg2)

    # 第一行（2个）：无框，叠在第二行框的上半部分
    leg1 = fig.legend(handles=handles[0:2],
                      loc="upper center", ncol=2,
                      fontsize=12, frameon=False,
                      bbox_to_anchor=(0.5, 1.1))

    leg3 = fig.legend(handles=handles[2:5], 
                      loc="upper center", ncol=3,
                      fontsize=12, frameon=False,
                      bbox_to_anchor=(0.5, 1.07))


    plt.tight_layout(rect=[0, 0, 1, 1])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f">>> 图已保存: {out_path}")


# ── Fig 3：test@40%，暖色调 ────────────────────────────────────────────────────
plot_test_results(
    dataset_configs=[
        {"label": "Baseline",     "csv": "results_test_40_by_train_40/Baseline/test_results_table.csv"},
        {"label": "Larger-Scale", "csv": "results_test_40_by_train_40/Larger-Scale/test_results_table.csv"},
        {"label": "Real-World",   "csv": "results_test_40_by_train_40/Real-World/test_results_table.csv"},
    ],
    algo_colors=["#F4A5A5", "#F7C59F", "#F5E0A0", "#A8D8A8", "#F0C8A0"],   # 暖色：玫瑰粉/蜜桃橙/淡黄/抹茶绿/杏色  ← 新增淡黄
    out_path="./figs/fig_test_different_network_scale_test_40_by_train_40.pdf",
)

# ── Fig 4：test@20% by train@40%，冷色调 ──────────────────────────────────────
plot_test_results(
    dataset_configs=[
        {"label": "Baseline",     "csv": "results_test_20_by_train_40/Baseline/test_results_table.csv"},
        {"label": "Larger-Scale", "csv": "results_test_20_by_train_40/Larger-Scale/test_results_table.csv"},
        {"label": "Real-World",   "csv": "results_test_20_by_train_40/Real-World/test_results_table.csv"},
    ],
    algo_colors=["#A8C8E8", "#B5A8D8", "#A0C8B8", "#A8D8D8", "#C4A8D8"],   # 冷色：天空蓝/薰衣草紫/薄荷绿/薄荷青/杏紫  ← 新增薄荷绿
    out_path="./figs/fig_test_different_network_scale_test_20_by_train_40.pdf",
)


# ============================================================
#  Training Efficiency 分析 —— CNCa vs LKP-GAT
#  以 Local Preference 为例，量化"收敛速度提升百分比"
# ============================================================

TARGET_FEATURE        = "predicate_connected_arg2"  # OSPF Weight
OURS_ALGO             = "GATv2_NUM"                 # CNCa (Ours)
BASELINE_ALGO         = "GAT_LKP"                   # LKP-GAT [ref]
FINAL_WINDOW          = 100    # 取最后 100 步均值作为 "最终收敛水平"
STABLE_WINDOW         = 20     # 连续 N 步都满足阈值，才算"稳定达到"
THRESHOLD_RATIOS      = [0.90, 0.95]  # 以 baseline 最终水平的 90% / 95% 作为阈值


def steps_to_threshold(curve, threshold, stable_window, mode="above"):
    """
    返回曲线第一次"稳定达到"阈值时的 step index。
    stable: 连续 stable_window 个点都满足条件。
    mode='above' : curve >= threshold  (用于 F1 / Accuracy)
    mode='below' : curve <= threshold  (用于 Loss)
    """
    if mode == "above":
        ok = curve >= threshold
    else:
        ok = curve <= threshold

    # 滑动窗口检查：从 i 到 i+stable_window 全部为 True
    n = len(curve)
    for i in range(n - stable_window + 1):
        if ok[i:i + stable_window].all():
            return i
    return None


def get_mean_curve(algo, feat, metric):
    """取某算法在某 feature / metric 下的多 run 平均曲线（已平滑）。"""
    feat_data = all_data.get(algo, {}).get(feat)
    if feat_data is None or metric not in feat_data:
        return None
    arr = feat_data[metric]  # (runs, steps)
    return smooth(arr.mean(axis=0), SMOOTH_WIN)


print("\n" + "=" * 70)
print("  Training Efficiency Analysis : CNCa (Ours) vs LKP-GAT [ref]")
print(f"  Feature : {feature_display_name(TARGET_FEATURE)}")
print("=" * 70)

eff_results = {}  # 存结果方便后面写句子

for metric, metric_name in zip(METRICS, METRIC_NAMES):
    curve_ours     = get_mean_curve(OURS_ALGO,     TARGET_FEATURE, metric)
    curve_baseline = get_mean_curve(BASELINE_ALGO, TARGET_FEATURE, metric)

    if curve_ours is None or curve_baseline is None:
        print(f"\n[{metric_name}] 数据缺失，跳过")
        continue

    # 对齐长度（两者 step 数可能不同）
    L = min(len(curve_ours), len(curve_baseline))
    curve_ours, curve_baseline = curve_ours[:L], curve_baseline[:L]

    # 决定 mode 和最终收敛水平
    if metric == "loss":
        mode = "below"
        final_level = curve_baseline[-FINAL_WINDOW:].mean()
        # loss 阈值 = baseline 最终 loss 的 1/ratio （越小越好，所以反过来）
        # 更自然的做法：以 baseline final loss 作为 1x，阈值设为 1.1x / 1.05x
        thresholds = {f"{int(r*100)}%_of_final": final_level / r
                      for r in THRESHOLD_RATIOS}
    else:
        mode = "above"
        final_level = curve_baseline[-FINAL_WINDOW:].mean()
        thresholds = {f"{int(r*100)}%_of_final": final_level * r
                      for r in THRESHOLD_RATIOS}

    print(f"\n[{metric_name}]  baseline final level = {final_level:.4f}")

    for tname, thr in thresholds.items():
        s_ours = steps_to_threshold(curve_ours,     thr, STABLE_WINDOW, mode)
        s_base = steps_to_threshold(curve_baseline, thr, STABLE_WINDOW, mode)

        if s_ours is None or s_base is None:
            print(f"  threshold={thr:.4f} ({tname}): "
                  f"CNCa {'N/A' if s_ours is None else s_ours} | "
                  f"LKP-GAT {'N/A' if s_base is None else s_base}  -> 无法计算")
            continue

        if s_base == 0:
            reduce_pct = 0.0
            speedup    = float('inf')
        else:
            reduce_pct = (s_base - s_ours) / s_base * 100
            speedup    = s_base / max(s_ours, 1)

        print(f"  threshold={thr:.4f} ({tname}): "
              f"CNCa={s_ours:4d} steps | LKP-GAT={s_base:4d} steps  "
              f"→ 减少 {reduce_pct:+.1f}%  ({speedup:.2f}× faster)")

        eff_results[(metric, tname)] = {
            "threshold":   thr,
            "ours_steps":  s_ours,
            "base_steps":  s_base,
            "reduce_pct":  reduce_pct,
            "speedup":     speedup,
        }

print("\n" + "=" * 70)
print("  可用于论文的句子（示例）")
print("=" * 70)

# 挑最有代表性的两个数字：F1 在 90% 阈值、Accuracy 在 95% 阈值
for (metric, tname), r in eff_results.items():
    if (metric, tname) in [("f1", "90%_of_final"), ("accuracy", "95%_of_final")]:
        metric_disp = {"f1": "F1 score", "accuracy": "accuracy"}[metric]
        ratio = tname.split("_")[0]
        print(
            f"\n  • On the {feature_display_name(TARGET_FEATURE)} feature, "
            f"CNCa reaches {ratio} of LKP-GAT's final {metric_disp} "
            f"in only {r['ours_steps']} training steps, "
            f"versus {r['base_steps']} steps for LKP-GAT — "
            f"a {r['reduce_pct']:.1f}% reduction "
            f"({r['speedup']:.2f}× speedup in convergence)."
        )
print()


# ============================================================
#  Final Performance 分析 —— Local Preference (training curve 用)
#  用"训练末期均值"比较 CNCa vs LKP-GAT 的最终水平
# ============================================================

LP_FEATURE = "predicate_bgp_route_arg2"   # Local Preference
FINAL_WIN  = 100                          # 取最后 100 步的均值

print("=" * 70)
print("  Final Performance Analysis : Local Preference (训练末期)")
print("=" * 70)

lp_final_results = {}

for metric, metric_name in zip(METRICS, METRIC_NAMES):
    curve_ours     = get_mean_curve(OURS_ALGO,     LP_FEATURE, metric)
    curve_baseline = get_mean_curve(BASELINE_ALGO, LP_FEATURE, metric)

    if curve_ours is None or curve_baseline is None:
        print(f"\n[{metric_name}] 数据缺失，跳过")
        continue

    L = min(len(curve_ours), len(curve_baseline))
    final_ours = curve_ours[-FINAL_WIN:].mean()
    final_base = curve_baseline[-FINAL_WIN:].mean()

    # 绝对差（百分点）和相对提升（百分比）
    abs_diff = (final_ours - final_base) * 100  # 百分点 (pp)
    if final_base > 1e-6:
        rel_improve = (final_ours - final_base) / final_base * 100
    else:
        rel_improve = float('inf')

    print(f"\n[{metric_name}]")
    print(f"  CNCa final    = {final_ours:.4f}  ({final_ours*100:.2f}%)")
    print(f"  LKP-GAT final = {final_base:.4f}  ({final_base*100:.2f}%)")
    if metric == "loss":
        # loss 越小越好，用下降比例
        if final_base > 1e-6:
            loss_drop_pct = (final_base - final_ours) / final_base * 100
        else:
            loss_drop_pct = float('inf')
        print(f"  → Loss 降低 {loss_drop_pct:.1f}% (相对 baseline)")
        lp_final_results[metric] = {
            "ours": final_ours, "base": final_base,
            "abs_diff_pp": abs_diff, "loss_drop_pct": loss_drop_pct,
        }
    else:
        print(f"  → 绝对提升 +{abs_diff:.2f} pp (percentage points)")
        print(f"  → 相对提升 +{rel_improve:.1f}%  (相对 baseline)")
        lp_final_results[metric] = {
            "ours": final_ours, "base": final_base,
            "abs_diff_pp": abs_diff, "rel_improve": rel_improve,
        }

print()


# ============================================================
#  Bar 图百分比分析 —— Local Preference, 两个 bar 图 × 3 个 test scale
#  Fig 3: test_40_by_train_40   (同 error rate 的拓扑泛化)
#  Fig 4: test_20_by_train_40   (跨 error rate 的难度泛化)
# ============================================================

BAR_FEATURE_KEY = "BGP Local Pref"   # CSV 里的 Feature 列名称
BAR_OURS_NAME   = "GATv2-NUM"        # CSV 里的算法名（注意是短横线）
BAR_BASE_NAME   = "GAT-LKP"

BAR_FIGURES = [
    {
        "fig_name":      "Fig 3 — test@40% by train@40%  (same error rate, cross-topology)",
        "short_name":    "test40_by_train40",
        "datasets": [
            {"label": "Baseline",     "csv": "results_test_40_by_train_40/Baseline/test_results_table.csv"},
            {"label": "Larger-Scale", "csv": "results_test_40_by_train_40/Larger-Scale/test_results_table.csv"},
            {"label": "Real-World",   "csv": "results_test_40_by_train_40/Real-World/test_results_table.csv"},
        ],
    },
    {
        "fig_name":      "Fig 4 — test@20% by train@40%  (cross error rate, cross-topology)",
        "short_name":    "test20_by_train40",
        "datasets": [
            {"label": "Baseline",     "csv": "results_test_20_by_train_40/Baseline/test_results_table.csv"},
            {"label": "Larger-Scale", "csv": "results_test_20_by_train_40/Larger-Scale/test_results_table.csv"},
            {"label": "Real-World",   "csv": "results_test_20_by_train_40/Real-World/test_results_table.csv"},
        ],
    },
]

# 嵌套存储：{fig_short_name: {dataset_label: {metric: {...}}}}
all_bar_results = {}

for fig_cfg in BAR_FIGURES:
    print("=" * 70)
    print(f"  Bar-Chart Analysis : {fig_cfg['fig_name']}")
    print(f"  Feature: Local Preference,  CNCa vs LKP-GAT")
    print("=" * 70)

    fig_results = {}

    for ds in fig_cfg["datasets"]:
        label, csv_path = ds["label"], ds["csv"]
        if not os.path.exists(csv_path):
            print(f"\n[{label}] CSV 不存在: {csv_path}")
            continue

        print(f"\n── {label} ──────────────────────────────────────────────")
        fig_results[label] = {}

        for metric in METRICS_TEST:   # ["F1", "Acc"]
            try:
                d = load_csv_test(csv_path, metric,
                                  [BAR_OURS_NAME, BAR_BASE_NAME],
                                  [BAR_FEATURE_KEY])
                ours_mean, ours_std = d[BAR_OURS_NAME][BAR_FEATURE_KEY]
                base_mean, base_std = d[BAR_BASE_NAME][BAR_FEATURE_KEY]
            except (KeyError, ValueError) as e:
                print(f"  [{metric}] 读取失败: {e}")
                continue

            if np.isnan(ours_mean) or np.isnan(base_mean):
                print(f"  [{metric}] 数据为 N/A")
                continue

            abs_diff = ours_mean - base_mean   # CSV 里单位是 %，直接减就是 pp
            if base_mean > 1e-6:
                rel_improve = (ours_mean - base_mean) / base_mean * 100
            else:
                rel_improve = float('inf')

            # 判断差距是否"超过一个标准差"——粗略的显著性指示
            # 用 max(两个 std) 作为参考尺度
            std_scale = max(ours_std, base_std) if not (np.isnan(ours_std) or np.isnan(base_std)) else 0
            if std_scale > 0:
                gap_in_stds = abs(abs_diff) / std_scale
                sig_flag = f"gap/max_std = {gap_in_stds:4.2f}"
                if gap_in_stds < 1.0:
                    sig_flag += "  ⚠️ 差距<1σ，不显著"
                elif gap_in_stds < 2.0:
                    sig_flag += "  △ 差距>1σ，弱显著"
                else:
                    sig_flag += "  ✅ 差距>2σ，显著"
            else:
                sig_flag = "std 缺失"

            print(f"  [{metric}]  "
                  f"CNCa={ours_mean:6.2f}±{ours_std:5.2f}%  |  "
                  f"LKP-GAT={base_mean:6.2f}±{base_std:5.2f}%  →  "
                  f"{abs_diff:+6.2f} pp  ({rel_improve:+6.1f}%)   {sig_flag}")

            fig_results[label][metric] = {
                "ours": ours_mean, "ours_std": ours_std,
                "base": base_mean, "base_std": base_std,
                "abs_diff_pp": abs_diff, "rel_improve": rel_improve,
                "gap_in_stds": gap_in_stds if std_scale > 0 else None,
            }

    all_bar_results[fig_cfg["short_name"]] = fig_results
    print()


# # ============================================================
# #  总结：可直接用于论文的句子
# # ============================================================

# print("=" * 70)
# print("  📝 论文可用句子汇总")
# print("=" * 70)

# # ---- 句子 1: Training efficiency (OSPF Weight) ----
# if ("f1", "90%_of_final") in eff_results:
#     r = eff_results[("f1", "90%_of_final")]
#     print("\n[Story 1 — Training Efficiency, 放 training curve 图附近]")
#     print(
#         f"  On the OSPF Weight feature, CNCa reaches 90% of LKP-GAT's "
#         f"final F1 score in only {r['ours_steps']} training steps, "
#         f"compared with {r['base_steps']} steps required by LKP-GAT — "
#         f"a {r['reduce_pct']:.1f}% reduction in training cost "
#         f"({r['speedup']:.2f}× speedup)."
#     )

# # ---- 句子 2: Final performance (Local Preference, training curve) ----
# if "accuracy" in lp_final_results and "f1" in lp_final_results:
#     acc = lp_final_results["accuracy"]
#     f1  = lp_final_results["f1"]
#     print("\n[Story 2 — Final Performance, 放 training curve 图附近]")
#     print(
#         f"  On the more challenging Local Preference feature, LKP-GAT "
#         f"fails to converge effectively (final F1 = {f1['base']*100:.1f}%, "
#         f"accuracy = {acc['base']*100:.1f}%), whereas CNCa attains "
#         f"F1 = {f1['ours']*100:.1f}% and accuracy = {acc['ours']*100:.1f}% — "
#         f"an absolute improvement of +{acc['abs_diff_pp']:.1f} pp in accuracy "
#         f"and +{f1['abs_diff_pp']:.1f} pp in F1."
#     )

# # ---- 句子 3 & 4: 两个 bar 图 ----
# bar_story_titles = {
#     "test40_by_train40": "Story 3 — Cross-topology generalization (Fig 3, same error rate)",
#     "test20_by_train40": "Story 4 — Cross-error-rate generalization (Fig 4)",
# }

# for short_name, fig_results in all_bar_results.items():
#     if not fig_results:
#         continue
#     title = bar_story_titles.get(short_name, short_name)
#     print(f"\n[{title}]")

#     # --- Fig 3 特殊处理：只讲 Real-World（供摘要使用）---
#     if short_name == "test40_by_train40":
#         rw = fig_results.get("Real-World", {})
#         if "Acc" in rw and "F1" in rw:
#             acc, f1 = rw["Acc"], rw["F1"]
#             print("  ▶ 摘要版 (只讲 Real-World，简洁有力):")
#             print(
#                 f"    On the real-world topology, CNCa achieves "
#                 f"{acc['ours']:.1f}% accuracy and {f1['ours']:.1f}% F1 score "
#                 f"on Local Preference, outperforming the LKP-GAT baseline "
#                 f"({acc['base']:.1f}% accuracy, {f1['base']:.1f}% F1) by "
#                 f"+{acc['abs_diff_pp']:.1f} pp and +{f1['abs_diff_pp']:.1f} pp, "
#                 f"respectively."
#             )
#             print("\n  ▶ 摘要版 (更紧凑，只用一个指标):")
#             # 挑差距更大的那个指标进摘要
#             if acc['abs_diff_pp'] >= f1['abs_diff_pp']:
#                 print(
#                     f"    On the real-world topology, CNCa improves Local "
#                     f"Preference prediction accuracy from {acc['base']:.1f}% "
#                     f"to {acc['ours']:.1f}% (+{acc['abs_diff_pp']:.1f} pp, "
#                     f"a {acc['rel_improve']:.1f}% relative gain) over the "
#                     f"LKP-GAT baseline."
#                 )
#             else:
#                 print(
#                     f"    On the real-world topology, CNCa improves Local "
#                     f"Preference F1 score from {f1['base']:.1f}% to "
#                     f"{f1['ours']:.1f}% (+{f1['abs_diff_pp']:.1f} pp, "
#                     f"a {f1['rel_improve']:.1f}% relative gain) over the "
#                     f"LKP-GAT baseline."
#                 )
#             print("\n  ▶ 正文版 (可以提一下其他 scale 以体现一致性):")
#             acc_parts = []
#             for ds_label, metric_map in fig_results.items():
#                 if "Acc" in metric_map:
#                     acc_parts.append(f"{ds_label} (+{metric_map['Acc']['abs_diff_pp']:.1f} pp)")
#             if acc_parts:
#                 print(
#                     f"    CNCa consistently outperforms LKP-GAT across all "
#                     f"three test topologies — {', '.join(acc_parts)} in "
#                     f"accuracy — with the largest gain observed on the "
#                     f"Real-World setting."
#                 )
#         else:
#             print("  [Real-World 数据缺失，无法生成摘要句子]")
#         continue

#     # --- Fig 4 保持原来的"综合 + 亮点"逻辑 ---
#     acc_parts = []
#     for ds_label, metric_map in fig_results.items():
#         if "Acc" in metric_map:
#             acc_parts.append(f"{ds_label} (+{metric_map['Acc']['abs_diff_pp']:.1f} pp)")
#     if acc_parts:
#         print(
#             f"  On Local Preference, CNCa consistently outperforms LKP-GAT "
#             f"across all three test scales: {', '.join(acc_parts)} in accuracy."
#         )

#     best_ds, best_r = None, None
#     for ds_label, metric_map in fig_results.items():
#         if "Acc" in metric_map:
#             if best_r is None or metric_map["Acc"]["abs_diff_pp"] > best_r["abs_diff_pp"]:
#                 best_ds, best_r = ds_label, metric_map["Acc"]
#     if best_ds and best_r:
#         print(
#             f"  In particular, on the {best_ds} setting, CNCa achieves "
#             f"{best_r['ours']:.1f}% accuracy versus LKP-GAT's {best_r['base']:.1f}% — "
#             f"an absolute gain of +{best_r['abs_diff_pp']:.1f} pp "
#             f"(relative improvement of {best_r['rel_improve']:.1f}%)."
#         )

# # ---- 跨 Fig 3 / Fig 4 的综合对比 ----
# if len(all_bar_results) == 2:
#     r3 = all_bar_results.get("test40_by_train40", {})
#     r4 = all_bar_results.get("test20_by_train40", {})
#     if r3 and r4:
#         # 挑一个共同的 dataset（比如 Real-World）做对比
#         common = set(r3.keys()) & set(r4.keys())
#         if "Real-World" in common:
#             key = "Real-World"
#         elif common:
#             key = sorted(common)[0]
#         else:
#             key = None

#         if key and "Acc" in r3.get(key, {}) and "Acc" in r4.get(key, {}):
#             print(f"\n[Cross-figure summary — 把 Fig 3 和 Fig 4 串起来的句子]")
#             print(
#                 f"  Across both generalization settings — same error rate "
#                 f"(Fig. 3) and cross error rate (Fig. 4) — CNCa maintains a "
#                 f"consistent advantage on {key}: "
#                 f"+{r3[key]['Acc']['abs_diff_pp']:.1f} pp accuracy in the "
#                 f"former and +{r4[key]['Acc']['abs_diff_pp']:.1f} pp in the "
#                 f"latter, demonstrating that the improvement transfers across "
#                 f"both topological and error-rate shifts."
#             )
# print()