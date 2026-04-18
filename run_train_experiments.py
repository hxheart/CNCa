import subprocess
import sys
import time
import os
import pandas as pd
from pathlib import Path

# ============================================================
#  实验配置
# ============================================================

MODELS = [
    # ("train_GATv2_NUM_0326.py", "GATv2_NUM"),
    # ("train_GATv2_LKP_0326.py", "GATv2_LKP"),
    ("train_GAT_LKP_0417.py",   "GAT_LKP"),
    # ("train_GAT_NUM_0326.py",   "GAT_NUM"),
    # ("train_GCN_NUM_0329.py",   "GCN_NUM"),
]

NUM_RUNS = 10

# ============================================================
#  运行
# ============================================================

total      = len(MODELS) * NUM_RUNS
done       = 0
failed     = []
start_time = time.time()


for model_script, algo_name in MODELS:
    # 数磁盘上已有几个 run，从下一个开始
    results_dir = Path(f"results_train/{algo_name}")
    if results_dir.exists():
        existing = [d for d in results_dir.iterdir()
                    if d.is_dir() and d.name.startswith(algo_name + "_")
                    and d.name.split("_")[-1].isdigit()]
        start_run = max([int(d.name.split("_")[-1]) for d in existing], default=0) + 1
    else:
        start_run = 1

    for run in range(start_run, start_run + NUM_RUNS):
        done += 1
        elapsed = time.time() - start_time
        avg     = elapsed / done if done > 1 else 0
        eta     = avg * (total - done)

        print(f"\n{'='*60}")
        print(f"[{done}/{total}]  model={algo_name}  run={run}/{NUM_RUNS}")
        print(f"  elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")
        print(f"{'='*60}")

        result = subprocess.run(
            [sys.executable, model_script, "--run-id", str(run)],
        )

        if result.returncode != 0:
            tag = f"{algo_name} | run {run}"
            failed.append(tag)
            print(f"  FAILED: {tag}")
        else:
            print(f"  OK: {algo_name} | run {run}")

# ============================================================
#  汇总：对每个 model 的多次 run 取平均，输出一个 summary CSV
# ============================================================

print(f"\n{'='*60}")
print("汇总各 model 的多次实验结果...")

for _, algo_name in MODELS:
    results_dir = Path(f"results_train/{algo_name}")
    if not results_dir.exists():
        print(f"  [SKIP] {algo_name}: 目录不存在")
        continue

    dfs = []
    for run in range(1, NUM_RUNS + 1):
        run_dir  = results_dir / f"{algo_name}_{run}"          # ← GATv2_NUM_1
        csv_path = run_dir / f"metrics_{algo_name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["run"] = run
            dfs.append(df)
        else:
            print(f"  [WARN] 找不到: {csv_path}")

    if not dfs:
        print(f"  [SKIP] {algo_name}: 没有可用的 CSV")
        continue

    combined = pd.concat(dfs)

    summary = combined.groupby("step")[["loss", "accuracy", "f1"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()

    # ← summary 放进最后一个 run 的目录里，不放在根层
    summary_path = results_dir / f"{algo_name}_{NUM_RUNS}" / f"summary_{algo_name}_N{NUM_RUNS}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  >>> 汇总 CSV 已保存: {summary_path}")

# ============================================================
#  最终报告
# ============================================================

print(f"\n{'='*60}")
print(f"完成  {done}/{total}  共用时 {(time.time()-start_time)/60:.1f} 分钟")

if failed:
    print(f"\n失败 {len(failed)} 次:")
    for f in failed:
        print(f"  - {f}")
else:
    print("所有实验全部成功")
print(f"{'='*60}")