import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='Generate topology-zoo-based synthesis problem instances.')
parser.add_argument('dataset', type=str)
parser.add_argument('unsat_results_file', type=str)
parser.add_argument('dataset_output', type=str)

args = parser.parse_args()

with open(args.unsat_results_file, "r") as f:
    lines = f.read().split("\n")

last_topo_id = -1

all_topo_ids = set()
topo_ids_with_unsat = set()

results = []

for l in lines:
    if l.strip() == "": 
        continue
    path,is_unsat = l.split(";")
    # get filename of path
    topo_id = int(path.split("/")[-1].split("-")[1][1:])
    sample_id = int(path.split("/")[-1].split("-")[-1][len("unsatsample"):])
    is_unsat = is_unsat == "True"

    all_topo_ids.add(topo_id)

    if not is_unsat: continue    
    if topo_id in topo_ids_with_unsat: continue
    topo_ids_with_unsat.add(topo_id)
    
    last_topo_id = topo_id
    results.append(f"ospf-n{topo_id}-unsatsample{sample_id}")

print(len(results), len(lines))

for topo_id in all_topo_ids - topo_ids_with_unsat:
    print("No unsat sample for topology {}".format(topo_id))

for r in results:
    suffixes = [".graphml", ".spec.json", ".logic"]
    for s in suffixes:
        src = os.path.join(args.dataset, r + s)
        dst = os.path.join(args.dataset_output, r + s)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
print(results)
print("Unsat Dataset contains {} samples".format(len(results)))