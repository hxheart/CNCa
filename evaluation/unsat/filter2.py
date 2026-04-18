import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='Generate topology-zoo-based synthesis problem instances.')
parser.add_argument('dataset', type=str)
parser.add_argument('dataset_output', type=str)

args = parser.parse_args()

all_topo_ids = set()
topo_ids_with_unsat = set()

results = []

ids = [6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23]

for f in os.listdir(args.dataset):
    comps = f.split("-")
    id = int(comps[1][1:])
    if not id in ids: 
        continue
    shutil.copy(args.dataset + "/" + f, args.dataset_output + "/" + f)

    if ".graphml" in f:
        results.append(f)

#     suffixes = [".graphml", ".spec.json", ".logic"]
#     for s in suffixes:
#         r = "ospf-n{}-unsatsample{}".format(id, s)
#         src = os.path.join(args.dataset, r + s)
#         dst = os.path.join(args.dataset_output, r + s)
#         os.makedirs(os.path.dirname(dst), exist_ok=True)
#         shutil.copy(src, dst)
    
print("Unsat Dataset contains {} samples".format(len(results)))