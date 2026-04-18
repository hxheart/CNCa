import sys
import os

ids = []

for f in os.listdir(sys.argv[1]):
    if not f.endswith(".graphml"): continue
    comps = f.split("-")
    id = comps[1][1:]
    ids.append(int(id))

print(sorted(ids))
print(len(ids))