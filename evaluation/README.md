# Evaluation

This directory contains the scripts used in our evaluation:

- `comparison-smt/` contains the scripts to measure synthesis time of our models as compared to SMT-based synthesis tool NetComplete.

- `consistency/` contains the scripts and data of our main evaluation with respect to synthesis quality.

- `unsat/` contains the scripts and data of our evaluation with respect unsatisfiable OSPF synthesis tasks.

- `beam/` contains unreported, so-far unsuccessful and preliminary experiments regarding beam search decoding.

Please see each of the individual directories for more detailed documentation of the respective scripts and data.

## Evaluation Datasets

To generate our datasets, we rely on `./generate-topologies.py` to first choose a set of real-world topologies from the Topology Zoo, saving them to a folder `networks/`. We then generate corresonding BGP/OSPF synthesis tasks using `generate-bgp-instances.py`, resulting in folder `datasets`.