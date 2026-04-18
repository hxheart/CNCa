## Unsat Experiments

This directory contains the files used to experiment with synthesis for unsatisfiable specifications.

To run the synthesizer model on a provided datsets of unsatifiable specification, you can run the following command in the `../consistency` directory:

```
EXP_ID=dataset-unsat-16 python3 eval_consistency.py --dataset ../unsat/filtered2/dataset-unsat-16 ../../trained-model/bgp-64-pred-6layers-model-epoch2800.pt --random 1 --protocol ospf --num-samples 5 --num-shots 4 --cpu 1
```

This will produce the result file as identifiable by the provided `EXP_ID` in the `../consistency/` working directory.