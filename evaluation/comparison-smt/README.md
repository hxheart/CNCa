# NetComplete comparison problem instances

This folder contains the files required to reproduce the results of our synthesis time comparison of our neural synthesizer model and the SMT-based synthesis tool NetComplete.

## Files

* `eval_synthesis_time.py` Evaluation script which runs our model on a dataset of problem instances and saves the synthesis time and consistency as a file `$PROT-reqs-$N-result-...`. Example usage:

    ```
    EXP_ID=bgp-reqs-2 python3 eval_synthesis_time.py ../../trained-model/bgp-64-pred-6layers-model-epoch2800.pt --num-samples=10 --random=1 --num-shots=4 --num-iterations=8 --dataset=./dataset-ported/bgp-reqs-2 --perf-prefix=bgp-reqs-2  --cpu=1
    ```

* `generate_synthesis_tasks.py` Generates the problems instances used for synthesis time comparison based on the topologies located in `../dataset`. Example usage:

    ```
    python3 generate_synthesis_tasks.py ${NUM_REQS} bgp-reqs-${NUM_REQS}-test ../datasets/ --protocol=bgp
    ```

* `results/*.csv` Raw data results of running our trained synthesizer model `bgp-64-pred-6layers-model-epoch2800.pt` on the problem instances in `../dataset`

* `netcomplete/` This folders contains the code to run NetComplete on the same evaluation dataset as well as the results we obtained during our experiments.

* `comparison-bgp.ipynb` and `comparison-ospf.ipynb` are Jupyter notebook in which we aggregate the raw data of this evaluation into the tables included in the paper.

## Datasets

The datasets used for this comparison are located in `dataset/` and `dataset-ported/`. Both folders contain the same set of problem instances, where `-ported` refers to the variant that is compatible with the latest PyG.