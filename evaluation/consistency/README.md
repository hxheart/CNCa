# Synthesis Quality Evaluation

This folder contains the required files to reproduce our synthesis quality evaluation.

## Files

* `eval_consistency.py` Runs a provided synthesizer model for a given dataset of evaluation samples. Example usage:
    ```
    EXP_ID=eval-64-bgp-reqs-2-4shot python3 eval_consistency.py ../../trained-model/bgp-64-pred-6layers-model-epoch2800.pt --num-samples=5 --random=1 --num-shots=4 --num-iterations=8 --dataset=./dataset-ported/bgp-qlty-reqs-2 --cpu=1
    ```

* `results/` The raw data results of our experiments on synthesis quality, as well as notebooks to format the data.

## Datasets

The datasets used for this comparison are located in `dataset-ported/`. 