ecli add consistency-bgp-reqs-2 EXP_ID=eval-64-bgp-reqs-2-4shot python3 eval_consistency.py ../../trained-model/bgp-64-pred-6layers-model-epoch2800.pt --num-samples=5 --random=1 --num-shots=4 --dataset ./dataset-ported/bgp-qlty-reqs-2 --cpu=1

ecli add consistency-bgp-reqs-8 EXP_ID=eval-64-bgp-reqs-8-4shot python3 eval_consistency.py ../../trained-model/bgp-64-pred-6layers-model-epoch2800.pt --num-samples=5 --random=1 --num-shots=4 --dataset ./dataset-ported/bgp-qlty-reqs-8 --cpu=1

ecli add consistency-bgp-reqs-16 EXP_ID=eval-64-bgp-reqs-16-4shot python3 eval_consistency.py ../../trained-model/bgp-64-pred-6layers-model-epoch2800.pt --num-samples=5 --random=1 --num-shots=4 --dataset ./dataset-ported/bgp-qlty-reqs-16 --cpu=1

ecli add consistency-bgp-reqs-16-oneshot EXP_ID=consistency-bgp-reqs-16-oneshot python3 eval_consistency.py ../../trained-model/bgp-64-pred-6layers-model-epoch2800.pt --num-samples=5 --random=1 --num-shots=4 --dataset ./dataset-ported/bgp-qlty-reqs-16 --cpu=1

ecli add num-samples-bgp-reqs-16 EXP_ID=eval-64-bgp-reqs-16-4shot-manysamples python3 eval_consistency.py ../../trained-model/bgp-64-pred-6layers-model-epoch2800.pt --num-samples=20 --random=1 --num-shots=4 --dataset ./dataset-ported/bgp-qlty-reqs-16 --cpu=1

ecli add consistency-bgp-reqs-16-1shot EXP_ID=eval-64-bgp-reqs-16-1shot python3 eval_consistency.py ../../trained-model/bgp-64-pred-6layers-model-epoch2800.pt --num-samples=5 --random=1 --num-shots=1 --dataset ./dataset-ported/bgp-qlty-reqs-16 --cpu=1