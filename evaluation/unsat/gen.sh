NUM_UNSAT=$1

mkdir -p dataset-unsat-$NUM_UNSAT
mkdir -p filtered/dataset-unsat-$NUM_UNSAT
mkdir -p filtered2/dataset-unsat-$NUM_UNSAT

#python3 generate_unsat_synthesis_tasks.py 16 dataset-unsat-$NUM_UNSAT ../dataset-ported --num_unsat_req 12

python3 filter_unsat_samples.py ./dataset-unsat-$NUM_UNSAT ../comparison-smt/netcomplete/unsat-dataset-unsat-$NUM_UNSAT-unsat-results.csv ./filtered/dataset-unsat-$NUM_UNSAT
python3 filter2.py filtered/dataset-unsat-$NUM_UNSAT filtered2/dataset-unsat-$NUM_UNSAT