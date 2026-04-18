#!/bin/bash
set -o xtrace

TIMEOUT=129
DATASET=$1

PYTHONPATH=synet-plus python2 ospf_synthesis.py outdir --dataset ../../unsat/$DATASET --perf-prefix=unsat-${DATASET} --timeout=${TIMEOUT} --save-unsat-results