#!/bin/bash
set -o xtrace

TIMEOUT=1500
DATASET="bgp-reqs-"

PYTHONPATH=synet-plus python2 bgp_synthesis.py outdir --dataset ../dataset/${DATASET}2 --perf-prefix=${DATASET}2 --timeout=${TIMEOUT}
PYTHONPATH=synet-plus python2 bgp_synthesis.py outdir --dataset ../dataset/${DATASET}8 --perf-prefix=${DATASET}8 --timeout=${TIMEOUT}
PYTHONPATH=synet-plus python2 bgp_synthesis.py outdir --dataset ../dataset/${DATASET}16 --perf-prefix=${DATASET}16 --timeout=${TIMEOUT}
