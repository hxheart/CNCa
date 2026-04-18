#!/bin/bash
set -o xtrace

TIMEOUT=1500
DATASET="ospf-reqs-"

PYTHONPATH=synet-plus python2 ospf_synthesis.py outdir --dataset ../dataset/${DATASET}2 --perf-prefix=${DATASET}2 --timeout=${TIMEOUT}
PYTHONPATH=synet-plus python2 ospf_synthesis.py outdir --dataset ../dataset/${DATASET}8 --perf-prefix=${DATASET}8 --timeout=${TIMEOUT}
PYTHONPATH=synet-plus python2 ospf_synthesis.py outdir --dataset ../dataset/${DATASET}16 --perf-prefix=${DATASET}16 --timeout=${TIMEOUT}