#!/bin/bash

# Usage: ./run_multiple.sh <number_of_runs>

if [ -z "$1" ]; then
    echo "Usage: $0 <number_of_runs>"
    exit 1
fi

num_runs=$1

for i in $(seq 1 $num_runs); do
    nohup ../../build/qsolver -f ../../INPUTS/input_test.ini > "local_${i}.log" 2>&1 &
    echo "Started run $i, logging to local_${i}.log"
done