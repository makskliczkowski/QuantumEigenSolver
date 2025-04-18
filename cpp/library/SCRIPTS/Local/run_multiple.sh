#!/bin/bash

# Usage: ./run_multiple.sh <number_of_runs>

if [ -z "$1" ]; then
    echo "Usage: $0 <number_of_runs>"
    exit 1
fi

start_run=$1
end_run=$2
num_runs=$((end_run - start_run + 1))

for i in $(seq 1 $num_runs); do
    idx=$((start_run + i - 1))
    echo "Running iteration $i with index $idx"
    # Create a unique log file for each run
    log_file="local_${idx}.log"
    nohup ../../build/qsolver -f ../../INPUTS/input_test.ini > "$log_file" 2>&1 &
    echo "Started run $i, logging to $log_file"
    echo "Run $i completed."
done