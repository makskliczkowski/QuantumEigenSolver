#!/bin/bash

start=$1
end=$2

for i in $(seq $start 1 $end)
do
    scancel $i
done
