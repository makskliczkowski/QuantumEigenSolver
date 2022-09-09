#!/bin/sh
start=$1
end=$2

tmp=($(seq $start 1 $end))
for i in ${tmp[@]};do
    scancel ${i}
done