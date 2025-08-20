#!/bin/bash
hamil=FreeFermions
# hamil=SYK2
maxjobs=6

mkdir -p logs

for lx in 10 12 14 20 100 200 500 1000 5000; do
    echo "[`date '+%H:%M:%S'`] starting lx=$lx ..."
    python nongaussianity.py --lx=$lx --hamil=$hamil --dtype=complex \
        --la=0.5 --occ=0.5 --nreal_comb=500 --nreal=500 \
        --gammas=1,2,3,4,5,6,7,8,9,10,0.5,*2,*3 \
        >logs/lx_${hamil}_${lx}.out 2>&1 &
    
    # if too many jobs, wait for one to finish
    while (( $(jobs -p | wc -l) >= maxjobs )); do
        sleep 1
    done
done
wait