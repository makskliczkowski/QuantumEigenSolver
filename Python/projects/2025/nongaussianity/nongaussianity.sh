#!bin/bash
# first argument is system size

lx=$1

python nongaussianity.py --lx=$lx --hamil=SYK2 --dtype=complex --la=0.5 --occ=0.5 --nreal_comb=500 --nreal=500 --gammas=1,2,3,4,5,6,7,8,9,10,0.5,*2,*3
