VALNONE=-1000
L=$1
dlt1=$2
eta1=$3
SYMS=$4
TIM=$5
MEM=$6
CPU=$7
FUN=$8
# u1=$4
# px=$5
# py=$6
# pz=$7
# r=$8
# k=$9

cd /home/klimak97/CODES/QuantumEigenSolver/cpp/library/

SSYYMS=$(tr -d ' ' <<< "$SYMS")

source /usr/local/sbin/modules.sh
module load intel/2022b
module load HDF5

./qsolver.o -fun ${FUN} -mod 1 -bc 0 -l 0 -d 1 -Lx ${L} -Ly 1 -Lz 1 -J1 1.0 -J2 0 -hx 0 -hz 0 -eta1 ${eta1} -eta2 0 -dlt1 ${dlt1} -S 1 ${SYMS} -th ${CPU} -dir SUSY_MISSING_NO/ >& ./LOG/log_${a}.txt

# -x ${r} -px ${px} -py ${py} -pz ${pz} -k ${k} -U1 ${u1}
# echo "finished"