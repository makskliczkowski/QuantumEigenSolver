Lsu2=$1
Lnsu2=$2
BC=$3
S=$4

hx=0.2
hz=0.8
J2=1.0


echo "run ./skrypt_run_symmetries_single_param_all_sym.sh ${Lnsu2} ${hx} ${hz} ${J2} 0.5 ${BC} ${S}"
bash ./skrypt_run_symmetries_single_param_all_sym.sh ${Lnsu2} ${hx} ${hz} ${J2} 0.5 ${BC} ${S}
#echo "run ./skrypt_run_symmetries_single_param_all_sym.sh ${Lsu2} 0 0.0 2.0 0.0 ${BC} ${S}"
#bash ./skrypt_run_symmetries_single_param_all_sym.sh ${Lsu2} 0 0.0 2.0 0.0 ${BC} ${S}
