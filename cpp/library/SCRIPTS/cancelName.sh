#!/bin/bash
NAME=$1
squeue -u ${USER} | grep ${NAME} | awk '{print $3}' # test to make sure.
# squeue -u ${USER} | grep ${NAME} | awk '{print $1}' | xargs -n 1 scancel