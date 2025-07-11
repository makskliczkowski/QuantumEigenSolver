#! /usr/bin/env bash

cd "$(dirname "$0")"
# make sure script is executable
echo "Running evolution script in $(pwd)"
echo "Files in this directory:"
ls -l

chmod +x ./evolution.py

# run with:
# 1) 
# ../../2025_um_evolution/data_big_times        \
python ./evolution.py      \
    data_big_times          \
    0.7                     \
    0.02                    \
    10                      \
    50                     \
    13                      \
    14                      \
    1                       \
    100000                  \
    4                       \
    16